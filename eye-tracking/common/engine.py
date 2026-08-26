"""
Threaded realtime recognition: snappy boxes + fast confirmed IDs.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import List, Optional

from .face_detector import FaceDetector
from .face_recognition import FaceRecognizer
from .profiles import RuntimeProfile
from .quality import assess_face_quality, is_frontal_enough
from .tracking import FaceTracker, TrackedFace
from .types import FaceSample


@dataclass
class EngineStats:
    """Live telemetry for the HUD."""

    fps: float = 0.0
    infer_ms: float = 0.0
    det_ms: float = 0.0
    n_faces: int = 0
    profile: str = "balanced"


class RealtimeRecognitionEngine:
    """
    Main thread: detection every frame (boxes).
    Worker: recognition with cooldown (avoids multi-second CPU pile-ups).

    By default only the largest face(s) are kept — ideal for access control
    and avoids labeling every tiny face in background photos.
    """

    def __init__(
        self,
        recognizer: FaceRecognizer,
        detector: FaceDetector,
        profile: RuntimeProfile,
        max_faces: int = 1,
        min_face_ratio: float = 0.08,
    ) -> None:
        self.recognizer = recognizer
        self.detector = detector
        self.profile = profile
        self.max_faces = max(1, int(max_faces))
        # Minimum face short-side / frame short-side (filters wall-photo faces).
        self.min_face_ratio = float(min_face_ratio)
        self.tracker = FaceTracker(
            iou_threshold=0.12,
            max_miss=10,
            bbox_alpha=0.9,
            confirm_hits=profile.confirm_hits,
            min_score=profile.match_threshold,
        )
        self._lock = threading.Lock()
        self._latest_frame = None
        self._tracks: List[TrackedFace] = []
        self._stats = EngineStats(profile=profile.name)
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._force_recognize = threading.Event()
        self._last_infer_t = 0.0
        self._min_infer_gap_s = 0.25

    def _select_primary_faces(self, detections: List[FaceSample], frame_shape) -> List[FaceSample]:
        """Keep the largest face(s) above a relative size threshold."""
        fh, fw = frame_shape[:2]
        frame_short = float(min(fh, fw))
        min_px = max(float(self.profile.min_face_px), self.min_face_ratio * frame_short)

        filtered = []
        for det in detections:
            short = float(min(det.bbox[2] - det.bbox[0], det.bbox[3] - det.bbox[1]))
            if short >= min_px:
                filtered.append((short, det))
        filtered.sort(key=lambda item: item[0], reverse=True)
        return [det for _, det in filtered[: self.max_faces]]

    def start(self) -> None:
        if self._worker is not None:
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._recognize_loop, name="face-recog", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        self._force_recognize.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None

    def submit_frame(self, frame) -> None:
        with self._lock:
            self._latest_frame = frame

    def update_from_frame(self, frame) -> List[TrackedFace]:
        t0 = time.time()
        det_side = min(480, self.profile.infer_size)
        detections = self.detector.detect(frame, max_side=det_side)
        detections = self._select_primary_faces(detections, frame.shape)
        det_ms = (time.time() - t0) * 1000.0

        with self._lock:
            self._tracks = self.tracker.update(detections)
            self._stats.det_ms = det_ms
            self._stats.n_faces = len(self._tracks)
            if any(tr.needs_id or tr.identity is None for tr in self._tracks):
                self._force_recognize.set()
            return list(self._tracks)

    def get_stats(self) -> EngineStats:
        with self._lock:
            return EngineStats(
                fps=self._stats.fps,
                infer_ms=self._stats.infer_ms,
                det_ms=self._stats.det_ms,
                n_faces=self._stats.n_faces,
                profile=self._stats.profile,
            )

    def _recognize_loop(self) -> None:
        every = max(1, int(self.profile.every))
        frame_i = 0
        while not self._stop.is_set():
            forced = self._force_recognize.is_set()
            if forced:
                self._force_recognize.clear()

            now = time.time()
            if (now - self._last_infer_t) < self._min_infer_gap_s:
                time.sleep(0.01)
                continue

            with self._lock:
                frame = None if self._latest_frame is None else self._latest_frame.copy()
                needs_fast = any(tr.needs_id or tr.identity is None for tr in self._tracks)

            if frame is None:
                time.sleep(0.004)
                continue

            frame_i += 1
            if not forced and not needs_fast and (frame_i % every != 0):
                time.sleep(0.002)
                continue

            t0 = time.time()
            # Fast ID path: no zoom unless face is small (handled inside analyze).
            faces = self.recognizer.analyze(
                frame,
                identify=True,
                max_side=self.profile.infer_size,
                refine_small=True,
                small_face_px=110,
            )
            faces = self._select_primary_faces(faces, frame.shape)
            cleaned: List[FaceSample] = []
            for face in faces:
                short = min(face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1])
                report = assess_face_quality(
                    frame,
                    face,
                    min_face_px=self.profile.min_face_px,
                    min_det_score=self.profile.min_det_score,
                    min_sharpness=25.0 if short < 100 else 35.0,
                )
                if not report.ok or not is_frontal_enough(face):
                    face.identity = None
                cleaned.append(face)

            with self._lock:
                self.tracker.apply_identities(cleaned)
                self._tracks = [tr for tr in self.tracker.tracks.values() if tr.miss == 0]
                self._stats.infer_ms = (time.time() - t0) * 1000.0
                self._stats.n_faces = len(self._tracks)
            self._last_infer_t = time.time()
