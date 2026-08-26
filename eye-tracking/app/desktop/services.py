"""Shared model / gallery helpers for the desktop app."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from common.engine import RealtimeRecognitionEngine
from common.face_detector import FaceDetector
from common.face_recognition import FaceGallery, FaceRecognizer
from common.profiles import RuntimeProfile, get_profile
from common.quality import assess_face_quality, is_frontal_enough

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GALLERY = _ROOT / "face-recognition" / "gallery"


class AppServices:
    """Lazy-loaded recognizer/detector shared across GUI pages."""

    def __init__(self) -> None:
        self.gallery_path = DEFAULT_GALLERY
        self.provider = "CPUExecutionProvider"
        self.profile_name = "balanced"
        self.camera_index = 0
        self.mirror = True
        self.max_faces = 1
        self.min_face_ratio = 0.10
        self.threshold_override: Optional[float] = None

        self._recognizer: Optional[FaceRecognizer] = None
        self._detector: Optional[FaceDetector] = None
        self._loaded_key: Optional[Tuple] = None

    @property
    def profile(self) -> RuntimeProfile:
        return get_profile(self.profile_name)

    def gallery(self) -> FaceGallery:
        g = FaceGallery(self.gallery_path)
        thr = self.effective_threshold()
        g.match_threshold = thr
        return g

    def effective_threshold(self) -> float:
        if self.threshold_override is not None:
            return float(self.threshold_override)
        return float(self.profile.match_threshold)

    def invalidate_models(self) -> None:
        self._recognizer = None
        self._detector = None
        self._loaded_key = None

    def ensure_models(self, *, det_size: Optional[int] = None) -> Tuple[FaceRecognizer, FaceDetector]:
        profile = self.profile
        gallery = self.gallery()
        key = (
            profile.name,
            profile.model_name,
            self.provider,
            str(self.gallery_path),
            self.effective_threshold(),
            det_size or profile.det_size,
        )
        if self._recognizer is not None and self._detector is not None and self._loaded_key == key:
            self._recognizer.gallery = gallery
            self._recognizer.match_threshold = self.effective_threshold()
            return self._recognizer, self._detector

        thr = self.effective_threshold()
        dsize = det_size or profile.det_size
        recognizer = FaceRecognizer(
            model_name=profile.model_name,
            provider=self.provider,
            min_score=profile.min_det_score,
            gallery=gallery,
            match_threshold=thr,
            match_margin=profile.match_margin,
            det_size=dsize,
        )
        detector = FaceDetector(
            backend="insightface",
            model_name=recognizer.model_name,
            provider=self.provider,
            min_score=profile.min_det_score,
            det_size=min(256, dsize),
        )
        self._recognizer = recognizer
        self._detector = detector
        self._loaded_key = key
        return recognizer, detector

    def make_engine(self) -> RealtimeRecognitionEngine:
        recognizer, detector = self.ensure_models()
        return RealtimeRecognitionEngine(
            recognizer,
            detector,
            self.profile,
            max_faces=self.max_faces,
            min_face_ratio=self.min_face_ratio,
        )

    def collect_images(self, paths: List[str]) -> List[Path]:
        out: List[Path] = []
        for raw in paths:
            path = Path(raw).expanduser().resolve()
            if path.is_dir():
                for child in sorted(path.rglob("*")):
                    if child.suffix.lower() in IMAGE_EXTS and child.is_file():
                        out.append(child)
            elif path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                out.append(path)
        seen = set()
        unique: List[Path] = []
        for p in out:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique

    def enroll_from_frames(
        self,
        name: str,
        frames: List[Tuple[str, np.ndarray]],
        *,
        merge: bool = False,
    ) -> Tuple[int, List[str]]:
        """Enroll from labeled BGR frames. Returns (n_samples_used, rejected)."""
        recognizer, _ = self.ensure_models(det_size=640)
        embeddings = []
        weights = []
        rejected: List[str] = []
        for label, frame in frames:
            if frame is None or frame.size == 0:
                rejected.append(f"{label}: empty")
                continue
            faces = recognizer.analyze(
                frame,
                identify=False,
                max_side=960,
                refine_small=True,
                small_face_px=140,
            )
            if not faces or faces[0].embedding is None:
                rejected.append(f"{label}: no face")
                continue
            face = faces[0]
            report = assess_face_quality(
                frame,
                face,
                min_face_px=40,
                min_det_score=0.35,
                min_sharpness=10.0,
            )
            embeddings.append(face.embedding.copy())
            weights.append(max(0.05, report.score if report.ok else 0.2))
        if not embeddings:
            return 0, rejected
        gallery = self.gallery()
        gallery.enroll(name, embeddings, replace=not merge, weights=weights, merge=merge)
        self.invalidate_models()
        return len(embeddings), rejected

    def enroll_from_images(
        self,
        name: str,
        image_paths: List[Path],
        *,
        merge: bool = False,
    ) -> Tuple[int, List[str]]:
        """Return (n_samples, rejected messages)."""
        frames: List[Tuple[str, np.ndarray]] = []
        rejected: List[str] = []
        for path in image_paths:
            frame = cv2.imread(str(path))
            if frame is None:
                rejected.append(f"{path.name}: unreadable")
                continue
            frames.append((path.name, frame))
        n, more = self.enroll_from_frames(name, frames, merge=merge)
        return n, rejected + more

    def recognize_image(self, image_path: Path):
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Cannot read image: {image_path}")
        gallery = self.gallery()
        if len(gallery) == 0:
            raise RuntimeError("Gallery is empty — enroll someone first.")
        recognizer, _ = self.ensure_models(det_size=640)
        faces = recognizer.analyze(frame, identify=True, max_side=960, refine_small=True)
        annotated = frame.copy()
        labels = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox
            if face.identity is not None:
                label = f"{face.identity} ({face.identity_score:.2f})"
                color = (0, 220, 0)
            else:
                label = f"Unknown ({face.identity_score:.2f})"
                color = (0, 165, 255)
            labels.append(label)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
        return annotated, labels

    def remove_identity(self, name: str) -> bool:
        ok = self.gallery().remove(name)
        if ok:
            self.invalidate_models()
        return ok


def embed_one(recognizer: FaceRecognizer, frame, refine_small: bool):
    faces = recognizer.analyze(
        frame,
        identify=False,
        max_side=640,
        refine_small=refine_small,
        small_face_px=140,
    )
    if not faces or faces[0].embedding is None:
        return None
    return faces[0]
