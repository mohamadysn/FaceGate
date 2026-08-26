"""
Face + landmark detection wrapper (FAIR: Reusable, Accessible).

Primary backend: InsightFace (buffalo_l detection).
Fallback: OpenCV Haar cascade (bbox only, no 5-point landmarks).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .types import BBox, EyeSample, FaceSample, Point

LEFT_EYE_IDX = 0
RIGHT_EYE_IDX = 1


def clip_bbox(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> Optional[BBox]:
    """Clip a box to image bounds; return None if empty."""
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def eye_rois_from_landmarks(
    landmarks: np.ndarray,
    frame_shape: Sequence[int],
    scale: float = 2.2,
) -> List[Tuple[str, BBox, Point]]:
    """
    Build left/right eye ROIs from InsightFace 5-point landmarks.

    Returns
    -------
    list of (side, bbox, landmark_point)
    """
    height, width = frame_shape[:2]
    left = landmarks[LEFT_EYE_IDX]
    right = landmarks[RIGHT_EYE_IDX]
    eye_distance = float(np.linalg.norm(right - left))
    half = max(8.0, eye_distance * 0.22 * scale)

    results: List[Tuple[str, BBox, Point]] = []
    for side, eye in (("left", left), ("right", right)):
        box = clip_bbox(
            int(eye[0] - half),
            int(eye[1] - half * 0.7),
            int(eye[0] + half),
            int(eye[1] + half * 0.7),
            width,
            height,
        )
        if box is not None:
            results.append((side, box, (float(eye[0]), float(eye[1]))))
    return results


class FaceDetector:
    """
    Detect faces and (when available) 5 facial landmarks.

    Parameters
    ----------
    backend :
        ``insightface`` (preferred) or ``opencv``.
    provider :
        ONNX provider for InsightFace.
    min_score :
        Minimum detection score kept in results.
    """

    def __init__(
        self,
        backend: str = "insightface",
        model_name: str = "buffalo_l",
        provider: str = "CPUExecutionProvider",
        device: int = 0,
        det_size: int = 320,
        min_score: float = 0.5,
    ) -> None:
        self.backend = backend.lower()
        self.min_score = min_score
        self._app = None
        self._cascade = None

        if self.backend == "insightface":
            self._init_insightface(model_name, provider, device, det_size, min_score)
        elif self.backend == "opencv":
            self._init_opencv()
        else:
            raise ValueError(f"Unknown face backend: {backend}")

    def _init_insightface(
        self,
        model_name: str,
        provider: str,
        device: int,
        det_size: int,
        min_score: float,
    ) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ImportError(
                "insightface is required for backend='insightface'. "
                "Install it or use --backend opencv."
            ) from exc

        providers = [provider]
        if "CUDA" in provider.upper():
            providers.append("CPUExecutionProvider")

        app = FaceAnalysis(name=model_name, allowed_modules=["detection"], providers=providers)
        size = None if det_size <= 0 else (det_size, det_size)
        ctx_id = int(device) if "CUDA" in provider.upper() else -1
        app.prepare(ctx_id=ctx_id, det_size=size, det_thresh=min_score)
        self._app = app

    def _init_opencv(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {cascade_path}")

    def detect(self, frame: np.ndarray, max_side: int = 640) -> List[FaceSample]:
        """
        Detect faces on a BGR frame.

        Parameters
        ----------
        max_side :
            Longest side for inference (smaller = faster). Boxes are scaled back.
        """
        if self.backend == "insightface":
            return self._detect_insightface(frame, max_side=max_side)
        return self._detect_opencv(frame)

    def _detect_insightface(self, frame: np.ndarray, max_side: int = 640) -> List[FaceSample]:
        from .camera_io import resize_for_inference

        assert self._app is not None
        infer_frame, scale = resize_for_inference(frame, max_side=max_side)
        faces = []
        for face in self._app.get(infer_frame):
            score = float(face.det_score)
            if score < self.min_score:
                continue
            x1, y1, x2, y2 = (face.bbox.astype(np.float32) * scale).astype(int)
            box = clip_bbox(x1, y1, x2, y2, frame.shape[1], frame.shape[0])
            if box is None:
                continue

            landmarks = None
            eyes: List[EyeSample] = []
            if getattr(face, "kps", None) is not None:
                landmarks = np.asarray(face.kps, dtype=np.float32) * scale
                for side, ebox, landmark in eye_rois_from_landmarks(landmarks, frame.shape):
                    eyes.append(EyeSample(side=side, bbox=ebox, landmark=landmark))

            faces.append(FaceSample(bbox=box, score=score, landmarks=landmarks, eyes=eyes))

        faces.sort(key=lambda item: item.score, reverse=True)
        return faces

    def _detect_opencv(self, frame: np.ndarray) -> List[FaceSample]:
        assert self._cascade is not None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        faces: List[FaceSample] = []
        for x, y, w, h in detections:
            box = clip_bbox(int(x), int(y), int(x + w), int(y + h), frame.shape[1], frame.shape[0])
            if box is None:
                continue
            # Approximate eye bands inside the face box (no landmarks available).
            x1, y1, x2, y2 = box
            fw, fh = x2 - x1, y2 - y1
            eyes = [
                EyeSample(
                    side="left",
                    bbox=(x1 + int(0.15 * fw), y1 + int(0.25 * fh), x1 + int(0.45 * fw), y1 + int(0.45 * fh)),
                ),
                EyeSample(
                    side="right",
                    bbox=(x1 + int(0.55 * fw), y1 + int(0.25 * fh), x1 + int(0.85 * fw), y1 + int(0.45 * fh)),
                ),
            ]
            faces.append(FaceSample(bbox=box, score=0.6, landmarks=None, eyes=eyes))
        return faces
