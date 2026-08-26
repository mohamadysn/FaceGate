"""
Face image quality heuristics (FAIR: Accessible / Reusable).

Used during enrollment and live matching so blurry / tiny / profile faces
do not pollute the gallery or trigger unstable IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .types import BBox, FaceSample


@dataclass
class QualityReport:
    """Scalar quality metrics for one face crop."""

    score: float  # combined [0, 1]
    sharpness: float
    face_size: float
    det_score: float
    ok: bool
    reason: str = ""


def bbox_size(bbox: BBox) -> Tuple[int, int]:
    """Return ``(width, height)`` of a bbox."""
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1), max(0, y2 - y1)


def estimate_sharpness(gray_crop: np.ndarray) -> float:
    """
    Laplacian variance sharpness proxy.

    Higher = sharper. Typical usable faces are often > ~40–60 depending on
    resolution; we normalize later.
    """
    if gray_crop.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())


def assess_face_quality(
    frame: np.ndarray,
    face: FaceSample,
    min_face_px: int = 80,
    min_det_score: float = 0.5,
    min_sharpness: float = 35.0,
) -> QualityReport:
    """
    Score whether a face is good enough for enrollment / recognition.

    Parameters
    ----------
    frame :
        Full BGR frame.
    face :
        Detected face sample.
    min_face_px :
        Minimum face box short side in pixels.
    min_det_score :
        Minimum detector confidence.
    min_sharpness :
        Minimum Laplacian variance on the face crop.
    """
    w, h = bbox_size(face.bbox)
    short = float(min(w, h))
    x1, y1, x2, y2 = face.bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return QualityReport(0.0, 0.0, short, face.score, False, "empty_crop")

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharp = estimate_sharpness(gray)
    det = float(face.score)

    if short < min_face_px:
        return QualityReport(0.0, sharp, short, det, False, "face_too_small")
    if det < min_det_score:
        return QualityReport(0.0, sharp, short, det, False, "low_det_score")
    if sharp < min_sharpness:
        return QualityReport(0.0, sharp, short, det, False, "blurry")

    # Soft [0,1] score for weighted enrollment.
    size_term = float(np.clip((short - min_face_px) / max(1.0, min_face_px), 0.0, 1.0))
    sharp_term = float(np.clip(sharp / 150.0, 0.0, 1.0))
    det_term = float(np.clip((det - min_det_score) / max(1e-6, 1.0 - min_det_score), 0.0, 1.0))
    score = 0.35 * size_term + 0.40 * sharp_term + 0.25 * det_term
    return QualityReport(score=score, sharpness=sharp, face_size=short, det_score=det, ok=True)


def is_frontal_enough(face: FaceSample, max_eye_asymmetry: float = 0.35) -> bool:
    """
    Cheap frontal check from 5-point landmarks when available.

    Rejects strong profiles where one eye landmark collapses toward the other.
    """
    if face.landmarks is None or len(face.landmarks) < 2:
        return True
    left, right = face.landmarks[0], face.landmarks[1]
    eye_dist = float(np.linalg.norm(right - left))
    if eye_dist < 1e-3:
        return False
    x1, _, x2, _ = face.bbox
    face_w = max(1.0, float(x2 - x1))
    # Eyes should span a reasonable fraction of face width.
    span_ratio = eye_dist / face_w
    return span_ratio >= (0.20 * (1.0 - max_eye_asymmetry))
