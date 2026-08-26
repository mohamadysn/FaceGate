"""
Shared typed results for the eye-tracking pipeline (FAIR: Interoperable).

These lightweight dataclasses are the exchange format between face / pupil / gaze / metrics modules so every
stage speaks the same vocabulary (pixels, scores, timestamps).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

BBox = Tuple[int, int, int, int]
Point = Tuple[float, float]


@dataclass
class EyeSample:
    """One eye measurement in image coordinates."""

    side: str  # "left" or "right"
    bbox: BBox
    pupil: Optional[Point] = None
    pupil_confidence: float = 0.0
    landmark: Optional[Point] = None


@dataclass
class FaceSample:
    """Face detection + landmark snapshot for a single frame."""

    bbox: BBox
    score: float
    landmarks: Optional[np.ndarray] = None  # shape (5, 2) InsightFace order
    eyes: List[EyeSample] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None  # L2-normalized identity vector
    identity: Optional[str] = None  # matched gallery name, if any
    identity_score: float = 0.0  # cosine similarity in [-1, 1]


@dataclass
class GazeSample:
    """Estimated gaze point in screen / canvas coordinates."""

    x: float
    y: float
    confidence: float
    raw_feature: Optional[np.ndarray] = None


@dataclass
class FrameMetrics:
    """Per-frame timing / quality metrics (performance)."""

    fps: float = 0.0
    latency_ms: float = 0.0
    n_faces: int = 0
    n_pupils: int = 0
    gaze_confidence: float = 0.0
