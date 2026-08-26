"""
Gaze feature extraction and screen mapping (FAIR-documented).

Method
------
1. Build a normalized feature vector from pupil positions relative to eye ROIs
   (and optionally inter-ocular geometry).
2. Fit a polynomial regressor from features -> screen (x, y) using a short
   on-screen calibration session (typically 9 targets).
3. At runtime, predict gaze and apply exponential smoothing for stability.

Artifacts
---------
Calibration is saved as JSON (Interoperable, Findable)::

    {
      "targets": [[x, y], ...],
      "features": [[...], ...],
      "coefficients_x": [...],
      "coefficients_y": [...],
      "feature_dim": 6,
      "canvas_size": [W, H]
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .pupil import is_valid_pupil
from .types import FaceSample, GazeSample, Point

PathLike = Union[str, Path]


def extract_gaze_features(face: FaceSample) -> Optional[np.ndarray]:
    """
    Build a 6-D gaze feature vector from a face with pupil estimates.

    Features (normalized roughly to [-1, 1] within each eye box)
    -----------------------------------------------------------
    0-1 : left pupil (x, y) relative to left eye bbox center / half-size
    2-3 : right pupil (x, y) relative to right eye bbox center / half-size
    4-5 : average pupil (x, y) relative to face bbox center / half-size

    Returns
    -------
    np.ndarray or None
        Feature vector, or ``None`` if both pupils are missing.
    """
    left = next((e for e in face.eyes if e.side == "left" and is_valid_pupil(e)), None)
    right = next((e for e in face.eyes if e.side == "right" and is_valid_pupil(e)), None)
    if left is None and right is None:
        return None

    def eye_local(eye) -> Tuple[float, float]:
        x1, y1, x2, y2 = eye.bbox
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        hx = max(1.0, 0.5 * (x2 - x1))
        hy = max(1.0, 0.5 * (y2 - y1))
        assert eye.pupil is not None
        return (eye.pupil[0] - cx) / hx, (eye.pupil[1] - cy) / hy

    if left is not None:
        lx, ly = eye_local(left)
    else:
        lx, ly = 0.0, 0.0
    if right is not None:
        rx, ry = eye_local(right)
    else:
        rx, ry = 0.0, 0.0

    fx1, fy1, fx2, fy2 = face.bbox
    fcx = 0.5 * (fx1 + fx2)
    fcy = 0.5 * (fy1 + fy2)
    fhx = max(1.0, 0.5 * (fx2 - fx1))
    fhy = max(1.0, 0.5 * (fy2 - fy1))

    pupils = [e.pupil for e in (left, right) if e is not None and e.pupil is not None]
    mx = float(np.mean([p[0] for p in pupils]))
    my = float(np.mean([p[1] for p in pupils]))

    return np.asarray(
        [lx, ly, rx, ry, (mx - fcx) / fhx, (my - fcy) / fhy],
        dtype=np.float64,
    )


def _design_matrix(features: np.ndarray) -> np.ndarray:
    """
    Expand features with bias + pairwise products (degree-2 polynomial).

    For a vector f of length d::
        [1, f0, f1, ..., fd-1, f0*f0, f0*f1, ..., f{d-1}*f{d-1}]
    """
    features = np.atleast_2d(features)
    n, d = features.shape
    cols: List[np.ndarray] = [np.ones((n, 1)), features]
    for i in range(d):
        for j in range(i, d):
            cols.append((features[:, i] * features[:, j])[:, None])
    return np.hstack(cols)


@dataclass
class GazeModel:
    """
    Polynomial gaze mapper: eye features -> canvas coordinates.

    Parameters
    ----------
    canvas_size :
        ``(width, height)`` of the calibration / prediction canvas.
    """

    canvas_size: Tuple[int, int]
    coefficients_x: Optional[np.ndarray] = None
    coefficients_y: Optional[np.ndarray] = None
    feature_dim: int = 6

    def is_fitted(self) -> bool:
        """Return True when both x/y coefficient vectors are available."""
        return self.coefficients_x is not None and self.coefficients_y is not None

    def fit(self, features: Sequence[np.ndarray], targets: Sequence[Point], ridge: float = 1e-3) -> None:
        """
        Fit ridge-regularized least squares for x and y.

        Parameters
        ----------
        features :
            List of feature vectors from :func:`extract_gaze_features`.
        targets :
            Matching screen points ``(x, y)``.
        ridge :
            L2 regularization strength (helps with few calibration samples).
        """
        if len(features) < 3:
            raise ValueError("Need at least 3 calibration samples to fit a gaze model.")

        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("features must be a 2-D array-like")
        self.feature_dim = x.shape[1]

        phi = _design_matrix(x)
        eye = np.eye(phi.shape[1]) * ridge
        # Do not regularize the bias term.
        eye[0, 0] = 0.0
        lhs = phi.T @ phi + eye
        self.coefficients_x = np.linalg.solve(lhs, phi.T @ y[:, 0])
        self.coefficients_y = np.linalg.solve(lhs, phi.T @ y[:, 1])

    def predict(self, feature: np.ndarray) -> GazeSample:
        """Predict a gaze point; clamps to the canvas and reports a soft confidence."""
        if not self.is_fitted():
            raise RuntimeError("GazeModel is not fitted. Run calibration first.")

        feature = np.asarray(feature, dtype=np.float64).reshape(1, -1)
        if feature.shape[1] != self.feature_dim:
            raise ValueError(f"Expected feature dim {self.feature_dim}, got {feature.shape[1]}")

        phi = _design_matrix(feature)
        assert self.coefficients_x is not None and self.coefficients_y is not None
        gx = float((phi @ self.coefficients_x).ravel()[0])
        gy = float((phi @ self.coefficients_y).ravel()[0])

        width, height = self.canvas_size
        # Soft confidence: higher when prediction sits comfortably inside canvas.
        margin_x = min(gx, width - gx) / max(1.0, width / 2.0)
        margin_y = min(gy, height - gy) / max(1.0, height / 2.0)
        confidence = float(np.clip(0.5 * (margin_x + margin_y), 0.0, 1.0))

        gx = float(np.clip(gx, 0.0, width - 1.0))
        gy = float(np.clip(gy, 0.0, height - 1.0))
        return GazeSample(x=gx, y=gy, confidence=confidence, raw_feature=feature.reshape(-1))

    def save(self, path: PathLike) -> Path:
        """Serialize the model to JSON (Interoperable artifact)."""
        if not self.is_fitted():
            raise RuntimeError("Cannot save an unfitted GazeModel.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "canvas_size": list(self.canvas_size),
            "feature_dim": self.feature_dim,
            "coefficients_x": self.coefficients_x.tolist(),
            "coefficients_y": self.coefficients_y.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path.resolve()

    @classmethod
    def load(cls, path: PathLike) -> "GazeModel":
        """Load a previously saved gaze model JSON."""
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = cls(
            canvas_size=(int(payload["canvas_size"][0]), int(payload["canvas_size"][1])),
            feature_dim=int(payload["feature_dim"]),
            coefficients_x=np.asarray(payload["coefficients_x"], dtype=np.float64),
            coefficients_y=np.asarray(payload["coefficients_y"], dtype=np.float64),
        )
        return model


class GazeSmoother:
    """Exponential moving average for gaze coordinates (reduces jitter)."""

    def __init__(self, alpha: float = 0.35) -> None:
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self._state: Optional[Point] = None

    def reset(self) -> None:
        """Clear the filter state (e.g. after recalibration)."""
        self._state = None

    def update(self, gaze: GazeSample) -> GazeSample:
        """Return a smoothed copy of ``gaze``."""
        if self._state is None:
            self._state = (gaze.x, gaze.y)
        else:
            ax = self.alpha
            self._state = (
                ax * gaze.x + (1.0 - ax) * self._state[0],
                ax * gaze.y + (1.0 - ax) * self._state[1],
            )
        return GazeSample(
            x=self._state[0],
            y=self._state[1],
            confidence=gaze.confidence,
            raw_feature=gaze.raw_feature,
        )


def default_calibration_targets(width: int, height: int, grid: int = 3, margin_ratio: float = 0.12) -> List[Point]:
    """
    Build an evenly spaced grid of calibration targets on a canvas.

    Parameters
    ----------
    width, height :
        Canvas size in pixels.
    grid :
        Points per axis (3 => 9-point calibration).
    margin_ratio :
        Inset from borders to avoid unreachable screen edges.
    """
    margin_x = width * margin_ratio
    margin_y = height * margin_ratio
    xs = np.linspace(margin_x, width - margin_x, grid)
    ys = np.linspace(margin_y, height - margin_y, grid)
    return [(float(x), float(y)) for y in ys for x in xs]
