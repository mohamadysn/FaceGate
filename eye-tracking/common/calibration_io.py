"""
Camera intrinsic / distortion I/O using OpenCV YAML (FAIR: Interoperable).

Stored schema (OpenCV FileStorage):
    K : 3x3 camera intrinsic matrix
    D : distortion coefficient vector

This format is portable across OpenCV language bindings and many robotics
calibration tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

PathLike = Union[str, Path]


def save_camera_coefficients(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    path: PathLike,
) -> Path:
    """
    Persist camera intrinsics ``K`` and distortion ``D`` to YAML.

    Parameters
    ----------
    camera_matrix :
        3x3 intrinsic matrix.
    dist_coeffs :
        Distortion coefficients (OpenCV layout).
    path :
        Output ``.yml`` / ``.yaml`` path.

    Returns
    -------
    Path
        Resolved output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    storage.write("K", camera_matrix)
    storage.write("D", dist_coeffs)
    storage.release()
    return path.resolve()


def load_camera_coefficients(
    path: PathLike,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load camera intrinsics ``K`` and distortion ``D`` from YAML.

    Parameters
    ----------
    path :
        Path to an OpenCV FileStorage YAML written by
        :func:`save_camera_coefficients`.

    Returns
    -------
    K, D
        Matrices if present and readable; ``(None, None)`` if the file is
        missing or incomplete. Callers can treat that as "no calibration".
    """
    path = Path(path)
    if not path.is_file():
        return None, None

    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    camera_matrix = storage.getNode("K").mat()
    dist_coeffs = storage.getNode("D").mat()
    storage.release()

    if camera_matrix is None or dist_coeffs is None:
        return None, None
    return camera_matrix, dist_coeffs
