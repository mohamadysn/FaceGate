#!/usr/bin/env python3
"""
Estimate camera intrinsics from chessboard images.

FAIR compliance notes
---------------------
Findable
    Writes a clearly named artifact ``camera_matrix.yml`` with keys ``K`` / ``D``.
Accessible
    Fully CLI-driven; reports RMS and mean reprojection error for quality checks.
Interoperable
    Uses OpenCV YAML FileStorage, readable by other OpenCV tools and languages.
Reusable
    ``calibrate``, ``save_camera_coefficients``, and ``load_camera_coefficients``
    are importable without launching the CLI.

Example
-------
::

    python calibration.py --images ./calib_imgs --square-size 0.025
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

_EYE_TRACKING_ROOT = Path(__file__).resolve().parents[1]
if str(_EYE_TRACKING_ROOT) not in sys.path:
    sys.path.insert(0, str(_EYE_TRACKING_ROOT))

from common.calibration_io import (  # noqa: E402
    load_camera_coefficients,
    save_camera_coefficients,
)

# Sub-pixel refinement stopping criteria (OpenCV convention).
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
HERE = Path(__file__).resolve().parent
PathLike = Union[str, Path]
PatternSize = Tuple[int, int]


def _pattern_candidates(width: int, height: int) -> List[PatternSize]:
    """Return nominal and nearby chessboard inner-corner sizes."""
    return [
        (width, height),
        (width - 1, height - 1),
        (width - 1, height),
        (width, height - 1),
    ]


def _mean_reprojection_error(
    objpoints: Sequence[np.ndarray],
    imgpoints: Sequence[np.ndarray],
    rvecs: Sequence[np.ndarray],
    tvecs: Sequence[np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    """
    Compute the RMS reprojection error in pixels.

    Lower is better. Values above ~1 px usually mean poor photos or a wrong
    board size / square size.
    """
    total_error = 0.0
    total_points = 0
    for i, object_points in enumerate(objpoints):
        projected, _ = cv2.projectPoints(object_points, rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], projected, cv2.NORM_L2)
        n_points = len(projected)
        total_error += error * error
        total_points += n_points
    if total_points == 0:
        return float("nan")
    return float(np.sqrt(total_error / total_points))


def calibrate(
    dirpath: PathLike,
    prefix: str,
    image_format: str,
    square_size: float,
    width: int = 9,
    height: int = 7,
    show_detections: bool = True,
) -> Dict[str, object]:
    """
    Calibrate a camera from chessboard images in a directory.

    Parameters
    ----------
    dirpath :
        Folder containing calibration frames.
    prefix :
        Filename prefix, e.g. ``calib_``.
    image_format :
        Extension without dot, e.g. ``jpg``.
    square_size :
        Physical length of one chessboard square (metres).
    width, height :
        Expected inner-corner counts (columns, rows).
    show_detections :
        If True, briefly display each accepted board detection.

    Returns
    -------
    dict
        Keys: ``rms``, ``mtx``, ``dist``, ``rvecs``, ``tvecs``,
        ``mean_error``, ``used``, ``pattern_size``.
    """
    dirpath = Path(dirpath)
    images = sorted(glob.glob(str(dirpath / f"{prefix}*.{image_format}")))
    if not images:
        raise FileNotFoundError(f"No images matching {prefix}*.{image_format} in {dirpath}")

    objpoints: List[np.ndarray] = []
    imgpoints: List[np.ndarray] = []
    used: List[str] = []

    pattern_size: Optional[PatternSize] = None
    object_template: Optional[np.ndarray] = None
    image_size: Optional[Tuple[int, int]] = None

    for fname in images:
        image = cv2.imread(fname)
        if image is None:
            print(f"Skipping unreadable image: {fname}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]  # (width, height)

        # Auto-detect board size on the first successful image.
        if pattern_size is None:
            for candidate in _pattern_candidates(width, height):
                if candidate[0] <= 0 or candidate[1] <= 0:
                    continue
                found, _ = cv2.findChessboardCorners(gray, candidate, None)
                if found:
                    pattern_size = candidate
                    object_template = np.zeros((candidate[1] * candidate[0], 3), np.float32)
                    object_template[:, :2] = np.mgrid[0 : candidate[0], 0 : candidate[1]].T.reshape(-1, 2)
                    object_template *= square_size
                    print(f"Detected board size: {candidate[0]}x{candidate[1]}")
                    break
            if pattern_size is None:
                print(f"No chessboard in: {fname}")
                continue

        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if not found:
            print(f"Corners not found: {fname}")
            continue

        # Refine to sub-pixel accuracy before feeding calibrateCamera.
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)
        objpoints.append(object_template.copy())
        imgpoints.append(corners_refined)
        used.append(fname)

        if show_detections:
            preview = image.copy()
            cv2.drawChessboardCorners(preview, pattern_size, corners_refined, found)
            cv2.imshow("Chessboard corners", preview)
            cv2.waitKey(200)

    if show_detections:
        cv2.destroyAllWindows()

    if not imgpoints or image_size is None:
        raise ValueError("No chessboard corners detected. Check images and board size.")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )
    mean_error = _mean_reprojection_error(objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs)

    print(f"Images used: {len(used)}/{len(images)}")
    print(f"calibrateCamera RMS: {rms:.4f}")
    print(f"Mean reprojection error: {mean_error:.4f} px")
    if mean_error > 1.0:
        print("Warning: error > 1 px. Capture more sharp images under varied angles.")

    return {
        "rms": rms,
        "mtx": camera_matrix,
        "dist": dist_coeffs,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "mean_error": mean_error,
        "used": used,
        "pattern_size": pattern_size,
    }


# Keep public aliases for backwards compatibility / external imports.
save_coefficients = save_camera_coefficients
load_coefficients = load_camera_coefficients


def display_calibration_result(
    image_path: PathLike,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> None:
    """
    Show side-by-side original vs undistorted calibration image.

    Parameters
    ----------
    image_path :
        Path to one calibration frame.
    camera_matrix, dist_coeffs :
        Intrinsics from :func:`calibrate`.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for pattern_size in [(9, 7), (8, 6), (7, 5), (6, 8), (9, 6)]:
        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if not found:
            continue

        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)
        left = image.copy()
        cv2.drawChessboardCorners(left, pattern_size, corners_refined, found)

        undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
        right = undistorted.copy()
        cv2.drawChessboardCorners(right, pattern_size, corners_refined, found)

        combined = np.hstack((left, right))
        cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(
            combined,
            "Undistorted",
            (left.shape[1] + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.namedWindow("Calibration result", cv2.WINDOW_NORMAL)
        cv2.imshow("Calibration result", combined)
        print("Press any key to close the comparison window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Fallback when corners cannot be re-detected: still show undistortion.
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
    combined = np.hstack((image, undistorted))
    cv2.namedWindow("Calibration result", cv2.WINDOW_NORMAL)
    cv2.imshow("Calibration result", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for offline camera calibration."""
    parser = argparse.ArgumentParser(
        description="Compute camera matrix K and distortion D from chessboard images.",
    )
    parser.add_argument("--images", type=str, default=str(HERE / "calib_imgs"), help="Input image folder.")
    parser.add_argument("--prefix", type=str, default="calib_", help="Filename prefix.")
    parser.add_argument("--format", type=str, default="jpg", help="Image extension.")
    parser.add_argument("--square-size", type=float, default=0.025, help="Square size in metres.")
    parser.add_argument("--cols", type=int, default=9, help="Inner corners (width).")
    parser.add_argument("--rows", type=int, default=7, help="Inner corners (height).")
    parser.add_argument(
        "--output",
        type=str,
        default=str(HERE / "camera_matrix.yml"),
        help="Output OpenCV YAML path (interoperable artifact).",
    )
    parser.add_argument("--no-preview", action="store_true", help="Skip visual comparison.")
    return parser.parse_args()


def main() -> None:
    """Run calibration, save YAML coefficients, optionally preview undistortion."""
    args = parse_args()
    result = calibrate(
        dirpath=args.images,
        prefix=args.prefix,
        image_format=args.format,
        square_size=args.square_size,
        width=args.cols,
        height=args.rows,
        show_detections=not args.no_preview,
    )

    output_path = save_camera_coefficients(result["mtx"], result["dist"], args.output)
    print(f"Saved calibration artifact: {output_path}")

    if not args.no_preview and result["used"]:
        display_calibration_result(result["used"][0], result["mtx"], result["dist"])


if __name__ == "__main__":
    main()
