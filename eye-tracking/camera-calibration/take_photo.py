#!/usr/bin/env python3
"""
Capture chessboard images for camera calibration.

FAIR compliance notes
---------------------
Findable
    Outputs use a stable naming scheme: ``calib_<n>.jpg`` under ``calib_imgs/``.
Accessible
    Interactive CLI; live feedback when the chessboard is detected.
Interoperable
    Stores plain JPEG frames consumable by OpenCV ``calibrateCamera``.
Reusable
    Detection helpers are pure functions; capture directory is explicit.

Example
-------
::

    python take_photo.py --camera 0 --cols 9 --rows 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

_EYE_TRACKING_ROOT = Path(__file__).resolve().parents[1]
if str(_EYE_TRACKING_ROOT) not in sys.path:
    sys.path.insert(0, str(_EYE_TRACKING_ROOT))

from common.camera_io import (  # noqa: E402
    draw_text_lines,
    open_camera,
    should_quit,
    window_was_closed,
)

PatternSize = Tuple[int, int]
Corners = np.ndarray


def next_capture_index(out_dir: Path, prefix: str) -> int:
    """
    Return the next free numeric index for ``prefix*.jpg`` in ``out_dir``.

    This avoids overwriting previous captures when the tool is restarted
    (Reusable / Findable artifact naming).
    """
    existing = sorted(out_dir.glob(f"{prefix}*.jpg"))
    numbers: List[int] = []
    for path in existing:
        suffix = path.stem[len(prefix) :]
        if suffix.isdigit():
            numbers.append(int(suffix))
    return (max(numbers) + 1) if numbers else 0


def detect_chessboard(
    gray: np.ndarray,
    candidates: Sequence[PatternSize],
) -> Tuple[Optional[PatternSize], Optional[Corners]]:
    """
    Try several inner-corner sizes until a chessboard is found.

    Parameters
    ----------
    gray :
        Single-channel image.
    candidates :
        Candidate ``(cols, rows)`` inner-corner sizes.

    Returns
    -------
    pattern_size, corners
        Detected size and corner array, or ``(None, None)`` if not found.
    """
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    for size in candidates:
        found, corners = cv2.findChessboardCorners(gray, size, flags)
        if found:
            return size, corners
    return None, None


def candidate_pattern_sizes(cols: int, rows: int) -> List[PatternSize]:
    """Build a small search list around the nominal chessboard size."""
    return [
        (cols, rows),
        (cols - 1, rows - 1),
        (cols - 1, rows),
        (cols, rows - 1),
    ]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for calibration image capture."""
    parser = argparse.ArgumentParser(
        description="Capture chessboard frames for camera calibration.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index.")
    parser.add_argument("--width", type=int, default=1280, help="Requested frame width.")
    parser.add_argument("--height", type=int, default=1024, help="Requested frame height.")
    parser.add_argument("--cols", type=int, default=9, help="Inner corners along width.")
    parser.add_argument("--rows", type=int, default=7, help="Inner corners along height.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory (default: ./calib_imgs next to this script).",
    )
    return parser.parse_args()


def main() -> None:
    """
    Interactive capture loop with live chessboard feedback.

    Controls
    --------
    Space : save frame only if a chessboard is detected
    f     : force-save the current frame
    q/Esc : quit
    """
    args = parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent / "calib_imgs"
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = "calib_"
    count = next_capture_index(out_dir, prefix)
    candidates = candidate_pattern_sizes(args.cols, args.rows)

    cap = open_camera(args.camera, args.width, args.height)
    window_name = "Calibration capture"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"Saving images to: {out_dir}")
    print("Space = capture if detected | f = force | q/Esc = quit")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Error: failed to read a frame.")
            break

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pattern_size, corners = detect_chessboard(gray, candidates)
        detected = corners is not None

        if detected:
            cv2.drawChessboardCorners(display, pattern_size, corners, True)
            status = f"BOARD OK {pattern_size[0]}x{pattern_size[1]}"
            color = (0, 220, 0)
        else:
            status = "Board not detected — adjust angle / distance"
            color = (0, 165, 255)

        draw_text_lines(
            display,
            [
                status,
                f"Captures index: {count}",
                "Space: capture | f: force | q: quit",
            ],
            color=color,
        )
        cv2.imshow(window_name, display)

        key = cv2.waitKey(15) & 0xFF
        if key == 32:  # Space
            if not detected:
                print("Skipped: board not detected (press f to force-save).")
            else:
                path = out_dir / f"{prefix}{count}.jpg"
                cv2.imwrite(str(path), frame)
                print(f"Saved: {path}")
                count += 1
        elif key in (ord("f"), ord("F")):
            path = out_dir / f"{prefix}{count}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"Forced save: {path}")
            count += 1
        elif should_quit(key) or window_was_closed(window_name):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Next free index would be {count} in {out_dir}")


if __name__ == "__main__":
    main()
