#!/usr/bin/env python3
"""
Webcam video acquisition.

FAIR compliance notes
---------------------
Findable
    Module ``webcam-capture`` matches the project specification; this module docstring
    states purpose, inputs, and outputs.
Accessible
    Runnable as a CLI script; depends only on OpenCV.
Interoperable
    Uses standard OpenCV ``VideoCapture`` indices and BGR frames.
Reusable
    Camera open / HUD / quit helpers live in ``common.camera_io`` so later
    modules can reuse the same behaviour.

Example
-------
::

    python input-camera.py --camera 0 --width 1280 --height 1024
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

# Make ``eye-tracking/common`` importable when the script is run directly.
_EYE_TRACKING_ROOT = Path(__file__).resolve().parents[1]
if str(_EYE_TRACKING_ROOT) not in sys.path:
    sys.path.insert(0, str(_EYE_TRACKING_ROOT))

from common.camera_io import (  # noqa: E402
    draw_text_lines,
    open_camera,
    should_quit,
    smooth_fps_update,
    window_was_closed,
)


def build_hud_lines(frame_shape, fps: float, mirror: bool) -> list[str]:
    """
    Build on-screen status lines for the live viewer.

    Parameters
    ----------
    frame_shape :
        ``(height, width, ...)`` of the current frame.
    fps :
        Smoothed frames-per-second estimate.
    mirror :
        Whether horizontal mirroring is enabled.
    """
    height, width = frame_shape[:2]
    lines = [
        f"FPS: {fps:.1f}",
        f"{width}x{height}",
        "q/Esc: quit  |  m: mirror",
    ]
    if mirror:
        lines.append("mirror: ON")
    return lines


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for webcam acquisition."""
    parser = argparse.ArgumentParser(
        description="Acquire and display webcam frames (FAIR-ready CLI).",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index.")
    parser.add_argument("--width", type=int, default=1280, help="Requested frame width.")
    parser.add_argument("--height", type=int, default=1024, help="Requested frame height.")
    parser.add_argument("--fps", type=int, default=30, help="Requested FPS (0 = driver default).")
    parser.add_argument("--mirror", action="store_true", help="Start with horizontal mirror ON.")
    return parser.parse_args()


def main() -> None:
    """
    Run the interactive webcam loop.

    Controls
    --------
    q / Esc : quit
    m       : toggle horizontal mirror
    window X: quit
    """
    args = parse_args()
    cap = open_camera(args.camera, args.width, args.height, args.fps)

    window_name = "Webcam"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    mirror = args.mirror
    previous_time = time.time()
    smoothed_fps = 0.0

    print("Click the video window. q/Esc = quit, m = mirror.")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Error: failed to read a frame from the camera.")
            break

        # Mirror after capture so HUD text stays left-aligned in screen space.
        if mirror:
            frame = cv2.flip(frame, 1)

        smoothed_fps, previous_time = smooth_fps_update(smoothed_fps, previous_time)
        draw_text_lines(frame, build_hud_lines(frame.shape, smoothed_fps, mirror))
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(15) & 0xFF
        if should_quit(key):
            break
        if key in (ord("m"), ord("M")):
            mirror = not mirror
        if window_was_closed(window_name):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
