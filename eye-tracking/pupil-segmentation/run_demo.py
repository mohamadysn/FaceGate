#!/usr/bin/env python3
"""
Live pupil segmentation demo.

Shows face boxes, eye ROIs, pupil centers and a zoomed eye panel.

Example
-------
::

    python run_demo.py --provider CPUExecutionProvider
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.camera_io import draw_text_lines, open_camera, should_quit, smooth_fps_update, window_was_closed
from common.face_detector import FaceDetector
from common.pupil import (
    count_valid_pupils,
    draw_eye_segmentation,
    make_eye_debug_panel,
    segment_eyes_for_face,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pupil segmentation / pupil localization demo.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--backend", type=str, default="insightface", choices=["insightface", "opencv"])
    parser.add_argument("--provider", type=str, default="CPUExecutionProvider")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--no-mirror", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = FaceDetector(
        backend=args.backend,
        provider=args.provider,
        min_score=args.min_score,
    )
    cap = open_camera(args.camera, args.width, args.height)
    window_name = "Pupil segmentation"
    preview_name = "Eye crops"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(preview_name, cv2.WINDOW_NORMAL)

    mirror = not args.no_mirror
    fps = 0.0
    prev = time.time()
    print("q/Esc quit | m mirror")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if mirror:
            frame = cv2.flip(frame, 1)

        t0 = time.time()
        faces = detector.detect(frame)
        if faces:
            faces[0] = segment_eyes_for_face(frame, faces[0])
            face = faces[0]
            x1, y1, x2, y2 = face.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            draw_eye_segmentation(frame, face)
            cv2.imshow(preview_name, make_eye_debug_panel(frame, face))
            n_pupils = count_valid_pupils(face)
        else:
            n_pupils = 0
            cv2.imshow(preview_name, make_eye_debug_panel(frame, None))

        latency_ms = (time.time() - t0) * 1000.0
        fps, prev = smooth_fps_update(fps, prev)
        draw_text_lines(
            frame,
            [
                f"FPS: {fps:.1f} | latency: {latency_ms:.0f} ms",
                f"Pupils: {n_pupils}",
                "q: quit  m: mirror",
            ],
        )
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(10) & 0xFF
        if should_quit(key) or window_was_closed(window_name):
            break
        if key in (ord("m"), ord("M")):
            mirror = not mirror

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
