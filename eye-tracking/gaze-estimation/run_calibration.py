#!/usr/bin/env python3
"""
Interactive gaze calibration and live gaze pointer.

Workflow
--------
1. A 3x3 grid of targets is shown on a canvas.
2. Look at each red target; press SPACE when fixating (samples are averaged).
3. A polynomial gaze model is fitted and saved to JSON.
4. Live mode draws a smoothed gaze cursor.

Example
-------
::

    python run_calibration.py --provider CPUExecutionProvider
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.camera_io import draw_text_lines, open_camera, should_quit, smooth_fps_update, window_was_closed
from common.face_detector import FaceDetector
from common.gaze_model import (
    GazeModel,
    GazeSmoother,
    default_calibration_targets,
    extract_gaze_features,
)
from common.pupil import draw_eye_segmentation, segment_eyes_for_face
from common.types import Point


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gaze calibration + live pointer.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--backend", type=str, default="insightface", choices=["insightface", "opencv"])
    parser.add_argument("--provider", type=str, default="CPUExecutionProvider")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--canvas-width", type=int, default=1280)
    parser.add_argument("--canvas-height", type=int, default=720)
    parser.add_argument("--grid", type=int, default=3, help="Calibration targets per axis.")
    parser.add_argument("--samples-per-target", type=int, default=12)
    parser.add_argument(
        "--model-out",
        type=str,
        default=str(Path(__file__).resolve().parent / "artifacts" / "gaze_model.json"),
    )
    parser.add_argument("--load-model", type=str, default="", help="Skip calibration and load JSON.")
    parser.add_argument("--no-mirror", action="store_true")
    return parser.parse_args()


def average_feature(samples: List[np.ndarray]) -> np.ndarray:
    """Mean feature vector for one fixation target."""
    return np.mean(np.stack(samples, axis=0), axis=0)


def run_calibration(
    detector: FaceDetector,
    cap: cv2.VideoCapture,
    canvas_size: tuple[int, int],
    grid: int,
    samples_per_target: int,
    mirror: bool,
) -> GazeModel:
    """Guided 9-point (or NxN) calibration session."""
    targets = default_calibration_targets(canvas_size[0], canvas_size[1], grid=grid)
    features: List[np.ndarray] = []
    kept_targets: List[Point] = []

    window = "Gaze calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, canvas_size[0], canvas_size[1])

    print("Look at the red target, then press SPACE to capture. q = abort.")

    for index, target in enumerate(targets):
        collected: List[np.ndarray] = []
        while len(collected) < samples_per_target:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("Camera read failed during calibration.")
            if mirror:
                frame = cv2.flip(frame, 1)

            faces = detector.detect(frame)
            feature = None
            if faces:
                face = segment_eyes_for_face(frame, faces[0])
                draw_eye_segmentation(frame, face)
                feature = extract_gaze_features(face)

            canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
            # Picture-in-picture camera preview.
            preview = cv2.resize(frame, (320, 180))
            canvas[10:190, 10:330] = preview

            for done_t in kept_targets:
                cv2.circle(canvas, (int(done_t[0]), int(done_t[1])), 10, (0, 180, 0), -1)
            cv2.circle(canvas, (int(target[0]), int(target[1])), 16, (0, 0, 255), -1)
            cv2.circle(canvas, (int(target[0]), int(target[1])), 4, (255, 255, 255), -1)

            draw_text_lines(
                canvas,
                [
                    f"Target {index + 1}/{len(targets)}",
                    f"Samples: {len(collected)}/{samples_per_target}",
                    "SPACE: sample   q: abort",
                    "Status: READY" if feature is not None else "Status: NO PUPIL — move closer / light up",
                ],
                color=(0, 255, 0) if feature is not None else (0, 165, 255),
            )
            cv2.imshow(window, canvas)

            key = cv2.waitKey(10) & 0xFF
            if should_quit(key) or window_was_closed(window):
                raise KeyboardInterrupt("Calibration aborted by user.")
            if key == 32 and feature is not None:
                collected.append(feature)

        features.append(average_feature(collected))
        kept_targets.append(target)
        print(f"Captured target {index + 1}/{len(targets)} at {target}")

    model = GazeModel(canvas_size=canvas_size)
    model.fit(features, kept_targets)
    print("Calibration fitted successfully.")
    return model


def run_live_gaze(
    detector: FaceDetector,
    cap: cv2.VideoCapture,
    model: GazeModel,
    mirror: bool,
) -> None:
    """Live gaze pointer after calibration."""
    window = "Live gaze"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    width, height = model.canvas_size
    cv2.resizeWindow(window, width, height)
    smoother = GazeSmoother(alpha=0.3)
    fps = 0.0
    prev = time.time()
    print("Live gaze running. q = quit.")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if mirror:
            frame = cv2.flip(frame, 1)

        t0 = time.time()
        faces = detector.detect(frame)
        gaze = None
        if faces:
            face = segment_eyes_for_face(frame, faces[0])
            draw_eye_segmentation(frame, face)
            feature = extract_gaze_features(face)
            if feature is not None and model.is_fitted():
                gaze = smoother.update(model.predict(feature))
        latency_ms = (time.time() - t0) * 1000.0
        fps, prev = smooth_fps_update(fps, prev)

        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        preview = cv2.resize(frame, (320, 180))
        canvas[10:190, 10:330] = preview

        if gaze is not None:
            cv2.circle(canvas, (int(gaze.x), int(gaze.y)), 18, (0, 255, 255), 2)
            cv2.circle(canvas, (int(gaze.x), int(gaze.y)), 5, (0, 255, 255), -1)

        draw_text_lines(
            canvas,
            [
                f"FPS: {fps:.1f} | latency: {latency_ms:.0f} ms",
                f"Gaze conf: {gaze.confidence:.2f}" if gaze else "Gaze: unavailable",
                "q: quit",
            ],
        )
        cv2.imshow(window, canvas)
        key = cv2.waitKey(10) & 0xFF
        if should_quit(key) or window_was_closed(window):
            break


def main() -> None:
    args = parse_args()
    detector = FaceDetector(backend=args.backend, provider=args.provider)
    cap = open_camera(args.camera, args.width, args.height)
    mirror = not args.no_mirror
    canvas_size = (args.canvas_width, args.canvas_height)

    try:
        if args.load_model:
            model = GazeModel.load(args.load_model)
            print(f"Loaded gaze model: {args.load_model}")
        else:
            model = run_calibration(
                detector,
                cap,
                canvas_size,
                args.grid,
                args.samples_per_target,
                mirror,
            )
            out = model.save(args.model_out)
            print(f"Saved gaze model: {out}")

        run_live_gaze(detector, cap, model, mirror)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
