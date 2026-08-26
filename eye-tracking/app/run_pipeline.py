#!/usr/bin/env python3
"""
Unified realtime application (face recognition + optional eye / gaze).

Stages per frame
----------------
1. ``webcam-capture`` — grab webcam frame
2. ``camera-calibration`` — optional undistortion (``camera_matrix.yml``)
3. ``face-recognition`` — detect (+ identify when ``--recognize``)
4. ``pupil-segmentation`` — eye ROI + pupil localization
5. ``gaze-estimation`` — gaze pointer if a model is available
6. ``performance-metrics`` — HUD + optional JSONL log

Example
-------
::

    python run_pipeline.py --recognize --provider CPUExecutionProvider
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.calibration_io import load_camera_coefficients
from common.camera_io import (
    draw_text_lines,
    grab_latest_frame,
    open_camera,
    should_quit,
    smooth_fps_update,
    window_was_closed,
)
from common.face_detector import FaceDetector
from common.face_recognition import DEFAULT_MATCH_THRESHOLD, FaceGallery, FaceRecognizer
from common.gaze_model import GazeModel, GazeSmoother, extract_gaze_features
from common.metrics import PerformanceTracker
from common.pupil import (
    count_valid_pupils,
    draw_eye_segmentation,
    make_eye_debug_panel,
    segment_eyes_for_face,
)
from common.types import FrameMetrics

DEFAULT_CALIB = _ROOT / "camera-calibration" / "camera_matrix.yml"
DEFAULT_GAZE = _ROOT / "gaze-estimation" / "artifacts" / "gaze_model.json"
DEFAULT_LOG = _ROOT / "performance-metrics" / "logs" / "session.jsonl"
DEFAULT_GALLERY = _ROOT / "face-recognition" / "gallery"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified pipeline: face recognition / tracking / gaze / metrics.",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--backend", type=str, default="insightface", choices=["insightface", "opencv"])
    parser.add_argument("--provider", type=str, default="CPUExecutionProvider")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--calib", type=str, default=str(DEFAULT_CALIB))
    parser.add_argument("--gaze-model", type=str, default=str(DEFAULT_GAZE))
    parser.add_argument("--log-metrics", type=str, default="", help="JSONL path (empty = no file log).")
    parser.add_argument("--enable-log", action="store_true", help=f"Log metrics to {DEFAULT_LOG}")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--undistort", action="store_true", help="Start with undistortion enabled.")
    parser.add_argument(
        "--recognize",
        action="store_true",
        help="Enable face recognition against the enrolled gallery.",
    )
    parser.add_argument("--gallery", type=str, default=str(DEFAULT_GALLERY))
    parser.add_argument("--threshold", type=float, default=DEFAULT_MATCH_THRESHOLD)
    parser.add_argument("--det-size", type=int, default=320, help="Detector input size (lower = faster).")
    parser.add_argument("--infer-size", type=int, default=480, help="Max inference side length.")
    parser.add_argument("--every", type=int, default=2, help="Run heavy inference every N frames.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    gallery: Optional[FaceGallery] = None
    detector: Union[FaceDetector, FaceRecognizer]
    if args.recognize:
        gallery = FaceGallery(args.gallery)
        gallery.match_threshold = args.threshold
        if len(gallery) == 0:
            print(f"[face] Gallery empty at {args.gallery}")
            print("       Enroll first: python face-recognition/enroll.py --name YourName")
        else:
            print(f"[face] Recognition ON — {len(gallery)} identities: {', '.join(gallery.names())}")
        detector = FaceRecognizer(
            provider=args.provider,
            device=args.device,
            min_score=args.min_score,
            gallery=gallery,
            match_threshold=args.threshold,
            det_size=args.det_size,
        )
    else:
        detector = FaceDetector(
            backend=args.backend,
            provider=args.provider,
            device=args.device,
            min_score=args.min_score,
            det_size=args.det_size,
        )

    camera_matrix, dist_coeffs = load_camera_coefficients(args.calib)
    if camera_matrix is not None:
        print(f"[calib] Loaded camera calibration: {args.calib}")
    else:
        print("[calib] No camera calibration found (undistort disabled).")

    gaze_model: Optional[GazeModel] = None
    gaze_path = Path(args.gaze_model)
    if gaze_path.is_file():
        gaze_model = GazeModel.load(gaze_path)
        print(f"[gaze] Loaded gaze model: {gaze_path}")
    else:
        print("[gaze] No gaze model found — run gaze calibration to enable the pointer.")

    log_path = None
    if args.enable_log:
        log_path = DEFAULT_LOG
    elif args.log_metrics:
        log_path = Path(args.log_metrics)
    tracker = PerformanceTracker(log_path=log_path)
    if log_path:
        print(f"[metrics] Logging metrics to: {log_path}")

    cap = open_camera(args.camera, args.width, args.height)
    main_win = "FaceGate" if args.recognize else "FaceGate — Eye Tracking"
    eye_win = "Eyes"
    gaze_win = "Gaze canvas"
    cv2.namedWindow(main_win, cv2.WINDOW_NORMAL)
    cv2.namedWindow(eye_win, cv2.WINDOW_NORMAL)

    mirror = not args.no_mirror
    use_undistort = bool(args.undistort and camera_matrix is not None)
    smoother = GazeSmoother(alpha=0.3)
    fps = 0.0
    prev = time.time()
    frame_i = 0
    every = max(1, int(args.every))
    last_faces = []
    last_face = None
    last_gaze = None
    latency_ms = 0.0

    print(f"Controls: q quit | m mirror | u undistort | every={every} infer<={args.infer_size}")

    while True:
        ok, frame = grab_latest_frame(cap, flush=1)
        if not ok or frame is None:
            print("Camera frame grab failed.")
            break

        if use_undistort and camera_matrix is not None and dist_coeffs is not None:
            frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
        if mirror:
            frame = cv2.flip(frame, 1)

        frame_i += 1
        do_infer = frame_i % every == 0
        n_pupils = 0
        identity_label = "—"
        gaze = last_gaze

        if do_infer:
            t0 = time.time()
            if isinstance(detector, FaceRecognizer):
                last_faces = detector.analyze(frame, identify=True, max_side=args.infer_size)
            else:
                last_faces = detector.detect(frame, max_side=args.infer_size)

            if last_faces:
                last_face = segment_eyes_for_face(frame, last_faces[0])
                last_face.embedding = last_faces[0].embedding
                last_face.identity = last_faces[0].identity
                last_face.identity_score = last_faces[0].identity_score
                n_pupils = count_valid_pupils(last_face)
                if gaze_model is not None and gaze_model.is_fitted():
                    feature = extract_gaze_features(last_face)
                    if feature is not None:
                        last_gaze = smoother.update(gaze_model.predict(feature))
                        gaze = last_gaze
            else:
                last_face = None
                last_gaze = None
                gaze = None

            latency_ms = (time.time() - t0) * 1000.0
            tracker.record(
                FrameMetrics(
                    fps=fps,
                    latency_ms=latency_ms,
                    n_faces=len(last_faces),
                    n_pupils=n_pupils,
                    gaze_confidence=gaze.confidence if gaze else 0.0,
                )
            )
        elif last_face is not None:
            n_pupils = count_valid_pupils(last_face)

        faces = last_faces
        face = last_face
        if face is not None:
            x1, y1, x2, y2 = face.bbox
            known = face.identity is not None
            color = (0, 220, 0) if known or not args.recognize else (0, 165, 255)
            if args.recognize:
                identity_label = (
                    f"{face.identity} ({face.identity_score:.2f})"
                    if known
                    else f"Unknown ({face.identity_score:.2f})"
                )
            else:
                identity_label = f"det {face.score * 100:.1f}%"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                identity_label,
                (x1, max(24, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
                cv2.LINE_AA,
            )
            draw_eye_segmentation(frame, face)

        fps, prev = smooth_fps_update(fps, prev)
        summary = tracker.summary()
        target_ok = summary["latency_ms_avg"] <= 200.0 if summary["frames"] > 10 else True
        hud = [
            f"FPS: {fps:.1f}  |  infer: {latency_ms:.0f} ms / every {every}",
            f"Faces: {len(faces)}  Pupils: {n_pupils}  ID: {identity_label}",
            f"Gaze: {'ON' if gaze else 'OFF'}  conf={(gaze.confidence if gaze else 0.0):.2f}",
            f"Spec <200ms: {'OK' if target_ok else 'OVER'}",
            "q quit | m mirror | u undistort",
        ]
        flags = []
        if mirror:
            flags.append("mirror")
        if use_undistort:
            flags.append("undistort")
        if args.recognize:
            flags.append("recognize")
        if flags:
            hud.append(" | ".join(flags))
        draw_text_lines(frame, hud, color=(40, 255, 120) if target_ok else (0, 165, 255))

        cv2.imshow(main_win, frame)
        if do_infer:
            cv2.imshow(eye_win, make_eye_debug_panel(frame, face))

        if gaze_model is not None and gaze is not None:
            gw, gh = gaze_model.canvas_size
            canvas = np.zeros((gh, gw, 3), dtype=np.uint8)
            cv2.circle(canvas, (int(gaze.x), int(gaze.y)), 20, (0, 255, 255), 2)
            cv2.circle(canvas, (int(gaze.x), int(gaze.y)), 5, (0, 255, 255), -1)
            draw_text_lines(canvas, ["Gaze canvas"], color=(200, 200, 200))
            cv2.imshow(gaze_win, canvas)

        key = cv2.waitKey(1) & 0xFF
        if should_quit(key) or window_was_closed(main_win):
            break
        if key in (ord("m"), ord("M")):
            mirror = not mirror
            smoother.reset()
        if key in (ord("u"), ord("U")):
            if camera_matrix is None:
                print("No camera calibration available.")
            else:
                use_undistort = not use_undistort
                smoother.reset()

    final = tracker.summary()
    print(
        f"[metrics] Session summary — frames={int(final['frames'])} "
        f"fps_avg={final['fps_avg']:.1f} latency_avg={final['latency_ms_avg']:.1f} ms "
        f"p95={final['latency_ms_p95']:.1f} ms "
        f"target_ok={tracker.meets_targets()}"
    )
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
