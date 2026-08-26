#!/usr/bin/env python3
"""
Powerful & efficient live face recognition.

Uses runtime profiles + threaded inference + face tracking + quality gates.

Profiles
--------
- ``fast``     : max FPS (buffalo_s, lighter settings)
- ``balanced`` : default (buffalo_l)
- ``accurate`` : best recognition quality

Example
-------
::

    python recognize_live.py --profile balanced --provider CPUExecutionProvider
    python recognize_live.py --profile accurate --provider CUDAExecutionProvider
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

from common.camera_io import (
    create_fitted_window,
    draw_text_lines,
    fit_frame_to_window,
    grab_latest_frame,
    open_camera,
    should_quit,
    smooth_fps_update,
    window_was_closed,
)
from common.engine import RealtimeRecognitionEngine
from common.face_detector import FaceDetector
from common.face_recognition import FaceGallery, FaceRecognizer
from common.profiles import PROFILES, get_profile

DEFAULT_GALLERY = Path(__file__).resolve().parent / "gallery"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Powerful live face recognition.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--provider", type=str, default="CPUExecutionProvider")
    parser.add_argument(
        "--profile",
        type=str,
        default="balanced",
        choices=sorted(PROFILES.keys()),
        help="Speed/accuracy trade-off.",
    )
    parser.add_argument("--gallery", type=str, default=str(DEFAULT_GALLERY))
    parser.add_argument("--threshold", type=float, default=None, help="Override match threshold.")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--width", type=int, default=None, help="Override capture width.")
    parser.add_argument("--height", type=int, default=None, help="Override capture height.")
    parser.add_argument(
        "--max-faces",
        type=int,
        default=1,
        help="Only keep the N largest faces (default 1 = access-control mode).",
    )
    parser.add_argument(
        "--min-face-ratio",
        type=float,
        default=0.10,
        help="Ignore faces smaller than this fraction of the frame (filters wall photos).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = get_profile(args.profile)
    gallery = FaceGallery(args.gallery)
    if len(gallery) == 0:
        print(f"Gallery is empty: {args.gallery}")
        print("Enroll someone first:")
        print("  python enroll.py --name Alice --provider CPUExecutionProvider")
        return

    threshold = profile.match_threshold if args.threshold is None else float(args.threshold)
    gallery.match_threshold = threshold

    print(f"Loading model '{profile.model_name}' ({profile.name})…")
    try:
        recognizer = FaceRecognizer(
            model_name=profile.model_name,
            provider=args.provider,
            min_score=profile.min_det_score,
            gallery=gallery,
            match_threshold=threshold,
            match_margin=profile.match_margin,
            det_size=profile.det_size,
        )
    except Exception as exc:
        if profile.model_name != "buffalo_l":
            print(f"Model '{profile.model_name}' unavailable ({exc}). Falling back to buffalo_l.")
            recognizer = FaceRecognizer(
                model_name="buffalo_l",
                provider=args.provider,
                min_score=profile.min_det_score,
                gallery=gallery,
                match_threshold=threshold,
                match_margin=profile.match_margin,
                det_size=profile.det_size,
            )
        else:
            raise

    detector = FaceDetector(
        backend="insightface",
        model_name=recognizer.model_name,
        provider=args.provider,
        min_score=profile.min_det_score,
        det_size=min(256, profile.det_size),
    )
    engine = RealtimeRecognitionEngine(
        recognizer,
        detector,
        profile,
        max_faces=args.max_faces,
        min_face_ratio=args.min_face_ratio,
    )
    engine.start()

    width = args.width or profile.width
    height = args.height or profile.height
    cap = open_camera(args.camera, width, height, fps=30)
    window = "FaceGate"
    win_w, win_h = create_fitted_window(window, width, height, max_screen_fraction=0.80)

    mirror = not args.no_mirror
    fps = 0.0
    prev = time.time()

    print(f"Gallery ({len(gallery)}): {', '.join(gallery.names())}")
    print(
        f"Profile={profile.name} | {width}x{height} | boxes=every-frame | "
        f"ID every={profile.every} | thr={threshold:.2f} | q=quit"
    )

    try:
        while True:
            ok, frame = grab_latest_frame(cap, flush=1)
            if not ok or frame is None:
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            # Boxes updated NOW on this frame (responsive to camera motion).
            tracks = engine.update_from_frame(frame)
            # Identity refreshed in background.
            engine.submit_frame(frame)
            stats = engine.get_stats()
            fps, prev = smooth_fps_update(fps, prev)

            for tr in tracks:
                x1, y1, x2, y2 = tr.bbox
                known = tr.identity is not None
                if known:
                    color = (0, 220, 0)
                    label = f"{tr.identity} ({tr.identity_score:.2f})"
                elif tr.pending_name and tr.pending_hits > 0:
                    color = (0, 200, 255)
                    label = f"... {tr.pending_name}? ({tr.pending_hits}/{profile.confirm_hits})"
                else:
                    color = (0, 165, 255)
                    label = f"Unknown ({tr.identity_score:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(24, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            draw_text_lines(
                frame,
                [
                    f"FPS: {fps:.1f} | det: {stats.det_ms:.0f} ms | ID: {stats.infer_ms:.0f} ms",
                    f"Faces: {len(tracks)} | Gallery: {len(gallery)} | profile: {profile.name}",
                    "q: quit",
                ],
            )
            cv2.imshow(window, fit_frame_to_window(frame, win_w, win_h))
            key = cv2.waitKey(1) & 0xFF
            if should_quit(key) or window_was_closed(window):
                break
    finally:
        engine.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
