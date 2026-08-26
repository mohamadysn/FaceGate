#!/usr/bin/env python3
"""
Fast enrollment UI (near + far).

Preview uses lightweight **detection only** every frame (smooth camera).
Full embedding extraction runs only when a sample is accepted.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

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
    window_was_closed,
)
from common.face_detector import FaceDetector
from common.face_recognition import FaceGallery, FaceRecognizer
from common.quality import assess_face_quality, is_frontal_enough
from common.types import FaceSample

DEFAULT_GALLERY = Path(__file__).resolve().parent / "gallery"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enroll a face at near and far distances (smooth preview).")
    parser.add_argument("--name", type=str, required=True, help="Person name to enroll.")
    parser.add_argument("--near-samples", type=int, default=12, help="Close-distance samples.")
    parser.add_argument("--far-samples", type=int, default=12, help="Far/small-face samples.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--provider", type=str, default="CPUExecutionProvider")
    parser.add_argument("--width", type=int, default=640, help="Capture width (lower = smoother).")
    parser.add_argument("--height", type=int, default=480, help="Capture height.")
    parser.add_argument("--gallery", type=str, default=str(DEFAULT_GALLERY))
    parser.add_argument("--min-score", type=float, default=0.45)
    parser.add_argument("--no-mirror", action="store_true")
    return parser.parse_args()


def _embed_once(recognizer: FaceRecognizer, frame, refine_small: bool) -> FaceSample | None:
    """Run the heavy embedding model once (only when capturing)."""
    faces = recognizer.analyze(
        frame,
        identify=False,
        max_side=640,
        refine_small=refine_small,
        small_face_px=140,
    )
    if not faces or faces[0].embedding is None:
        return None
    return faces[0]


def _capture_phase(
    *,
    cap,
    detector: FaceDetector,
    recognizer: FaceRecognizer,
    window: str,
    win_w: int,
    win_h: int,
    name: str,
    phase: str,
    needed: int,
    mirror: bool,
    min_score: float,
    min_face_px: int,
    min_sharpness: float,
    hint: str,
) -> Tuple[List, List[float]]:
    embeddings: List = []
    weights: List[float] = []
    rejected = 0
    last_capture = 0.0
    refine_small = phase == "FAR"

    print(f"[{phase}] {hint}")
    while len(embeddings) < needed:
        ok, frame = grab_latest_frame(cap, flush=1)
        if not ok or frame is None:
            break
        if mirror:
            frame = cv2.flip(frame, 1)

        # Fast path: detection only for smooth preview.
        detections = detector.detect(frame, max_side=480)
        display = frame
        status = "no face"
        color = (0, 165, 255)
        good_preview = False

        if detections:
            det = detections[0]
            x1, y1, x2, y2 = det.bbox
            short = min(x2 - x1, y2 - y1)
            report = assess_face_quality(
                frame,
                det,
                min_face_px=min_face_px,
                min_det_score=min_score,
                min_sharpness=min_sharpness,
            )
            frontal = is_frontal_enough(det)
            good_preview = report.ok and frontal

            if phase == "FAR" and short > 180:
                good_preview = False
                status = "move farther (face still too big)"
            elif phase == "NEAR" and short < 100:
                good_preview = False
                status = "move closer (face too small)"
            elif good_preview:
                status = f"READY size={short}px — capturing…"
                color = (0, 255, 0)
            else:
                reason = report.reason if not report.ok else "not_frontal"
                status = f"reject:{reason}"

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 220, 0) if good_preview else (0, 165, 255),
                2,
            )

            # Heavy embedding only when the pose looks good + paced.
            now = time.time()
            if good_preview and (now - last_capture) >= 0.35:
                face = _embed_once(recognizer, frame, refine_small=refine_small)
                if face is not None and face.embedding is not None:
                    q = assess_face_quality(
                        frame,
                        face,
                        min_face_px=min_face_px,
                        min_det_score=min_score,
                        min_sharpness=min_sharpness,
                    )
                    if q.ok and is_frontal_enough(face):
                        embeddings.append(face.embedding.copy())
                        weights.append(max(0.05, q.score))
                        last_capture = now
                        status = f"SAVED {len(embeddings)}/{needed} q={q.score:.2f}"
                        color = (0, 255, 0)
                    else:
                        rejected += 1
                        status = f"reject:{q.reason if not q.ok else 'not_frontal'}"
                else:
                    rejected += 1
                    status = "reject:embed_failed"

        draw_text_lines(
            display,
            [
                f"Enrolling: {name}  [{phase}]",
                f"Accepted: {len(embeddings)}/{needed}  rejected: {rejected}",
                hint,
                f"Status: {status}",
                "q: cancel",
            ],
            color=color,
        )
        cv2.imshow(window, fit_frame_to_window(display, win_w, win_h))
        key = cv2.waitKey(1) & 0xFF
        if should_quit(key) or window_was_closed(window):
            raise KeyboardInterrupt("Enrollment cancelled.")

    return embeddings, weights


def main() -> None:
    args = parse_args()
    gallery = FaceGallery(args.gallery)

    # Light detector = smooth camera preview.
    detector = FaceDetector(
        backend="insightface",
        model_name="buffalo_l",
        provider=args.provider,
        min_score=args.min_score,
        det_size=320,
    )
    # Recognizer used only when saving a sample.
    recognizer = FaceRecognizer(
        model_name="buffalo_l",
        provider=args.provider,
        min_score=args.min_score,
        gallery=gallery,
        det_size=320,
    )

    cap = open_camera(args.camera, args.width, args.height, fps=30)
    window = "FaceGate — Enroll"
    win_w, win_h = create_fitted_window(window, args.width, args.height, max_screen_fraction=0.80)
    mirror = not args.no_mirror
    print("Smooth preview ON (detection). Embedding runs only when capturing a sample.")

    try:
        near_emb, near_w = _capture_phase(
            cap=cap,
            detector=detector,
            recognizer=recognizer,
            window=window,
            win_w=win_w,
            win_h=win_h,
            name=args.name,
            phase="NEAR",
            needed=args.near_samples,
            mirror=mirror,
            min_score=args.min_score,
            min_face_px=90,
            min_sharpness=30.0,
            hint="Phase 1/2 NEAR: stand close, face the camera",
        )
        far_emb, far_w = _capture_phase(
            cap=cap,
            detector=detector,
            recognizer=recognizer,
            window=window,
            win_w=win_w,
            win_h=win_h,
            name=args.name,
            phase="FAR",
            needed=args.far_samples,
            mirror=mirror,
            min_score=max(0.35, args.min_score - 0.1),
            min_face_px=40,
            min_sharpness=12.0,
            hint="Phase 2/2 FAR: step back until face is small, stay frontal",
        )
    except KeyboardInterrupt:
        print("Enrollment cancelled.")
        cap.release()
        cv2.destroyAllWindows()
        return

    embeddings = near_emb + far_emb
    weights = near_w + far_w
    entry = gallery.enroll(args.name, embeddings, replace=True, weights=weights)
    print(
        f"Enrolled '{entry.name}' with {entry.n_samples} samples "
        f"(near={len(near_emb)}, far={len(far_emb)}). Gallery size={len(gallery)}."
    )
    print(f"Saved: {gallery.json_path}")
    print("Now test with: python recognize_live.py --profile balanced")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
