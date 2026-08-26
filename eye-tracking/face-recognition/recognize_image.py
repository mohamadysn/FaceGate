#!/usr/bin/env python3
"""
Recognize people in a PNG/JPEG image against the gallery.

Example
-------
::

    python recognize_image.py --image test.jpg
    python recognize_image.py --image test.jpg --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.camera_io import create_fitted_window, fit_frame_to_window
from common.face_recognition import DEFAULT_MATCH_THRESHOLD, FaceGallery, FaceRecognizer

DEFAULT_GALLERY = Path(__file__).resolve().parent / "gallery"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recognize faces in an image file.")
    parser.add_argument("--image", type=str, required=True, help="Path to PNG/JPEG image.")
    parser.add_argument("--gallery", type=str, default=str(DEFAULT_GALLERY))
    parser.add_argument("--provider", type=str, default="CPUExecutionProvider")
    parser.add_argument("--threshold", type=float, default=DEFAULT_MATCH_THRESHOLD)
    parser.add_argument("--show", action="store_true", help="Show annotated image window.")
    parser.add_argument("--save", type=str, default="", help="Optional output annotated image path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        print(f"Image not found: {image_path}")
        return

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Could not read image: {image_path}")
        return

    gallery = FaceGallery(args.gallery)
    if len(gallery) == 0:
        print(f"Gallery empty: {args.gallery}")
        print("Enroll first with enroll.py or enroll_image.py")
        return

    gallery.match_threshold = args.threshold
    recognizer = FaceRecognizer(
        model_name="buffalo_l",
        provider=args.provider,
        gallery=gallery,
        match_threshold=args.threshold,
        match_margin=0.08,
        det_size=640,
        min_score=0.45,
    )

    faces = recognizer.analyze(frame, identify=True, max_side=960, refine_small=True)
    if not faces:
        print("No face detected in the image.")
        return

    print(f"Image: {image_path.name}")
    print(f"Gallery ({len(gallery)}): {', '.join(gallery.names())}")
    print(f"Faces found: {len(faces)}\n")

    for i, face in enumerate(faces, start=1):
        x1, y1, x2, y2 = face.bbox
        if face.identity is not None:
            label = f"{face.identity} ({face.identity_score:.2f})"
            color = (0, 220, 0)
            print(f"[{i}] RECOGNIZED -> {face.identity}  score={face.identity_score:.3f}  box=({x1},{y1})-({x2},{y2})")
        else:
            label = f"Unknown ({face.identity_score:.2f})"
            color = (0, 165, 255)
            print(f"[{i}] Unknown  best_score={face.identity_score:.3f}  box=({x1},{y1})-({x2},{y2})")

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if args.save:
        out = Path(args.save).expanduser().resolve()
        cv2.imwrite(str(out), frame)
        print(f"\nSaved annotated image: {out}")

    if args.show:
        h, w = frame.shape[:2]
        win = "FaceGate — Recognize image"
        win_w, win_h = create_fitted_window(win, w, h, max_screen_fraction=0.85)
        cv2.imshow(win, fit_frame_to_window(frame, win_w, win_h))
        print("Press any key in the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
