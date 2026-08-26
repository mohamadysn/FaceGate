#!/usr/bin/env python3
"""
Enroll a person from one or more image files (PNG/JPEG).

Example
-------
::

    python enroll_image.py --name Alice --images photo1.jpg photo2.png
    python enroll_image.py --name Alice --images ./photos_alice/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.face_recognition import FaceGallery, FaceRecognizer
from common.quality import assess_face_quality, is_frontal_enough

DEFAULT_GALLERY = Path(__file__).resolve().parent / "gallery"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enroll identity from image file(s).")
    parser.add_argument("--name", type=str, required=True, help="Person name.")
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Image files and/or folders containing images.",
    )
    parser.add_argument("--gallery", type=str, default=str(DEFAULT_GALLERY))
    parser.add_argument("--provider", type=str, default="CPUExecutionProvider")
    parser.add_argument("--min-score", type=float, default=0.45)
    parser.add_argument("--replace", action="store_true", default=True)
    return parser.parse_args()


def collect_images(paths: List[str]) -> List[Path]:
    """Expand files/folders into a sorted list of image paths."""
    out: List[Path] = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.suffix.lower() in IMAGE_EXTS and child.is_file():
                    out.append(child)
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            out.append(path)
        else:
            print(f"Skip (not an image): {path}")
    # unique preserve order
    seen = set()
    unique = []
    for p in out:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def main() -> None:
    args = parse_args()
    images = collect_images(args.images)
    if not images:
        print("No images found.")
        return

    gallery = FaceGallery(args.gallery)
    recognizer = FaceRecognizer(
        model_name="buffalo_l",
        provider=args.provider,
        min_score=args.min_score,
        gallery=gallery,
        det_size=640,
    )

    embeddings = []
    weights = []
    used = []
    rejected = []

    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            rejected.append((path.name, "unreadable"))
            continue

        faces = recognizer.analyze(
            frame,
            identify=False,
            max_side=960,
            refine_small=True,
            small_face_px=140,
        )
        if not faces or faces[0].embedding is None:
            rejected.append((path.name, "no_face"))
            continue

        face = faces[0]
        report = assess_face_quality(
            frame,
            face,
            min_face_px=40,
            min_det_score=args.min_score,
            min_sharpness=10.0,
        )
        if not report.ok:
            # Still accept image enrollment with a warning — photos vary a lot.
            print(f"Warning soft-accept {path.name}: {report.reason} (q={report.score:.2f})")
        if not is_frontal_enough(face):
            print(f"Warning: {path.name} may not be frontal enough.")

        embeddings.append(face.embedding.copy())
        weights.append(max(0.05, report.score if report.ok else 0.2))
        used.append(path.name)
        print(f"OK  {path.name}  det={face.score:.2f}  size={min(face.bbox[2]-face.bbox[0], face.bbox[3]-face.bbox[1])}px")

    if not embeddings:
        print("No usable face embeddings. Enrollment aborted.")
        for name, reason in rejected:
            print(f"  rejected {name}: {reason}")
        return

    entry = gallery.enroll(args.name, embeddings, replace=True, weights=weights)
    print(
        f"\nEnrolled '{entry.name}' from {len(embeddings)} image(s): {', '.join(used)}"
    )
    if rejected:
        print("Rejected:")
        for name, reason in rejected:
            print(f"  {name}: {reason}")
    print(f"Gallery -> {gallery.json_path}  (total identities: {len(gallery)})")


if __name__ == "__main__":
    main()
