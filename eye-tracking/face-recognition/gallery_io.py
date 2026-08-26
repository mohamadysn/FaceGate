#!/usr/bin/env python3
"""CLI: export / import FaceGate gallery archives."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.face_recognition import FaceGallery

DEFAULT_GALLERY = Path(__file__).resolve().parent / "gallery"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FaceGate gallery export / import.")
    parser.add_argument("--gallery", type=str, default=str(DEFAULT_GALLERY))
    sub = parser.add_subparsers(dest="command", required=True)

    exp = sub.add_parser("export", help="Export gallery to a ZIP archive.")
    exp.add_argument("-o", "--output", required=True, help="Output .zip path.")

    imp = sub.add_parser("import", help="Import gallery from a ZIP archive.")
    imp.add_argument("-i", "--input", required=True, help="Input .zip path.")
    imp.add_argument(
        "--replace",
        action="store_true",
        help="Replace the whole local gallery (default: merge by name).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gallery = FaceGallery(args.gallery)
    if args.command == "export":
        out = gallery.export_archive(args.output)
        print(f"Exported {len(gallery)} identities → {out}")
        return
    if args.command == "import":
        n = gallery.import_archive(args.input, merge=not args.replace)
        print(
            f"Imported {n} identities "
            f"({'replaced gallery' if args.replace else 'merged'}). "
            f"Gallery size now: {len(gallery)}."
        )


if __name__ == "__main__":
    main()
