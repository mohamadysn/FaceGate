"""Unit tests for FaceGate desktop service helpers (no models loaded)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.desktop.services import AppServices, IMAGE_EXTS


class ServicesHelperTests(unittest.TestCase):
    def test_image_exts(self) -> None:
        self.assertIn(".jpg", IMAGE_EXTS)
        self.assertIn(".png", IMAGE_EXTS)

    def test_collect_images_files_and_folder(self) -> None:
        services = AppServices()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.jpg").write_bytes(b"x")
            (root / "b.txt").write_text("nope", encoding="utf-8")
            sub = root / "album"
            sub.mkdir()
            (sub / "c.PNG").write_bytes(b"y")
            (sub / "d.webp").write_bytes(b"z")

            files = services.collect_images([str(root / "a.jpg"), str(sub)])
            names = {p.name.lower() for p in files}
            self.assertEqual(names, {"a.jpg", "c.png", "d.webp"})

    def test_effective_threshold_override(self) -> None:
        services = AppServices()
        services.profile_name = "balanced"
        self.assertAlmostEqual(services.effective_threshold(), services.profile.match_threshold)
        services.threshold_override = 0.33
        self.assertAlmostEqual(services.effective_threshold(), 0.33)


if __name__ == "__main__":
    unittest.main()
