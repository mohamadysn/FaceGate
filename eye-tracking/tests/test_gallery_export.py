"""Tests for FaceGallery ZIP export / import."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.face_recognition import FaceGallery, l2_normalize


class GalleryExportImportTests(unittest.TestCase):
    def _vec(self, seed: int, dim: int = 32) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return l2_normalize(rng.normal(size=(dim,)).astype(np.float32))

    def test_export_import_replace(self) -> None:
        alice = self._vec(1)
        bob = self._vec(2)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            dst = Path(tmp) / "dst"
            archive = Path(tmp) / "gallery.zip"
            g1 = FaceGallery(src)
            g1.enroll("Alice", [alice])
            g1.enroll("Bob", [bob])
            out = g1.export_archive(archive)
            self.assertTrue(out.is_file())

            g2 = FaceGallery(dst)
            g2.enroll("Carol", [self._vec(3)])
            n = g2.import_archive(archive, merge=False)
            self.assertEqual(n, 2)
            self.assertEqual(len(g2), 2)
            names = {e.name for e in g2.entries}
            self.assertEqual(names, {"Alice", "Bob"})

    def test_export_import_merge(self) -> None:
        alice = self._vec(11)
        bob = self._vec(12)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            dst = Path(tmp) / "dst"
            archive = Path(tmp) / "g.zip"
            FaceGallery(src).enroll("Alice", [alice])
            FaceGallery(src).enroll("Bob", [bob])
            # Re-open so export sees both.
            FaceGallery(src).export_archive(archive)

            g_dst = FaceGallery(dst)
            g_dst.enroll("Alice", [self._vec(99)])
            g_dst.enroll("Dana", [self._vec(13)])
            n = g_dst.import_archive(archive, merge=True)
            self.assertEqual(n, 2)
            names = {e.name for e in g_dst.entries}
            self.assertEqual(names, {"Alice", "Bob", "Dana"})
            name, score = g_dst.match(alice, threshold=0.2, margin=0.0)
            self.assertEqual(name, "Alice")
            self.assertGreater(score, 0.9)

    def test_import_missing_archive_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            g = FaceGallery(tmp)
            with self.assertRaises(FileNotFoundError):
                g.import_archive(Path(tmp) / "missing.zip")


if __name__ == "__main__":
    unittest.main()
