"""Additional FaceGallery edge cases (no InsightFace required)."""

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


class GalleryEdgeTests(unittest.TestCase):
    def test_empty_gallery_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gallery = FaceGallery(tmp)
            name, score = gallery.match(l2_normalize(np.ones(8, dtype=np.float32)))
            self.assertIsNone(name)
            self.assertEqual(score, -1.0)

    def test_enroll_rejects_empty_name_and_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gallery = FaceGallery(tmp)
            vec = l2_normalize(np.ones(16, dtype=np.float32))
            with self.assertRaises(ValueError):
                gallery.enroll("  ", [vec])
            with self.assertRaises(ValueError):
                gallery.enroll("Alice", [])

    def test_replace_overwrites_identity(self) -> None:
        rng = np.random.default_rng(3)
        a = l2_normalize(rng.normal(size=(32,)).astype(np.float32))
        b = l2_normalize(rng.normal(size=(32,)).astype(np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            gallery = FaceGallery(tmp)
            gallery.enroll("Alice", [a], replace=True)
            gallery.enroll("Alice", [b], replace=True)
            self.assertEqual(len(gallery), 1)
            self.assertEqual(gallery.entries[0].n_samples, 1)
            name, score = gallery.match(b, threshold=0.2, margin=0.0)
            self.assertEqual(name, "Alice")
            self.assertGreater(score, 0.9)

    def test_replace_false_raises_on_duplicate(self) -> None:
        vec = l2_normalize(np.ones(16, dtype=np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            gallery = FaceGallery(tmp)
            gallery.enroll("Alice", [vec], replace=True)
            with self.assertRaises(ValueError):
                gallery.enroll("Alice", [vec], replace=False)

    def test_remove_identity(self) -> None:
        vec = l2_normalize(np.ones(16, dtype=np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            gallery = FaceGallery(tmp)
            gallery.enroll("Alice", [vec])
            gallery.enroll("Bob", [l2_normalize(np.arange(16, dtype=np.float32) + 1)])
            self.assertTrue(gallery.remove("Alice"))
            self.assertFalse(gallery.remove("Alice"))
            self.assertEqual(gallery.names(), ["Bob"])
            # Persist remove
            reloaded = FaceGallery(tmp)
            self.assertEqual(reloaded.names(), ["Bob"])

    def test_weighted_enroll_prefers_high_weight(self) -> None:
        # Two opposite directions; heavy weight on first should keep match closer to first.
        v1 = l2_normalize(np.array([1.0] + [0.0] * 15, dtype=np.float32))
        v2 = l2_normalize(np.array([-1.0] + [0.0] * 15, dtype=np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            gallery = FaceGallery(tmp)
            gallery.enroll("Alice", [v1, v2], weights=[10.0, 0.1])
            name, score = gallery.match(v1, threshold=0.0, margin=0.0)
            self.assertEqual(name, "Alice")
            self.assertGreater(score, 0.5)


if __name__ == "__main__":
    unittest.main()
