"""Unit tests for face gallery matching (no camera / InsightFace required)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.face_recognition import FaceGallery, cosine_similarity, l2_normalize


class GalleryTests(unittest.TestCase):
    def test_enroll_match_and_unknown(self) -> None:
        rng = np.random.default_rng(42)
        alice = l2_normalize(rng.normal(size=(512,)).astype(np.float32))
        bob = l2_normalize(rng.normal(size=(512,)).astype(np.float32))
        # Near-duplicate of Alice (same identity, small noise).
        alice_query = l2_normalize(alice + rng.normal(0, 0.01, size=512).astype(np.float32))
        stranger = l2_normalize(rng.normal(size=(512,)).astype(np.float32))

        with tempfile.TemporaryDirectory() as tmp:
            gallery = FaceGallery(tmp)
            gallery.match_threshold = 0.35
            gallery.enroll("Alice", [alice])
            gallery.enroll("Bob", [bob])

            # Reload from disk (Interoperable artifact round-trip).
            gallery2 = FaceGallery(tmp)
            self.assertEqual(len(gallery2), 2)

            name, score = gallery2.match(alice_query)
            self.assertEqual(name, "Alice")
            self.assertGreater(score, 0.9)

            name2, score2 = gallery2.match(stranger)
            self.assertIsNone(name2)
            self.assertLess(score2, 0.35)

    def test_margin_rejects_ambiguous(self) -> None:
        # Two nearly identical gallery vectors => ambiguous match.
        base = l2_normalize(np.ones(32, dtype=np.float32))
        twin = l2_normalize(base + np.array([0.01] + [0.0] * 31, dtype=np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            gallery = FaceGallery(tmp)
            gallery.match_threshold = 0.2
            gallery.enroll("A", [base])
            gallery.enroll("B", [twin])
            name, score = gallery.match(base, threshold=0.2, margin=0.05)
            self.assertIsNone(name)
            self.assertGreater(score, 0.2)

    def test_merge_adds_samples(self) -> None:
        rng = np.random.default_rng(7)
        a1 = l2_normalize(rng.normal(size=(32,)).astype(np.float32))
        a2 = l2_normalize(a1 + rng.normal(0, 0.05, size=32).astype(np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            gallery = FaceGallery(tmp)
            gallery.enroll("Alice", [a1])
            self.assertEqual(gallery.entries[0].n_samples, 1)
            gallery.enroll("Alice", [a2], merge=True)
            self.assertEqual(len(gallery), 1)
            self.assertEqual(gallery.entries[0].n_samples, 2)
            name, score = gallery.match(a1, threshold=0.2, margin=0.0)
            self.assertEqual(name, "Alice")
            self.assertGreater(score, 0.5)

    def test_cosine_similarity_identical(self) -> None:
        v = l2_normalize(np.ones(16, dtype=np.float32))
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
