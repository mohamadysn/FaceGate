"""Unit tests for face quality heuristics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.quality import assess_face_quality, bbox_size, estimate_sharpness, is_frontal_enough
from common.types import FaceSample


class QualityTests(unittest.TestCase):
    def test_bbox_size(self) -> None:
        self.assertEqual(bbox_size((10, 20, 40, 50)), (30, 30))

    def test_sharpness_blur_vs_edges(self) -> None:
        blurry = np.full((64, 64), 128, dtype=np.uint8)
        sharp = np.zeros((64, 64), dtype=np.uint8)
        sharp[:, ::2] = 255
        self.assertGreater(estimate_sharpness(sharp), estimate_sharpness(blurry))

    def test_assess_rejects_small_and_blurry(self) -> None:
        frame = np.full((200, 200, 3), 80, dtype=np.uint8)
        small = FaceSample(bbox=(10, 10, 30, 30), score=0.9)
        report = assess_face_quality(frame, small, min_face_px=80)
        self.assertFalse(report.ok)
        self.assertEqual(report.reason, "face_too_small")

        # Large but flat (blurry) crop
        face = FaceSample(bbox=(20, 20, 160, 160), score=0.9)
        report2 = assess_face_quality(frame, face, min_face_px=40, min_sharpness=50.0)
        self.assertFalse(report2.ok)
        self.assertEqual(report2.reason, "blurry")

    def test_assess_accepts_sharp_face(self) -> None:
        frame = np.full((240, 240, 3), 120, dtype=np.uint8)
        # Draw high-contrast pattern in face region
        cv2.rectangle(frame, (40, 40), (200, 200), (20, 20, 20), -1)
        for i in range(40, 200, 4):
            cv2.line(frame, (i, 40), (i, 200), (220, 220, 220), 1)
        face = FaceSample(bbox=(40, 40, 200, 200), score=0.95)
        report = assess_face_quality(frame, face, min_face_px=50, min_sharpness=20.0)
        self.assertTrue(report.ok)
        self.assertGreater(report.score, 0.0)

    def test_assess_low_det_score(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        face = FaceSample(bbox=(10, 10, 90, 90), score=0.1)
        report = assess_face_quality(frame, face, min_face_px=20, min_det_score=0.5, min_sharpness=0.0)
        self.assertFalse(report.ok)
        self.assertEqual(report.reason, "low_det_score")

    def test_frontal_enough(self) -> None:
        face = FaceSample(bbox=(0, 0, 100, 100), score=0.9)
        self.assertTrue(is_frontal_enough(face))  # no landmarks → permissive
        face.landmarks = np.array([[20.0, 40.0], [80.0, 40.0], [50.0, 60.0], [30.0, 80.0], [70.0, 80.0]])
        self.assertTrue(is_frontal_enough(face))
        face.landmarks = np.array([[50.0, 40.0], [52.0, 40.0], [51.0, 60.0], [50.0, 80.0], [52.0, 80.0]])
        self.assertFalse(is_frontal_enough(face))


if __name__ == "__main__":
    unittest.main()
