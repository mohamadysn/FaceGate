"""Unit tests for pure gaze / pupil helpers (no camera required)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.gaze_model import GazeModel, default_calibration_targets, extract_gaze_features
from common.pupil import _pupil_from_roi
from common.types import EyeSample, FaceSample


class GazeModelTests(unittest.TestCase):
    def test_default_targets_count(self) -> None:
        targets = default_calibration_targets(1000, 800, grid=3)
        self.assertEqual(len(targets), 9)

    def test_fit_and_predict_roundtrip(self) -> None:
        # Synthetic identity-like mapping: feature[4:6] roughly encode position.
        rng = np.random.default_rng(0)
        features = []
        targets = []
        for x in (100.0, 500.0, 900.0):
            for y in (80.0, 400.0, 720.0):
                # Fake eye-local features + face-relative channels correlated with screen.
                f = rng.normal(0, 0.05, size=6)
                f[4] = (x / 1000.0) * 2 - 1
                f[5] = (y / 800.0) * 2 - 1
                features.append(f)
                targets.append((x, y))

        model = GazeModel(canvas_size=(1000, 800))
        model.fit(features, targets, ridge=1e-2)
        pred = model.predict(features[4])
        self.assertLess(abs(pred.x - targets[4][0]), 80.0)
        self.assertLess(abs(pred.y - targets[4][1]), 80.0)

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gaze_model_test.json"
            model.save(path)
            loaded = GazeModel.load(path)
            pred2 = loaded.predict(features[4])
            self.assertAlmostEqual(pred.x, pred2.x, places=3)
            self.assertAlmostEqual(pred.y, pred2.y, places=3)

    def test_extract_features_requires_pupil(self) -> None:
        face = FaceSample(
            bbox=(0, 0, 100, 100),
            score=0.9,
            eyes=[
                EyeSample(side="left", bbox=(10, 20, 40, 40), pupil=None),
                EyeSample(side="right", bbox=(60, 20, 90, 40), pupil=None),
            ],
        )
        self.assertIsNone(extract_gaze_features(face))

        face.eyes[0].pupil = (25.0, 30.0)
        face.eyes[0].pupil_confidence = 0.9
        face.eyes[1].pupil = (75.0, 30.0)
        face.eyes[1].pupil_confidence = 0.9
        feat = extract_gaze_features(face)
        self.assertIsNotNone(feat)
        self.assertEqual(feat.shape, (6,))


class PupilTests(unittest.TestCase):
    def test_dark_blob_detected(self) -> None:
        # Realistic open-eye crop: sclera + iris + pupil.
        img = np.full((50, 70, 3), 140, dtype=np.uint8)
        cv2 = __import__("cv2")
        cv2.ellipse(img, (35, 25), (30, 16), 0, 0, 360, (220, 220, 220), -1)
        cv2.circle(img, (35, 25), 10, (90, 70, 50), -1)
        cv2.circle(img, (35, 25), 4, (10, 10, 10), -1)
        pupil, conf, _ = _pupil_from_roi(img)
        self.assertIsNotNone(pupil)
        self.assertGreater(conf, 0.2)
        self.assertLess(abs(pupil[0] - 35), 8)
        self.assertLess(abs(pupil[1] - 25), 8)

    def test_uniform_occlusion_rejected(self) -> None:
        # Hand-like flat dark patch should not invent a pupil.
        img = np.full((60, 80, 3), 40, dtype=np.uint8)
        pupil, conf, _ = _pupil_from_roi(img)
        self.assertIsNone(pupil)
        self.assertLessEqual(conf, 0.0)

    def test_textured_hand_rejected(self) -> None:
        rng = np.random.default_rng(1)
        img = np.clip(rng.normal(95, 8, (50, 70, 3)), 40, 160).astype(np.uint8)
        pupil, conf, _ = _pupil_from_roi(img)
        self.assertIsNone(pupil)


if __name__ == "__main__":
    unittest.main()
