"""Unit tests for runtime profiles, metrics, calibration I/O, camera helpers."""

from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.calibration_io import load_camera_coefficients, save_camera_coefficients
from common.camera_io import (
    KEY_QUIT,
    fit_frame_to_window,
    preferred_camera_backends,
    resize_for_inference,
    should_quit,
    smooth_fps_update,
)
from common.face_detector import clip_bbox, eye_rois_from_landmarks
from common.metrics import PerformanceTracker
from common.profiles import PROFILES, get_profile
from common.types import FrameMetrics


class ProfileTests(unittest.TestCase):
    def test_known_profiles(self) -> None:
        self.assertEqual(set(PROFILES), {"fast", "balanced", "accurate"})
        for name in PROFILES:
            profile = get_profile(name)
            self.assertEqual(profile.name, name)
            self.assertGreater(profile.width, 0)
            self.assertGreater(profile.match_threshold, 0)

    def test_profile_case_insensitive(self) -> None:
        self.assertEqual(get_profile("BALANCED").name, "balanced")

    def test_unknown_profile_raises(self) -> None:
        with self.assertRaises(KeyError):
            get_profile("turbo")


class MetricsTests(unittest.TestCase):
    def test_summary_and_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "session.jsonl"
            tracker = PerformanceTracker(window=10, log_path=log)
            for _ in range(5):
                tracker.record(FrameMetrics(fps=30.0, latency_ms=50.0, n_faces=1, n_pupils=2))
            summary = tracker.summary()
            self.assertEqual(summary["frames"], 5.0)
            self.assertAlmostEqual(summary["fps_avg"], 30.0)
            self.assertAlmostEqual(summary["latency_ms_avg"], 50.0)
            self.assertTrue(tracker.meets_targets(max_latency_ms=100, min_fps=20))
            self.assertFalse(tracker.meets_targets(max_latency_ms=10, min_fps=20))
            self.assertEqual(len(log.read_text(encoding="utf-8").strip().splitlines()), 5)


class CalibrationIOTests(unittest.TestCase):
    def test_save_load_roundtrip(self) -> None:
        k = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]], dtype=np.float64)
        d = np.array([[0.1, -0.2, 0.0, 0.0, 0.05]], dtype=np.float64)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "camera_matrix.yml"
            save_camera_coefficients(k, d, path)
            k2, d2 = load_camera_coefficients(path)
            self.assertIsNotNone(k2)
            self.assertIsNotNone(d2)
            np.testing.assert_allclose(k2, k, rtol=1e-5)
            np.testing.assert_allclose(d2.reshape(-1)[:5], d.reshape(-1)[:5], rtol=1e-5)

    def test_missing_file_returns_none(self) -> None:
        k, d = load_camera_coefficients("/tmp/does-not-exist-facegate.yml")
        self.assertIsNone(k)
        self.assertIsNone(d)


class CameraHelperTests(unittest.TestCase):
    def test_preferred_backends_non_empty(self) -> None:
        backends = preferred_camera_backends()
        self.assertGreaterEqual(len(backends), 1)

    def test_backends_by_platform(self) -> None:
        with mock.patch("common.camera_io.sys.platform", "win32"):
            self.assertIn(cv2.CAP_DSHOW, preferred_camera_backends())
        with mock.patch("common.camera_io.sys.platform", "darwin"):
            self.assertIn(cv2.CAP_AVFOUNDATION, preferred_camera_backends())
        with mock.patch("common.camera_io.sys.platform", "linux"):
            self.assertIn(cv2.CAP_V4L2, preferred_camera_backends())

    def test_resize_for_inference(self) -> None:
        frame = np.zeros((1000, 800, 3), dtype=np.uint8)
        small, scale = resize_for_inference(frame, max_side=400)
        self.assertLessEqual(max(small.shape[:2]), 400)
        self.assertGreater(scale, 1.0)
        same, scale2 = resize_for_inference(frame, max_side=2000)
        self.assertEqual(scale2, 1.0)
        self.assertEqual(same.shape, frame.shape)

    def test_fit_frame_to_window(self) -> None:
        frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
        fitted = fit_frame_to_window(frame, 200, 200)
        self.assertLessEqual(fitted.shape[0], 200)
        self.assertLessEqual(fitted.shape[1], 200)

    def test_should_quit(self) -> None:
        self.assertTrue(should_quit(ord("q")))
        self.assertTrue(should_quit(27))
        self.assertFalse(should_quit(ord("a")))
        self.assertEqual(KEY_QUIT, {ord("q"), ord("Q"), 27})

    def test_smooth_fps_update(self) -> None:
        t0 = time.time() - 0.05
        fps, t1 = smooth_fps_update(0.0, t0, alpha=1.0)
        self.assertGreater(fps, 10.0)
        self.assertGreater(t1, t0)


class FaceDetectorHelperTests(unittest.TestCase):
    def test_clip_bbox(self) -> None:
        self.assertEqual(clip_bbox(-10, -5, 50, 40, 100, 80), (0, 0, 50, 40))
        self.assertIsNone(clip_bbox(10, 10, 10, 20, 100, 100))

    def test_eye_rois_from_landmarks(self) -> None:
        landmarks = np.array(
            [[30.0, 40.0], [70.0, 40.0], [50.0, 55.0], [35.0, 70.0], [65.0, 70.0]],
            dtype=np.float32,
        )
        rois = eye_rois_from_landmarks(landmarks, (120, 100, 3))
        self.assertEqual(len(rois), 2)
        sides = {r[0] for r in rois}
        self.assertEqual(sides, {"left", "right"})
        for side, bbox, point in rois:
            self.assertEqual(len(bbox), 4)
            self.assertEqual(len(point), 2)


if __name__ == "__main__":
    unittest.main()
