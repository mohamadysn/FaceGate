"""Optional camera integration tests (skipped by default).

Enable when a webcam is available:

    FACEGATE_CAMERA_TESTS=1 python -m unittest tests.test_camera_integration -v

Or probe automatically (skip if no device):

    FACEGATE_CAMERA_TESTS=auto python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.camera_io import grab_latest_frame, open_camera, preferred_camera_backends


def _camera_mode() -> str:
    return os.environ.get("FACEGATE_CAMERA_TESTS", "0").strip().lower()


def _try_open(index: int = 0):
    try:
        return open_camera(index=index, width=320, height=240, fps=15)
    except Exception:
        return None


def _should_run_camera_tests() -> bool:
    mode = _camera_mode()
    if mode in {"0", "false", "no", "off", "skip", ""}:
        return False
    if mode in {"1", "true", "yes", "on", "force"}:
        return True
    if mode == "auto":
        cap = _try_open(0)
        if cap is None:
            return False
        cap.release()
        return True
    return False


@unittest.skipUnless(_should_run_camera_tests(), "No camera / FACEGATE_CAMERA_TESTS disabled")
class CameraIntegrationTests(unittest.TestCase):
    def test_preferred_backends_nonempty(self) -> None:
        self.assertGreater(len(preferred_camera_backends()), 0)

    def test_open_and_grab_frame(self) -> None:
        cap = open_camera(index=0, width=320, height=240, fps=15)
        self.addCleanup(cap.release)
        self.assertTrue(cap.isOpened())
        ok, frame = grab_latest_frame(cap, flush=2)
        self.assertTrue(ok, "Camera opened but failed to read a frame")
        self.assertIsNotNone(frame)
        self.assertEqual(frame.ndim, 3)
        self.assertGreater(frame.shape[0], 0)
        self.assertGreater(frame.shape[1], 0)


if __name__ == "__main__":
    unittest.main()
