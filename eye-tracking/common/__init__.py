"""
Shared helpers for the eye-tracking pipeline (FAIR: Findable / Reusable).

Import submodules explicitly for clarity and lighter dependencies, e.g.::

    from common.gaze_model import GazeModel
    from common.face_detector import FaceDetector
"""

__all__ = [
    "calibration_io",
    "camera_io",
    "face_detector",
    "gaze_model",
    "metrics",
    "pupil",
    "types",
]
