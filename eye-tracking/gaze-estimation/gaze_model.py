"""gaze-estimation library shim — implementation lives in ``common.gaze_model``."""

from common.gaze_model import (  # noqa: F401
    GazeModel,
    GazeSmoother,
    default_calibration_targets,
    extract_gaze_features,
)
