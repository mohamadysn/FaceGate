"""
Runtime performance metrics (FAIR: Findable quality signals).

Tracks FPS, latency, detection counts and can export a JSONL session log for
later analysis / optimization work.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Deque, Dict, Optional, Union

from .types import FrameMetrics

PathLike = Union[str, Path]


class PerformanceTracker:
    """
    Collect per-frame metrics and optional JSONL persistence.

    Parameters
    ----------
    window :
        Rolling window size for averages.
    log_path :
        If set, append one JSON object per ``record`` call.
    """

    def __init__(self, window: int = 120, log_path: Optional[PathLike] = None) -> None:
        self.window = window
        self._latencies: Deque[float] = deque(maxlen=window)
        self._fps_samples: Deque[float] = deque(maxlen=window)
        self._records = 0
        self.log_path = Path(log_path) if log_path else None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, metrics: FrameMetrics) -> None:
        """Store one frame of metrics and optionally append to the JSONL log."""
        self._latencies.append(metrics.latency_ms)
        self._fps_samples.append(metrics.fps)
        self._records += 1

        if self.log_path is not None:
            payload = asdict(metrics)
            payload["t"] = time.time()
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")

    def summary(self) -> Dict[str, float]:
        """Return rolling averages useful for HUD / reports."""
        def mean(values: Deque[float]) -> float:
            return float(sum(values) / len(values)) if values else 0.0

        return {
            "frames": float(self._records),
            "fps_avg": mean(self._fps_samples),
            "latency_ms_avg": mean(self._latencies),
            "latency_ms_p95": float(sorted(self._latencies)[int(0.95 * (len(self._latencies) - 1))])
            if self._latencies
            else 0.0,
        }

    def meets_targets(self, max_latency_ms: float = 200.0, min_fps: float = 15.0) -> bool:
        """
        Check specification targets from the project requirements.

        Default targets: identification / response under 200 ms.
        """
        summary = self.summary()
        return summary["latency_ms_avg"] <= max_latency_ms and summary["fps_avg"] >= min_fps
