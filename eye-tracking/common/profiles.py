"""
Runtime profiles tuned for CPU-friendly ID latency + fewer false matches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class RuntimeProfile:
    """Tunable knobs applied by the live recognition engine."""

    name: str
    width: int
    height: int
    det_size: int
    infer_size: int
    every: int
    model_name: str
    min_face_px: int
    min_det_score: float
    match_threshold: float
    match_margin: float
    hold_frames: int
    confirm_hits: int


PROFILES: Dict[str, RuntimeProfile] = {
    "fast": RuntimeProfile(
        name="fast",
        width=640,
        height=480,
        det_size=256,
        infer_size=416,
        every=2,
        model_name="buffalo_l",
        min_face_px=45,
        min_det_score=0.50,
        match_threshold=0.42,
        match_margin=0.07,
        hold_frames=12,
        confirm_hits=2,
    ),
    # Target: ID under ~300–800ms on CPU laptop (not multi-second).
    "balanced": RuntimeProfile(
        name="balanced",
        width=640,
        height=480,
        det_size=320,
        infer_size=480,
        every=2,
        model_name="buffalo_l",
        min_face_px=50,
        min_det_score=0.50,
        match_threshold=0.42,
        match_margin=0.08,
        hold_frames=14,
        confirm_hits=2,
    ),
    "accurate": RuntimeProfile(
        name="accurate",
        width=800,
        height=600,
        det_size=480,
        infer_size=640,
        every=1,
        model_name="buffalo_l",
        min_face_px=55,
        min_det_score=0.55,
        match_threshold=0.45,
        match_margin=0.09,
        hold_frames=18,
        confirm_hits=2,
    ),
}


def get_profile(name: str) -> RuntimeProfile:
    """Return a named profile or raise ``KeyError`` with available options."""
    key = name.strip().lower()
    if key not in PROFILES:
        raise KeyError(f"Unknown profile '{name}'. Choose from: {', '.join(PROFILES)}")
    return PROFILES[key]
