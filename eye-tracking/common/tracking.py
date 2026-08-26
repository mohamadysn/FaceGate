"""
Temporal identity confirmation and responsive face tracking.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from .types import BBox, FaceSample


def iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union between two axis-aligned boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def bbox_center(b: BBox) -> Tuple[float, float]:
    """Return the center of a bbox."""
    return 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])


def blend_bbox(old: BBox, new: BBox, alpha: float = 0.85) -> BBox:
    """Exponentially blend toward ``new`` for responsive boxes."""
    a = float(max(0.0, min(1.0, alpha)))
    return (
        int(round(a * new[0] + (1.0 - a) * old[0])),
        int(round(a * new[1] + (1.0 - a) * old[1])),
        int(round(a * new[2] + (1.0 - a) * old[2])),
        int(round(a * new[3] + (1.0 - a) * old[3])),
    )


def association_score(track_bbox: BBox, det_bbox: BBox) -> float:
    """Combine IoU with center proximity for fast camera motion."""
    overlap = iou(track_bbox, det_bbox)
    tcx, tcy = bbox_center(track_bbox)
    dcx, dcy = bbox_center(det_bbox)
    tw = max(1.0, track_bbox[2] - track_bbox[0])
    th = max(1.0, track_bbox[3] - track_bbox[1])
    dist = ((tcx - dcx) ** 2 + (tcy - dcy) ** 2) ** 0.5
    norm = ((tw ** 2 + th ** 2) ** 0.5) + 1e-6
    proximity = max(0.0, 1.0 - dist / (1.8 * norm))
    return 0.55 * overlap + 0.45 * proximity


@dataclass
class TrackedFace:
    """
    One tracked face.

    ``identity`` is only set after ``confirm_hits`` consecutive agreeing votes,
    which blocks one-shot false recognitions.
    """

    track_id: int
    bbox: BBox
    score: float
    identity: Optional[str] = None
    identity_score: float = 0.0
    miss: int = 0
    pending_name: Optional[str] = None
    pending_hits: int = 0
    negative_hits: int = 0
    needs_id: bool = True  # True until a confirmed identity exists


class FaceTracker:
    """Responsive boxes + confirmed identities (anti false-positive)."""

    def __init__(
        self,
        iou_threshold: float = 0.15,
        max_miss: int = 8,
        bbox_alpha: float = 0.9,
        confirm_hits: int = 2,
        min_score: float = 0.45,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_miss = max_miss
        self.bbox_alpha = bbox_alpha
        self.confirm_hits = max(1, int(confirm_hits))
        self.min_score = float(min_score)
        self._next_id = 1
        self.tracks: Dict[int, TrackedFace] = {}

    def update(self, detections: List[FaceSample]) -> List[TrackedFace]:
        """Refresh boxes from detections. Does not invent identities."""
        track_ids = list(self.tracks.keys())
        assigned_tracks = set()
        assigned_dets = set()
        pairs: List[Tuple[float, int, int]] = []

        for ti, tid in enumerate(track_ids):
            for di, det in enumerate(detections):
                pairs.append((association_score(self.tracks[tid].bbox, det.bbox), ti, di))
        pairs.sort(reverse=True)

        for score, ti, di in pairs:
            tid = track_ids[ti]
            if score < self.iou_threshold or tid in assigned_tracks or di in assigned_dets:
                continue
            det = detections[di]
            track = self.tracks[tid]
            track.bbox = blend_bbox(track.bbox, det.bbox, alpha=self.bbox_alpha)
            track.score = det.score
            track.miss = 0
            assigned_tracks.add(tid)
            assigned_dets.add(di)

        for di, det in enumerate(detections):
            if di in assigned_dets:
                continue
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = TrackedFace(
                track_id=tid,
                bbox=det.bbox,
                score=det.score,
                needs_id=True,
            )

        for tid in track_ids:
            if tid in assigned_tracks:
                continue
            self.tracks[tid].miss += 1

        stale = [tid for tid, tr in self.tracks.items() if tr.miss > self.max_miss]
        for tid in stale:
            del self.tracks[tid]

        return [tr for tr in self.tracks.values() if tr.miss == 0]

    def apply_identities(self, faces: List[FaceSample]) -> None:
        """
        Apply recognition votes with confirmation.

        - Unknown / low score increments ``negative_hits`` and can clear ID.
        - Same name with good score increments ``pending_hits``.
        - Name is published only after ``confirm_hits`` agreements.
        """
        for face in faces:
            best_tid = None
            best_score = -1.0
            for tid, tr in self.tracks.items():
                score = association_score(tr.bbox, face.bbox)
                if score > best_score:
                    best_score = score
                    best_tid = tid
            if best_tid is None or best_score < 0.12:
                continue

            tr = self.tracks[best_tid]
            tr.bbox = blend_bbox(tr.bbox, face.bbox, alpha=0.30)
            tr.identity_score = float(face.identity_score)

            name = face.identity
            score = float(face.identity_score)
            accepted = name is not None and score >= self.min_score

            if not accepted:
                tr.negative_hits += 1
                tr.pending_name = None
                tr.pending_hits = 0
                # Drop a confirmed ID only after repeated weak/unknown reads.
                if tr.identity is not None and tr.negative_hits >= self.confirm_hits:
                    tr.identity = None
                    tr.needs_id = True
                    tr.negative_hits = 0
                continue

            tr.negative_hits = 0
            if tr.pending_name == name:
                tr.pending_hits += 1
            else:
                tr.pending_name = name
                tr.pending_hits = 1

            if tr.identity == name:
                # Reinforce existing ID quickly.
                tr.pending_hits = self.confirm_hits
                tr.needs_id = False
                continue

            if tr.pending_hits >= self.confirm_hits:
                # If switching from another confirmed person, demand a fresh confirm.
                if tr.identity is not None and tr.identity != name:
                    if tr.pending_hits < self.confirm_hits + 1:
                        continue
                tr.identity = name
                tr.needs_id = False
