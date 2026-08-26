"""Unit tests for face tracking helpers (no camera / models)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.tracking import FaceTracker, association_score, blend_bbox, bbox_center, iou
from common.types import FaceSample


class GeometryTests(unittest.TestCase):
    def test_iou_identical_and_disjoint(self) -> None:
        box = (0, 0, 100, 100)
        self.assertAlmostEqual(iou(box, box), 1.0)
        self.assertEqual(iou(box, (200, 200, 300, 300)), 0.0)

    def test_iou_partial_overlap(self) -> None:
        a = (0, 0, 100, 100)
        b = (50, 50, 150, 150)
        self.assertGreater(iou(a, b), 0.1)
        self.assertLess(iou(a, b), 0.5)

    def test_bbox_center_and_blend(self) -> None:
        self.assertEqual(bbox_center((0, 0, 10, 20)), (5.0, 10.0))
        blended = blend_bbox((0, 0, 0, 0), (10, 10, 10, 10), alpha=1.0)
        self.assertEqual(blended, (10, 10, 10, 10))
        mid = blend_bbox((0, 0, 0, 0), (10, 10, 10, 10), alpha=0.5)
        self.assertEqual(mid, (5, 5, 5, 5))

    def test_association_score_prefers_overlap(self) -> None:
        a = (0, 0, 100, 100)
        close = (10, 10, 110, 110)
        far = (400, 400, 500, 500)
        self.assertGreater(association_score(a, close), association_score(a, far))


class FaceTrackerTests(unittest.TestCase):
    def test_creates_and_keeps_track(self) -> None:
        tracker = FaceTracker(confirm_hits=2, max_miss=2)
        det = FaceSample(bbox=(10, 10, 80, 80), score=0.9)
        tracks = tracker.update([det])
        self.assertEqual(len(tracks), 1)
        tid = tracks[0].track_id
        tracks2 = tracker.update([FaceSample(bbox=(12, 12, 82, 82), score=0.91)])
        self.assertEqual(len(tracks2), 1)
        self.assertEqual(tracks2[0].track_id, tid)

    def test_drops_stale_tracks(self) -> None:
        tracker = FaceTracker(max_miss=1)
        tracker.update([FaceSample(bbox=(0, 0, 50, 50), score=0.9)])
        # No detections for two updates → miss exceeds max_miss
        tracker.update([])
        tracks = tracker.update([])
        self.assertEqual(len(tracker.tracks), 0)
        self.assertEqual(tracks, [])

    def test_identity_requires_confirmation(self) -> None:
        tracker = FaceTracker(confirm_hits=2, min_score=0.4)
        tracker.update([FaceSample(bbox=(0, 0, 100, 100), score=0.9)])
        vote = FaceSample(bbox=(0, 0, 100, 100), score=0.9, identity="Alice", identity_score=0.8)
        tracker.apply_identities([vote])
        self.assertIsNone(list(tracker.tracks.values())[0].identity)
        tracker.apply_identities([vote])
        self.assertEqual(list(tracker.tracks.values())[0].identity, "Alice")

    def test_unknown_clears_confirmed_after_negatives(self) -> None:
        tracker = FaceTracker(confirm_hits=2, min_score=0.4)
        tracker.update([FaceSample(bbox=(0, 0, 100, 100), score=0.9)])
        vote = FaceSample(bbox=(0, 0, 100, 100), score=0.9, identity="Alice", identity_score=0.8)
        tracker.apply_identities([vote])
        tracker.apply_identities([vote])
        self.assertEqual(list(tracker.tracks.values())[0].identity, "Alice")
        unknown = FaceSample(bbox=(0, 0, 100, 100), score=0.9, identity=None, identity_score=0.1)
        tracker.apply_identities([unknown])
        tracker.apply_identities([unknown])
        self.assertIsNone(list(tracker.tracks.values())[0].identity)


if __name__ == "__main__":
    unittest.main()
