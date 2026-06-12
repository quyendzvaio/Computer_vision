"""Integration test for AlertPipeline: ROI → Classify → Cooldown → Dispatch."""
import json
from unittest.mock import MagicMock

from shared.models import DetectionResult, DetectedObject, BBox


def make_detection(camera_id="cam-01", has_helmet=True, has_vest=True):
    """Helper to build a DetectionResult with configurable equipment."""
    objects = [
        DetectedObject(bbox=BBox(100, 100, 200, 300), cls="person", conf=0.9),
    ]
    if has_helmet:
        objects.append(
            DetectedObject(bbox=BBox(110, 100, 180, 150), cls="helmet", conf=0.8)
        )
    if has_vest:
        objects.append(
            DetectedObject(bbox=BBox(110, 150, 190, 260), cls="vest", conf=0.8)
        )
    return DetectionResult(camera_id=camera_id, objects=objects)


class TestAlertPipeline:
    """End-to-end alert pipeline tests via AlertPipeline.process()."""

    def test_pipeline_roi_blocks(self):
        """Violation outside ROI should be blocked."""
        from alert import AlertPipeline
        from alert.roi_matcher import ROIMatcher
        from alert.classifier import ViolationClassifier
        from alert.cooldown import CooldownManager
        from alert.dispatcher import Dispatcher

        mock_db = MagicMock()
        # ROI is a small polygon far from the person (center ~150,200)
        mock_db.get_roi.return_value = {
            "camera_id": "cam-01",
            "polygon": json.dumps([(500, 500), (600, 500), (600, 600), (500, 600)]),
        }
        mock_db.insert_violation.return_value = 1

        pipeline = AlertPipeline(
            roi_matcher=ROIMatcher(mock_db),
            classifier=ViolationClassifier(),
            cooldown=CooldownManager(cooldown_seconds=0),
            dispatcher=Dispatcher(db=mock_db, ws_manager=None),
        )

        # Person at center (150, 200) — outside ROI at (500-600, 500-600)
        result = make_detection(has_helmet=False, has_vest=False)
        dispatched = pipeline.process(result, frame_bgr=None)

        # All violations blocked by ROI → 0 dispatched
        assert len(dispatched) == 0
        mock_db.insert_violation.assert_not_called()

    def test_pipeline_cooldown_blocks_duplicate(self):
        """Second call with same violation type within cooldown should be blocked."""
        from alert import AlertPipeline
        from alert.roi_matcher import ROIMatcher
        from alert.classifier import ViolationClassifier
        from alert.cooldown import CooldownManager
        from alert.dispatcher import Dispatcher

        mock_db = MagicMock()
        # ROI covers the entire frame
        mock_db.get_roi.return_value = {
            "camera_id": "cam-01",
            "polygon": json.dumps([(0, 0), (640, 0), (640, 480), (0, 480)]),
        }
        mock_db.insert_violation.return_value = 1

        pipeline = AlertPipeline(
            roi_matcher=ROIMatcher(mock_db),
            classifier=ViolationClassifier(),
            cooldown=CooldownManager(cooldown_seconds=5),
            dispatcher=Dispatcher(db=mock_db, ws_manager=None),
        )

        # Person without helmet (but with vest, no boot detection)
        # → NO_HELMET + NO_BOOT violations
        result = make_detection(has_helmet=False, has_vest=True)

        # First pass: violations should dispatch
        dispatched1 = pipeline.process(result, frame_bgr=None)
        assert len(dispatched1) >= 1
        first_call_count = mock_db.insert_violation.call_count

        # Second pass: cooldown blocks all same types
        dispatched2 = pipeline.process(result, frame_bgr=None)
        assert len(dispatched2) == 0
        # No additional DB inserts
        assert mock_db.insert_violation.call_count == first_call_count

    def test_pipeline_dispatches_valid_violation(self):
        """A valid violation inside ROI with no cooldown should dispatch."""
        from alert import AlertPipeline
        from alert.roi_matcher import ROIMatcher
        from alert.classifier import ViolationClassifier
        from alert.cooldown import CooldownManager
        from alert.dispatcher import Dispatcher

        mock_db = MagicMock()
        mock_db.get_roi.return_value = {
            "camera_id": "cam-01",
            "polygon": json.dumps([(0, 0), (640, 0), (640, 480), (0, 480)]),
        }
        mock_db.insert_violation.return_value = 1

        pipeline = AlertPipeline(
            roi_matcher=ROIMatcher(mock_db),
            classifier=ViolationClassifier(),
            cooldown=CooldownManager(cooldown_seconds=0),  # No cooldown
            dispatcher=Dispatcher(db=mock_db, ws_manager=None),
        )

        # Person without helmet and vest → NO_HELMET + NO_VEST + NO_BOOT
        result = make_detection(has_helmet=False, has_vest=False)
        dispatched = pipeline.process(result, frame_bgr=None)

        assert len(dispatched) >= 2  # NO_HELMET + NO_VEST at minimum
        violation_types = {v.type for v in dispatched}
        assert "NO_HELMET" in violation_types
        assert "NO_VEST" in violation_types
        assert mock_db.insert_violation.call_count == len(dispatched)

        # Verify dispatched violations have expected structure
        for v in dispatched:
            assert v.camera_id == "cam-01"
            assert v.id == 1  # From mock return value
            assert v.thumbnail_path != ""
