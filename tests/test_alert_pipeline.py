"""Integration test for the full alert pipeline: ROI → Classify → Cooldown → Dispatch."""
import json
from pathlib import Path
from unittest.mock import MagicMock

from shared.models import DetectionResult, DetectedObject, BBox, Keypoint


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
    """End-to-end alert pipeline tests with mocked DB and WebSocket."""

    def test_full_pipeline_roi_blocked(self):
        """Violation outside ROI should be blocked before classification."""
        from alert.roi_matcher import ROIMatcher
        from alert.classifier import ViolationClassifier
        from alert.cooldown import CooldownManager
        from alert.dispatcher import Dispatcher

        mock_db = MagicMock()
        # ROI is a small polygon far from the person
        mock_db.get_roi.return_value = {
            "camera_id": "cam-01",
            "polygon": json.dumps([(500, 500), (600, 500), (600, 600), (500, 600)]),
        }
        mock_db.insert_violation.return_value = 1

        roi = ROIMatcher(mock_db)
        classifier = ViolationClassifier()
        cooldown = CooldownManager()
        dispatcher = Dispatcher(db=mock_db, ws_manager=None)

        # Person at (100-200, 100-300) — center (150, 200)
        # ROI is at (500-600, 500-600) — person is outside
        result = make_detection(has_helmet=False, has_vest=False)

        violations_dispatched = 0
        for obj in result.objects:
            if obj.cls != "person":
                continue
            bbox_list = [obj.bbox.x1, obj.bbox.y1, obj.bbox.x2, obj.bbox.y2]
            if not roi.is_in_roi(result.camera_id, bbox_list):
                continue  # Blocked by ROI

            # This code path should NOT be reached
            violations = classifier.classify(result)
            for v in violations:
                if cooldown.should_alert(v.camera_id, v.type):
                    dispatcher.dispatch(v, frame_bgr=None)
                    violations_dispatched += 1

        assert violations_dispatched == 0

    def test_full_pipeline_cooldown_blocks_duplicate(self):
        """Second identical violation within 5s should be blocked by cooldown."""
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

        roi = ROIMatcher(mock_db)
        classifier = ViolationClassifier()
        cooldown = CooldownManager(cooldown_seconds=5)
        dispatcher = Dispatcher(db=mock_db, ws_manager=None)

        # Person without helmet
        result = make_detection(has_helmet=False, has_vest=True)

        def run_pipeline():
            count = 0
            for obj in result.objects:
                if obj.cls != "person":
                    continue
                bbox_list = [obj.bbox.x1, obj.bbox.y1, obj.bbox.x2, obj.bbox.y2]
                if not roi.is_in_roi(result.camera_id, bbox_list):
                    continue
                violations = classifier.classify(result)
                for v in violations:
                    if cooldown.should_alert(v.camera_id, v.type):
                        dispatcher.dispatch(v, frame_bgr=None)
                        count += 1
            return count

        # First pass: should dispatch NO_HELMET and NO_BOOT (no boot detection in result)
        assert run_pipeline() == 2
        # Second pass: cooldown blocks both
        assert run_pipeline() == 0

    def test_full_pipeline_dispatches_valid_violation(self):
        """A valid violation inside ROI with no cooldown should dispatch."""
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

        roi = ROIMatcher(mock_db)
        classifier = ViolationClassifier()
        cooldown = CooldownManager(cooldown_seconds=0)  # No cooldown
        dispatcher = Dispatcher(db=mock_db, ws_manager=None)

        result = make_detection(has_helmet=False, has_vest=False)

        violations_dispatched = 0
        for obj in result.objects:
            if obj.cls != "person":
                continue
            bbox_list = [obj.bbox.x1, obj.bbox.y1, obj.bbox.x2, obj.bbox.y2]
            if not roi.is_in_roi(result.camera_id, bbox_list):
                continue
            violations = classifier.classify(result)
            for v in violations:
                if cooldown.should_alert(v.camera_id, v.type):
                    dispatcher.dispatch(v, frame_bgr=None)
                    violations_dispatched += 1

        # Should dispatch NO_HELMET and NO_VEST (and maybe NO_BOOT)
        assert violations_dispatched >= 2
        assert mock_db.insert_violation.call_count >= 2
