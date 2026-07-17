"""Follow-system integration: edge(ZMQ) -> detect -> alert pipeline.

Tests full data flow with synthetic data and mocked DB.
No hardware or real model needed.
"""
import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from shared.models import BBox, DetectedObject, DetectionResult


# --- ROI + cooldown + dispatch integration ---

class TestAlertPipelineFlow:

    def test_person_in_roi_dispatches(self):
        from alert import AlertPipeline
        from alert.roi_matcher import ROIMatcher
        from alert.classifier import ViolationClassifier
        from alert.cooldown import CooldownManager
        from alert.dispatcher import Dispatcher

        mock_db = MagicMock()
        mock_db.get_roi.return_value = {
            "camera_id": "cam-01",
            "polygon": json.dumps([(0,0), (640,0), (640,480), (0,480)]),
        }
        mock_db.insert_violation.return_value = 1

        pipeline = AlertPipeline(
            roi_matcher=ROIMatcher(mock_db),
            classifier=ViolationClassifier(),
            cooldown=CooldownManager(cooldown_seconds=0),
            dispatcher=Dispatcher(db=mock_db, ws_manager=None),
        )

        det = DetectionResult("cam-01", objects=[
            DetectedObject(BBox(100,100,200,300), "person", 0.9),
        ])
        fired = pipeline.process(det, frame_bgr=np.zeros((480,640,3), dtype=np.uint8))
        assert len(fired) >= 2
        assert mock_db.insert_violation.call_count == len(fired)

    def test_cooldown_blocks_duplicate(self):
        from alert import AlertPipeline
        from alert.roi_matcher import ROIMatcher
        from alert.classifier import ViolationClassifier
        from alert.cooldown import CooldownManager
        from alert.dispatcher import Dispatcher

        mock_db = MagicMock()
        mock_db.get_roi.return_value = {
            "camera_id": "cam-01",
            "polygon": json.dumps([(0,0), (640,0), (640,480), (0,480)]),
        }
        mock_db.insert_violation.return_value = 1

        pipeline = AlertPipeline(
            roi_matcher=ROIMatcher(mock_db),
            classifier=ViolationClassifier(),
            cooldown=CooldownManager(cooldown_seconds=10),
            dispatcher=Dispatcher(db=mock_db, ws_manager=None),
        )

        det = DetectionResult("cam-01", objects=[
            DetectedObject(BBox(100,100,200,300), "person", 0.9),
        ])
        fired1 = pipeline.process(det, frame_bgr=np.zeros((480,640,3), dtype=np.uint8))
        assert len(fired1) >= 1
        c1 = mock_db.insert_violation.call_count
        fired2 = pipeline.process(det, frame_bgr=None)
        assert len(fired2) == 0
        assert mock_db.insert_violation.call_count == c1

    def test_person_with_ppe_no_violation(self):
        from alert import AlertPipeline
        from alert.roi_matcher import ROIMatcher
        from alert.classifier import ViolationClassifier
        from alert.cooldown import CooldownManager
        from alert.dispatcher import Dispatcher

        mock_db = MagicMock()
        mock_db.get_roi.return_value = {
            "camera_id": "cam-01",
            "polygon": json.dumps([(0,0), (640,0), (640,480), (0,480)]),
        }

        pipeline = AlertPipeline(
            roi_matcher=ROIMatcher(mock_db),
            classifier=ViolationClassifier(),
            cooldown=CooldownManager(),
            dispatcher=Dispatcher(db=mock_db, ws_manager=None),
        )

        det = DetectionResult("cam-01", objects=[
            DetectedObject(BBox(100,100,200,300), "person", 0.9),
            DetectedObject(BBox(105,100,180,160), "helmet", 0.8),
            DetectedObject(BBox(100,160,190,260), "vest", 0.8),
            DetectedObject(BBox(110,280,180,300), "boot", 0.7),
        ])
        assert len(pipeline.process(det, frame_bgr=None)) == 0

    def test_outside_roi_blocked(self):
        from alert import AlertPipeline
        from alert.roi_matcher import ROIMatcher
        from alert.classifier import ViolationClassifier
        from alert.cooldown import CooldownManager
        from alert.dispatcher import Dispatcher

        mock_db = MagicMock()
        mock_db.get_roi.return_value = {
            "camera_id": "cam-01",
            "polygon": json.dumps([(500,500), (600,500), (600,600), (500,600)]),
        }

        pipeline = AlertPipeline(
            roi_matcher=ROIMatcher(mock_db),
            classifier=ViolationClassifier(),
            cooldown=CooldownManager(cooldown_seconds=0),
            dispatcher=Dispatcher(db=mock_db, ws_manager=None),
        )

        det = DetectionResult("cam-01", objects=[
            DetectedObject(BBox(100,100,200,300), "person", 0.9),
        ])
        assert len(pipeline.process(det, frame_bgr=np.zeros((480,640,3), dtype=np.uint8))) == 0
        mock_db.insert_violation.assert_not_called()


# --- ROI crop logic ---

def test_roi_bounds_crop():
    from gpu.roi_checker import compute_roi_bounds
    rois = [{"zone_name": "Z1", "enabled": True,
             "points_json": json.dumps([[100,100],[300,100],[300,300],[100,300]])}]
    b = compute_roi_bounds(rois, margin=0, frame_size=(640,480))
    assert b == (100, 100, 200, 200)

def test_roi_bounds_empty():
    from gpu.roi_checker import compute_roi_bounds
    assert compute_roi_bounds([]) is None
    assert compute_roi_bounds([{"enabled": False, "points_json": "[]"}]) is None


# --- Config ---

def test_cpu_config():
    import yaml
    with open("edge/config.cpu.yaml") as f:
        cfg = yaml.safe_load(f)
    for c in cfg["cameras"]:
        assert c["resolution"] == [320, 240]
        assert c["fps"] == 3

def test_gpu_config():
    import yaml
    with open("edge/config.yaml") as f:
        cfg = yaml.safe_load(f)
    for c in cfg["cameras"]:
        assert c["resolution"] == [640, 480]
        assert c["fps"] == 5
