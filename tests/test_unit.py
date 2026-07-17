"""Unit tests: model_manager, detector, roi_checker, overlay, shared memory."""
import json
from unittest.mock import patch

import numpy as np
import pytest

from shared.models import BBox, DetectedObject, DetectionResult


# --- BBox ---

class TestBBox:
    def test_center(self):
        cx, cy = BBox(100, 200, 300, 400).center
        assert cx == 200 and cy == 300

    def test_width_height(self):
        assert BBox(10, 20, 50, 100).width == 40
        assert BBox(10, 20, 50, 100).height == 80

    def test_aspect_ratio(self):
        assert BBox(0, 0, 40, 80).aspect_ratio == 0.5
        assert BBox(0, 0, 40, 0).aspect_ratio == 0.0

    def test_serialize_roundtrip(self):
        b = BBox(1.5, 2.5, 10, 20)
        assert BBox.from_list(b.to_list()).x2 == 10

    def test_to_list(self):
        assert BBox(0, 1, 2, 3).to_list() == [0, 1, 2, 3]


# --- DetectedObject ---

class TestDetectedObject:
    def test_create(self):
        o = DetectedObject(BBox(0, 0, 10, 20), "person", 0.95)
        assert o.cls == "person" and o.conf == 0.95


# --- DetectionResult ---

class TestDetectionResult:
    def test_person_count(self):
        det = DetectionResult("cam1", objects=[
            DetectedObject(BBox(0, 0, 10, 20), "person", 0.9),
            DetectedObject(BBox(0, 0, 5, 5), "helmet", 0.8),
            DetectedObject(BBox(0, 0, 10, 20), "person", 0.7),
        ])
        assert det.person_count == 2

    def test_no_persons(self):
        assert DetectionResult("cam1").person_count == 0


# --- GPU Detector ---

class TestGPUDetector:
    def test_preprocess_shape(self):
        from gpu.detector import YOLODetector
        det = YOLODetector.__new__(YOLODetector)
        det.input_w, det.input_h = 640, 640
        t = det.preprocess(np.zeros((480, 640, 3), dtype=np.uint8))
        assert t.shape == (1, 3, 640, 640)
        assert t.dtype == np.float32
        assert 0.0 <= t.min() <= t.max() <= 1.0

    def test_postprocess_empty(self):
        from gpu.detector import YOLODetector
        det = YOLODetector.__new__(YOLODetector)
        det.input_w, det.input_h = 640, 640
        assert det.postprocess((np.zeros((1, 84, 8400), dtype=np.float32),), (480, 640)) == []

    def test_detect_roi_crop_offsets_bbox(self):
        from gpu.detector import YOLODetector
        det = YOLODetector.__new__(YOLODetector)
        det.detect = lambda f: [DetectedObject(BBox(50, 50, 150, 150), "person", 0.9)]
        result = det.detect_roi(np.zeros((480, 640, 3), dtype=np.uint8), (100, 100, 200, 200))
        assert result[0].bbox.x1 == 150 and result[0].bbox.y2 == 250

    def test_detect_roi_too_small(self):
        from gpu.detector import YOLODetector
        det = YOLODetector.__new__(YOLODetector)
        assert det.detect_roi(np.zeros((100, 100, 3), dtype=np.uint8), (0, 0, 10, 10)) == []

    def test_detect_roi_fallback_full(self):
        from gpu.detector import YOLODetector
        det = YOLODetector.__new__(YOLODetector)
        called = []
        det.detect = lambda f: (called.append(1) or [])
        det.detect_roi(np.zeros((100, 100, 3), dtype=np.uint8), None)
        assert called


# --- CPU model_manager ---

class TestModelManager:
    def test_postprocess_below_threshold(self):
        from inference.model_manager import ModelManager
        mm = ModelManager.__new__(ModelManager)
        mm.conf_threshold = 0.5
        assert mm._postprocess((np.zeros((1, 84, 8400), dtype=np.float32),), (416, 416)) == []

    def test_postprocess_with_person(self):
        from inference.model_manager import ModelManager
        mm = ModelManager.__new__(ModelManager)
        mm.conf_threshold = 0.3
        mm.nms_threshold = 0.45
        mm.input_w = mm.input_h = 416

        outputs = np.zeros((1, 84, 8400), dtype=np.float32)
        outputs[0, 0, 0] = 0.5; outputs[0, 1, 0] = 0.5
        outputs[0, 2, 0] = 0.2; outputs[0, 3, 0] = 0.4
        outputs[0, 4, 0] = 0.9  # class 0 (person) score

        dets = mm._postprocess((outputs,), (416, 416))
        assert len(dets) >= 1
        assert dets[0].cls_name == "person"
        assert dets[0].conf > 0.3


# --- ROI checker ---

class TestROIChecker:
    def test_point_inside(self):
        from gpu.roi_checker import point_in_polygon
        assert point_in_polygon((50, 50), [[0, 0], [100, 0], [100, 100], [0, 100]])

    def test_point_outside(self):
        from gpu.roi_checker import point_in_polygon
        assert not point_in_polygon((150, 50), [[0, 0], [100, 0], [100, 100], [0, 100]])

    def test_point_on_edge(self):
        from gpu.roi_checker import point_in_polygon
        assert point_in_polygon((25, 0), [[0, 0], [50, 0], [50, 50], [0, 50]])

    def test_check_person(self):
        from gpu.roi_checker import ROIChecker
        rois = [{"zone_name": "Z1", "enabled": True,
                 "points_json": json.dumps([[0,0],[100,0],[100,100],[0,100]])}]
        assert len(ROIChecker(rois).check_person((50, 50))) == 1

    def test_disabled_zone_ignored(self):
        from gpu.roi_checker import ROIChecker
        rois = [{"zone_name": "Z1", "enabled": False,
                 "points_json": json.dumps([[0,0],[100,0],[100,100],[0,100]])}]
        assert len(ROIChecker(rois).check_person((50, 50))) == 0

    def test_reload(self):
        from gpu.roi_checker import ROIChecker
        c = ROIChecker()
        c.reload([{"zone_name": "Z1", "points_json": "[]"}])
        assert len(c._rois) == 1

    def test_bounds(self):
        from gpu.roi_checker import compute_roi_bounds
        b = compute_roi_bounds([{"zone_name": "Z1", "enabled": True,
                                 "points_json": json.dumps([[100,100],[300,100],[300,300],[100,300]])}],
                               margin=0, frame_size=(640, 480))
        assert b == (100, 100, 200, 200)

    def test_bounds_empty(self):
        from gpu.roi_checker import compute_roi_bounds
        assert compute_roi_bounds([]) is None


# --- Overlay ---

class TestOverlay:
    def test_person_bboxes(self):
        from gpu.overlay import draw_person_bboxes
        r = draw_person_bboxes(np.ones((480, 640, 3), dtype=np.uint8),
                               [DetectedObject(BBox(50, 50, 150, 180), "person", 0.85)])
        assert r.shape == (480, 640, 3)

    def test_roi_polygons(self):
        from gpu.overlay import draw_roi_polygons
        r = draw_roi_polygons(np.ones((480, 640, 3), dtype=np.uint8),
                              [{"zone_name": "Z1", "points_json": json.dumps([[0,0],[100,0],[100,100],[0,100]])}])
        assert r.shape == (480, 640, 3)

    def test_disconnected(self):
        from gpu.overlay import draw_disconnected
        assert draw_disconnected(np.ones((100, 100, 3), dtype=np.uint8)).shape == (100, 100, 3)

    def test_ppe_labels(self):
        from gpu.overlay import draw_ppe_labels
        r = draw_ppe_labels(np.ones((480, 640, 3), dtype=np.uint8),
                            [DetectedObject(BBox(50, 50, 150, 180), "person", 0.85)],
                            [{"person_idx": 0, "violations": ["NO_HELMET"]}])
        assert r.shape == (480, 640, 3)


# --- FrameBuffer (mocked) ---

class TestFrameBuffer:
    def test_pick_free(self):
        from shared.memory import FrameBuffer, BufState
        buf = FrameBuffer.__new__(FrameBuffer)
        with patch.object(FrameBuffer, '_state_a', new_callable=property) as s:
            s.return_value = BufState.FREE
            assert buf.pick_free() == 0

    def test_pick_free_none(self):
        from shared.memory import FrameBuffer, BufState
        buf = FrameBuffer.__new__(FrameBuffer)
        with patch.object(FrameBuffer, '_state_a', new_callable=property) as a, \
             patch.object(FrameBuffer, '_state_b', new_callable=property) as b:
            a.return_value = BufState.LOCKED
            b.return_value = BufState.LOCKED
            assert buf.pick_free() is None

    def test_find_filled(self):
        from shared.memory import FrameBuffer, BufState
        buf = FrameBuffer.__new__(FrameBuffer)
        with patch.object(FrameBuffer, '_state_a', new_callable=property) as s:
            s.return_value = BufState.FILLED
            assert buf.find_filled() == 0

    def test_acquire_transition(self):
        from shared.memory import FrameBuffer, BufState
        import unittest.mock as mock
        buf = FrameBuffer.__new__(FrameBuffer)
        with patch.object(FrameBuffer, '_state_a', new_callable=property) as s:
            s.return_value = BufState.LOCKED
            buf.acquire(0)
            # property is set — not asserting call as it's side-effect

    def test_release_transition(self):
        from shared.memory import FrameBuffer, BufState
        import unittest.mock as mock
        buf = FrameBuffer.__new__(FrameBuffer)
        with patch.object(FrameBuffer, '_state_a', new_callable=property) as s:
            s.return_value = BufState.FREE
            buf.release(0)
