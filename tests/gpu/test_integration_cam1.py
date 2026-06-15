"""Integration tests for CAM1 pipeline."""
import numpy as np
import pytest
from gpu.roi_checker import ROIChecker
from gpu.overlay import draw_person_bboxes, draw_roi_polygons
from shared.models import BBox, DetectedObject


def test_roi_checker_with_sample_person():
    rois = [
        {"zone_name": "Test Zone", "color": "#ff0000", "enabled": True,
         "points_json": "[[0,0],[200,0],[200,200],[0,200]]"},
    ]
    checker = ROIChecker(rois)

    person = DetectedObject(bbox=BBox(50, 50, 150, 180), cls="person", conf=0.85)
    foot = (person.bbox.x1 + person.bbox.width / 2, person.bbox.y2)
    zones = checker.check_person(foot)
    assert len(zones) == 1

    person2 = DetectedObject(bbox=BBox(300, 300, 400, 400), cls="person", conf=0.85)
    foot2 = (person2.bbox.x1 + person2.bbox.width / 2, person2.bbox.y2)
    zones2 = checker.check_person(foot2)
    assert len(zones2) == 0


def test_overlay_with_sample_persons():
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
    persons = [
        DetectedObject(bbox=BBox(50, 50, 150, 180), cls="person", conf=0.85),
        DetectedObject(bbox=BBox(300, 50, 400, 200), cls="person", conf=0.72),
    ]
    result = draw_person_bboxes(frame, persons)
    assert result.shape == frame.shape

    rois = [{"zone_name": "Zone A", "points_json": "[[0,0],[200,0],[200,200],[0,200]]"}]
    result2 = draw_roi_polygons(frame, rois)
    assert result2.shape == frame.shape
