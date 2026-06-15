"""Test overlay drawing functions output valid frames."""
import numpy as np
import pytest
from gpu.overlay import (
    draw_person_bboxes, draw_roi_polygons, draw_ppe_labels,
    draw_disconnected, draw_detection_offline,
)
from shared.models import BBox, DetectedObject


@pytest.fixture
def frame():
    return np.ones((480, 640, 3), dtype=np.uint8) * 200


def test_draw_person_bboxes_output_shape(frame):
    persons = [DetectedObject(bbox=BBox(50, 50, 200, 400), cls="person", conf=0.85)]
    result = draw_person_bboxes(frame, persons)
    assert result.shape == frame.shape
    assert result.dtype == frame.dtype


def test_draw_person_bboxes_empty(frame):
    result = draw_person_bboxes(frame, [])
    assert np.array_equal(result, frame)


def test_draw_disconnected(frame):
    result = draw_disconnected(frame)
    assert result.shape == frame.shape


def test_draw_detection_offline(frame):
    result = draw_detection_offline(frame)
    assert result.shape == frame.shape


def test_draw_roi_polygons(frame):
    rois = [{"zone_name": "Zone A", "points_json": "[[0,0],[100,0],[100,100],[0,100]]"}]
    result = draw_roi_polygons(frame, rois)
    assert result.shape == frame.shape


def test_draw_ppe_labels(frame):
    persons = [DetectedObject(bbox=BBox(50, 50, 200, 400), cls="person", conf=0.85)]
    alerts = [{"person_idx": 0, "violations": ["NO_VEST"]}]
    result = draw_ppe_labels(frame, persons, alerts)
    assert result.shape == frame.shape
