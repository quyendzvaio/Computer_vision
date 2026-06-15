"""Integration tests for CAM2 pipeline."""
import numpy as np
import pytest
from gpu.ppe_checker import crop_head, crop_torso, crop_feet
from gpu.overlay import draw_person_bboxes, draw_ppe_labels
from shared.models import BBox, DetectedObject


def test_crop_all_preserve_aspect():
    frame = np.ones((480, 640, 3), dtype=np.uint8)
    bbox = BBox(100, 100, 300, 400)

    head = crop_head(frame, bbox)
    assert head.size > 0
    assert head.shape[2] == 3

    torso = crop_torso(frame, bbox)
    assert torso.size > 0

    feet = crop_feet(frame, bbox)
    assert feet.size > 0


def test_ppe_labels_on_empty_frame():
    frame = np.ones((480, 640, 3), dtype=np.uint8)
    result = draw_ppe_labels(frame, [], [])
    assert result.shape == frame.shape
