"""Test PPE checker crop logic."""
import numpy as np
import pytest
from gpu.ppe_checker import crop_head, crop_torso, crop_feet
from shared.models import BBox


def test_crop_head_valid():
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
    bbox = BBox(100, 100, 300, 400)
    head = crop_head(frame, bbox)
    assert head.shape[0] > 0
    assert head.shape[1] > 0
    assert head.shape[2] == 3


def test_crop_head_clamps_to_frame():
    frame = np.ones((480, 640, 3), dtype=np.uint8)
    bbox = BBox(0, 0, 50, 100)
    head = crop_head(frame, bbox)
    assert head.shape[0] > 0
    assert head.shape[1] > 0


def test_crop_feet_at_bottom():
    frame = np.ones((480, 640, 3), dtype=np.uint8)
    bbox = BBox(100, 100, 300, 400)
    feet = crop_feet(frame, bbox)
    assert 20 <= feet.shape[0] <= 60


def test_crop_all_valid_shapes():
    frame = np.ones((480, 640, 3), dtype=np.uint8)
    bbox = BBox(50, 50, 300, 400)
    assert crop_head(frame, bbox).size > 0
    assert crop_torso(frame, bbox).size > 0
    assert crop_feet(frame, bbox).size > 0
