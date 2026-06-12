import os
import tempfile
import pytest
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Temporary directory that cleans up after test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_bbox():
    from shared.models import BBox
    return BBox(100, 200, 300, 400)


@pytest.fixture
def sample_detection():
    from shared.models import DetectionResult, DetectedObject, BBox
    return DetectionResult(
        camera_id="cam-01",
        objects=[
            DetectedObject(bbox=BBox(100, 100, 200, 300), cls="person", conf=0.9),
            DetectedObject(bbox=BBox(300, 100, 400, 300), cls="person", conf=0.85),
            DetectedObject(bbox=BBox(110, 110, 150, 160), cls="helmet", conf=0.7),
        ],
        keypoints=None,
    )


@pytest.fixture
def sample_violation():
    from shared.models import Violation
    return Violation(
        camera_id="cam-01",
        type="NO_HELMET",
        severity="HIGH",
        bbox=[100, 100, 200, 300],
    )
