import tempfile
from pathlib import Path

import pytest

from shared.models import BBox, DetectedObject, DetectionResult, Violation


@pytest.fixture
def temp_dir():
    """Temporary directory that cleans up after test.

    Note: The directory is destroyed on fixture teardown. Tests should
    not store the path for use outside the fixture scope.
    """
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_bbox():
    return BBox(100, 200, 300, 400)


@pytest.fixture
def sample_detection():
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
    return Violation(
        camera_id="cam-01",
        type="NO_HELMET",
        severity="HIGH",
        bbox=BBox(100, 100, 200, 300),
    )
