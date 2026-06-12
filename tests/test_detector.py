"""Tests for the detector module — converts ModelManager detections to DetectionResult."""
import pytest
from unittest.mock import MagicMock
import numpy as np
from shared.models import DetectionResult, DetectedObject, BBox


def test_detector_converts_detections():
    """Detector.run() should convert raw detections to DetectionResult."""
    from inference.detector import Detector
    from inference.model_manager import Detection

    mock_mm = MagicMock()
    mock_mm.detect.return_value = [
        Detection(bbox=(100, 100, 200, 300), cls=0, cls_name='person', conf=0.9),
        Detection(bbox=(110, 100, 180, 160), cls=0, cls_name='helmet', conf=0.7),
    ]
    # Return a valid frame so the detector proceeds to call detect()
    mock_mm.preprocess_jpeg.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

    detector = Detector(mock_mm)

    result = detector.run(b"fake_jpeg_bytes", "cam-01")

    assert isinstance(result, DetectionResult)
    assert result.camera_id == "cam-01"
    assert len(result.objects) == 2

    person = result.objects[0]
    assert person.cls == "person"
    assert person.conf == 0.9
    assert person.bbox.x1 == 100
    assert person.bbox.y1 == 100
    assert person.bbox.x2 == 200
    assert person.bbox.y2 == 300


def test_detector_empty_frame():
    """Detector should handle frames with no detections."""
    from inference.detector import Detector

    mock_mm = MagicMock()
    mock_mm.detect.return_value = []

    detector = Detector(mock_mm)

    result = detector.run(b"fake_jpeg_bytes", "cam-01")

    assert len(result.objects) == 0
    assert result.camera_id == "cam-01"


def test_detector_corrupt_jpeg_returns_empty():
    """Detector should return empty DetectionResult when JPEG decoding fails."""
    from inference.detector import Detector

    mock_mm = MagicMock()
    mock_mm.preprocess_jpeg.return_value = None

    detector = Detector(mock_mm)

    result = detector.run(b"corrupt_bytes", "cam-01")

    assert isinstance(result, DetectionResult)
    assert result.camera_id == "cam-01"
    assert len(result.objects) == 0
    # detect() should NOT have been called since frame is None
    mock_mm.detect.assert_not_called()
