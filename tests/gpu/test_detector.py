"""Test YOLODetector preprocessing and postprocessing logic."""
import numpy as np
import pytest
from gpu.detector import YOLODetector


def test_preprocess_output_shape():
    """Preprocess should return NCHW float32 tensor."""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    det = YOLODetector.__new__(YOLODetector)
    det.input_w, det.input_h = 640, 640
    tensor = det.preprocess(frame)
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert 0.0 <= tensor.min() <= tensor.max() <= 1.0


def test_postprocess_empty():
    """Postprocess with no detections above threshold returns empty list."""
    det = YOLODetector.__new__(YOLODetector)
    det.input_w, det.input_h = 640, 640
    dummy = np.zeros((1, 84, 8400), dtype=np.float32)
    result = det.postprocess(dummy, (480, 640))
    assert result == []
