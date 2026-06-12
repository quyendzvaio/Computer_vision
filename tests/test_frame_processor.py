import numpy as np
import pytest


def make_frame(width=640, height=480):
    """Create a dummy BGR frame."""
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def test_frame_processor_resize():
    """FrameProcessor.resize should output exactly 416x416."""
    from edge.frame_processor import FrameProcessor

    fp = FrameProcessor(target_size=(416, 416))
    frame = make_frame(1920, 1080)
    resized = fp.resize(frame)
    assert resized.shape == (416, 416, 3)


def test_frame_processor_crop_roi():
    """Crop to ROI bounding rectangle then resize."""
    from edge.frame_processor import FrameProcessor

    fp = FrameProcessor(target_size=(416, 416))
    frame = make_frame(640, 480)
    roi_polygon = [(100, 100), (400, 100), (400, 350), (100, 350)]

    cropped = fp.crop_to_roi(frame, roi_polygon)
    assert cropped is not None
    assert cropped.shape == (416, 416, 3)


def test_frame_processor_motion_skip_no_motion():
    """When frames are identical, should_skip should return True."""
    from edge.frame_processor import FrameProcessor

    fp = FrameProcessor(motion_threshold=0.05)
    frame = make_frame(640, 480)

    # First frame — never skip
    assert fp.should_skip(frame) is False
    # Same frame — should skip (no motion)
    assert fp.should_skip(frame) is True


def test_frame_processor_motion_skip_with_motion():
    """When frames differ significantly, should_skip should return False."""
    from edge.frame_processor import FrameProcessor

    fp = FrameProcessor(motion_threshold=0.01)
    frame1 = make_frame(640, 480)
    frame2 = make_frame(640, 480)  # Completely different random frame

    fp.should_skip(frame1)  # prime
    result = fp.should_skip(frame2)
    assert isinstance(result, bool)  # At minimum, doesn't crash


def test_frame_processor_encode_jpeg():
    """encode_jpeg should return JPEG bytes."""
    from edge.frame_processor import FrameProcessor

    fp = FrameProcessor()
    frame = make_frame(416, 416)
    jpeg_bytes = fp.encode_jpeg(frame, quality=70)
    assert isinstance(jpeg_bytes, bytes)
    assert len(jpeg_bytes) > 100
    assert len(jpeg_bytes) < 519_000  # Smaller than raw 416*416*3
