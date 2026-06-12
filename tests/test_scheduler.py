"""Tests for the round-robin scheduler."""
import pytest
from unittest.mock import MagicMock


def test_scheduler_registers_cameras():
    """Scheduler should accept camera registration."""
    from inference.scheduler import Scheduler

    sched = Scheduler()
    sched.register_camera("cam-01")
    sched.register_camera("cam-02")

    assert sched.camera_count == 2
    assert "cam-01" in sched._queues
    assert "cam-02" in sched._queues


def test_scheduler_round_robin_order():
    """Scheduler should return cameras in round-robin order."""
    from inference.scheduler import Scheduler

    sched = Scheduler()

    # Add frames to queues
    sched.register_camera("cam-01")
    sched.register_camera("cam-02")
    sched.add_frame("cam-01", b"frame1")
    sched.add_frame("cam-02", b"frame2")
    sched.add_frame("cam-01", b"frame3")

    # First poll → cam-01
    result = sched.poll()
    assert result is not None
    cam_id, frame = result
    assert cam_id == "cam-01"
    assert frame == b"frame1"

    # Second poll → cam-02
    result = sched.poll()
    assert result is not None
    cam_id, frame = result
    assert cam_id == "cam-02"
    assert frame == b"frame2"

    # Third poll → cam-01 again (round-robin)
    result = sched.poll()
    assert result is not None
    cam_id, frame = result
    assert cam_id == "cam-01"
    assert frame == b"frame3"


def test_scheduler_empty_queues():
    """Scheduler should return None when all queues are empty."""
    from inference.scheduler import Scheduler

    sched = Scheduler()
    sched.register_camera("cam-01")
    sched.register_camera("cam-02")

    assert sched.poll() is None


def test_scheduler_mixed_empty():
    """Scheduler should skip cameras with empty queues."""
    from inference.scheduler import Scheduler

    sched = Scheduler()
    sched.register_camera("cam-01")
    sched.register_camera("cam-02")

    # Only cam-02 has a frame
    sched.add_frame("cam-02", b"frame2")

    result = sched.poll()
    assert result is not None
    cam_id, frame = result
    assert cam_id == "cam-02"


def test_scheduler_queue_overflow():
    """When more than 5 frames are added, oldest frames should be evicted."""
    from inference.scheduler import Scheduler

    sched = Scheduler()
    sched.register_camera("cam-01")

    # Add 7 frames (maxlen=5, so first 2 should be dropped)
    for i in range(7):
        sched.add_frame("cam-01", f"frame{i}".encode())

    # The queue should have exactly 5 frames (frames 2-6)
    # Poll 5 times, should get frames 2, 3, 4, 5, 6
    for expected in range(2, 7):
        result = sched.poll()
        assert result is not None
        cam_id, frame = result
        assert cam_id == "cam-01"
        assert frame == f"frame{expected}".encode()

    # 6th poll should return None (all empty)
    assert sched.poll() is None
