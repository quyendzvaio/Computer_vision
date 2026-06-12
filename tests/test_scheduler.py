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
