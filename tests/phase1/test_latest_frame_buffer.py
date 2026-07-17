import multiprocessing as mp
import time

import numpy as np

from edge_runtime.capture.latest_frame_buffer import LatestFrameBuffer


def make_buffer() -> LatestFrameBuffer:
    return LatestFrameBuffer("cam-test", 16, 12, mp.get_context("spawn"))


def test_buffer_returns_only_latest_frame() -> None:
    buffer = make_buffer()
    buffer.publish(np.zeros((12, 16, 3), dtype=np.uint8))
    newest = buffer.publish(np.full((12, 16, 3), 9, dtype=np.uint8))

    packet = buffer.read_latest()

    assert packet is not None
    assert packet.frame_id == newest
    assert int(packet.frame.mean()) == 9
    assert buffer.dropped_frames == 1
    assert buffer.read_latest(after_sequence=newest) is None


def test_stale_frame_is_dropped_once() -> None:
    buffer = make_buffer()
    buffer.publish(
        np.zeros((12, 16, 3), dtype=np.uint8),
        captured_at_monotonic_ns=time.monotonic_ns() - 2_000_000_000,
    )
    assert buffer.read_latest(max_age_ms=100) is None
    assert buffer.stale_frames == 1


def test_rejects_frame_larger_than_slot() -> None:
    buffer = make_buffer()
    try:
        buffer.publish(np.zeros((13, 16, 3), dtype=np.uint8))
    except ValueError as exc:
        assert "exceeds buffer" in str(exc)
    else:
        raise AssertionError("oversized frame was accepted")
