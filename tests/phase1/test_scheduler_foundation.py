import multiprocessing as mp
import time

import numpy as np

from edge_runtime.capture.latest_frame_buffer import LatestFrameBuffer
from edge_runtime.inference.interfaces import PoseRequest
from edge_runtime.inference.scheduler import BoundedPoseQueue, SharedInferenceScheduler
from shared.enums import PosePriority
from shared.schemas import SchedulerConfig


class EchoPipeline:
    def infer_frames(self, frames):
        return [{"value": int(packet.frame.mean())} for packet in frames]


def make_buffers(count: int = 3):
    context = mp.get_context("spawn")
    buffers = {}
    for index in range(count):
        camera_id = f"cam-{index}"
        buffer = LatestFrameBuffer(camera_id, 8, 8, context)
        buffer.publish(np.full((8, 8, 3), index, dtype=np.uint8))
        buffers[camera_id] = buffer
    return buffers


def test_scheduler_maps_batch_to_camera_and_frame() -> None:
    scheduler = SharedInferenceScheduler(
        make_buffers(3),
        EchoPipeline(),
        SchedulerConfig(detector_batch_size=3),
    )
    assert scheduler.run_once() == 3
    results = [scheduler.get_result() for _ in range(3)]
    assert {result.camera_id for result in results if result} == {
        "cam-0",
        "cam-1",
        "cam-2",
    }
    assert {result.output["value"] for result in results if result} == {0, 1, 2}


def test_scheduler_is_round_robin_when_batch_size_is_one() -> None:
    scheduler = SharedInferenceScheduler(
        make_buffers(2),
        EchoPipeline(),
        SchedulerConfig(detector_batch_size=1),
    )
    assert scheduler.run_once() == 1
    first = scheduler.get_result()
    assert scheduler.run_once() == 1
    second = scheduler.get_result()
    assert first and second and first.camera_id != second.camera_id


def test_high_priority_pose_replaces_low_priority_when_full() -> None:
    buffer = next(iter(make_buffers(1).values()))
    frame = buffer.read_latest()
    assert frame is not None
    queue = BoundedPoseQueue(maxsize=1)
    low = PoseRequest("cam-0", 1, frame, (0, 0, 1, 1), PosePriority.LOW, 1)
    high = PoseRequest("cam-0", 2, frame, (0, 0, 1, 1), PosePriority.HIGH, 2)
    assert queue.put(low)
    assert queue.put(high)
    assert queue.get() == high
    assert queue.dropped == 1


def test_scheduler_drops_stale_frame_without_inference() -> None:
    context = mp.get_context("spawn")
    buffer = LatestFrameBuffer("cam-stale", 8, 8, context)
    buffer.publish(
        np.zeros((8, 8, 3), dtype=np.uint8),
        captured_at_monotonic_ns=time.monotonic_ns() - 1_000_000_000,
    )
    scheduler = SharedInferenceScheduler(
        {"cam-stale": buffer},
        EchoPipeline(),
        SchedulerConfig(max_frame_age_ms=50),
    )
    assert scheduler.run_once() == 0
    assert scheduler.metrics.stale_frames == 1
    assert scheduler.run_once() == 0
    assert scheduler.metrics.stale_frames == 1


def test_scheduler_respects_per_camera_analytics_rate() -> None:
    buffers = make_buffers(1)
    scheduler = SharedInferenceScheduler(
        buffers,
        EchoPipeline(),
        SchedulerConfig(),
        analytics_fps={"cam-0": 1.0},
    )
    assert scheduler.run_once() == 1
    buffers["cam-0"].publish(np.ones((8, 8, 3), dtype=np.uint8))
    assert scheduler.run_once() == 0
