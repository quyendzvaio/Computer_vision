"""Fair latest-frame GPU scheduler shared by every camera."""

import heapq
import queue
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field

from edge_runtime.capture.latest_frame_buffer import FramePacket, LatestFrameBuffer
from edge_runtime.inference.interfaces import (
    FrameInferencePipeline,
    PoseRequest,
    ScheduledResult,
)
from shared.enums import PosePriority
from shared.errors import InferenceTimeoutError
from shared.schemas import SchedulerConfig

_PRIORITY_RANK = {
    PosePriority.HIGH: 0,
    PosePriority.MEDIUM: 1,
    PosePriority.LOW: 2,
}


@dataclass(slots=True)
class SchedulerMetrics:
    processed_frames: int = 0
    stale_frames: int = 0
    failed_batches: int = 0
    result_drops: int = 0
    circuit_open_count: int = 0
    last_batch_latency_ms: float = 0.0
    maximum_batch_latency_ms: float = 0.0


@dataclass(order=True, slots=True)
class _PrioritizedPoseRequest:
    rank: int
    requested_at_monotonic_ns: int
    request: PoseRequest = field(compare=False)


class BoundedPoseQueue:
    """Bounded priority queue that discards lower-priority work first."""

    def __init__(self, maxsize: int = 64) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be positive")
        self.maxsize = maxsize
        self._heap: list[_PrioritizedPoseRequest] = []
        self._lock = threading.Lock()
        self.dropped = 0

    def put(self, request: PoseRequest) -> bool:
        item = _PrioritizedPoseRequest(
            _PRIORITY_RANK[request.priority],
            request.requested_at_monotonic_ns,
            request,
        )
        with self._lock:
            if len(self._heap) < self.maxsize:
                heapq.heappush(self._heap, item)
                return True
            worst_index = max(
                range(len(self._heap)),
                key=lambda index: (
                    self._heap[index].rank,
                    -self._heap[index].requested_at_monotonic_ns,
                ),
            )
            worst = self._heap[worst_index]
            if item.rank > worst.rank:
                self.dropped += 1
                return False
            self._heap[worst_index] = item
            heapq.heapify(self._heap)
            self.dropped += 1
            return True

    def get(self) -> PoseRequest | None:
        with self._lock:
            if not self._heap:
                return None
            return heapq.heappop(self._heap).request

    def __len__(self) -> int:
        with self._lock:
            return len(self._heap)


class SharedInferenceScheduler:
    """Pull at most the newest fresh frame per camera into bounded batches."""

    def __init__(
        self,
        buffers: Mapping[str, LatestFrameBuffer],
        pipeline: FrameInferencePipeline,
        configuration: SchedulerConfig,
        *,
        analytics_fps: Mapping[str, float] | None = None,
        result_queue_size: int = 64,
    ) -> None:
        self._buffers = dict(buffers)
        self._pipeline = pipeline
        self._configuration = configuration
        self._camera_order = list(self._buffers)
        self._cursor = 0
        self._last_sequences = {camera_id: 0 for camera_id in self._buffers}
        self._last_scheduled_ns = {camera_id: 0 for camera_id in self._buffers}
        self._analytics_intervals_ns = {
            camera_id: int(1_000_000_000 / fps)
            for camera_id, fps in (analytics_fps or {}).items()
            if fps > 0
        }
        self._results: queue.Queue[ScheduledResult] = queue.Queue(maxsize=result_queue_size)
        self.pose_requests = BoundedPoseQueue()
        self.metrics = SchedulerMetrics()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._consecutive_failures = 0
        self._circuit_open_until_ns = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="shared-inference-scheduler",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)

    def run_once(self) -> int:
        if time.monotonic_ns() < self._circuit_open_until_ns:
            return 0
        batch = self._collect_batch()
        if not batch:
            return 0
        started = time.perf_counter()
        try:
            outputs = list(self._pipeline.infer_frames(batch))
            elapsed_ms = (time.perf_counter() - started) * 1_000
            if elapsed_ms > self._configuration.inference_timeout_ms:
                raise InferenceTimeoutError(
                    f"inference batch exceeded deadline: {elapsed_ms:.1f}ms"
                )
            if len(outputs) != len(batch):
                raise RuntimeError(
                    f"batch mapping mismatch: {len(batch)} inputs, {len(outputs)} outputs"
                )
        except Exception:
            self.metrics.failed_batches += 1
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._configuration.circuit_breaker_failures:
                self._circuit_open_until_ns = (
                    time.monotonic_ns()
                    + self._configuration.circuit_breaker_reset_ms * 1_000_000
                )
                self.metrics.circuit_open_count += 1
                self._consecutive_failures = 0
            return 0

        self._consecutive_failures = 0
        self.metrics.last_batch_latency_ms = elapsed_ms
        self.metrics.maximum_batch_latency_ms = max(
            self.metrics.maximum_batch_latency_ms,
            elapsed_ms,
        )
        for packet, output in zip(batch, outputs):
            result = ScheduledResult(
                camera_id=packet.camera_id,
                frame_id=packet.frame_id,
                captured_at_unix_ns=packet.captured_at_unix_ns,
                frame_age_ms=packet.age_ms,
                output=output,
                frame=packet,
            )
            self._put_result(result)
        self.metrics.processed_frames += len(batch)
        return len(batch)

    def get_result(self, timeout: float | None = None) -> ScheduledResult | None:
        try:
            return self._results.get(timeout=timeout)
        except queue.Empty:
            return None

    def _collect_batch(self) -> list[FramePacket]:
        if not self._camera_order:
            return []
        batch: list[FramePacket] = []
        checked = 0
        while (
            checked < len(self._camera_order)
            and len(batch) < self._configuration.detector_batch_size
        ):
            camera_id = self._camera_order[self._cursor]
            self._cursor = (self._cursor + 1) % len(self._camera_order)
            checked += 1
            interval_ns = self._analytics_intervals_ns.get(camera_id, 0)
            since_last_ns = time.monotonic_ns() - self._last_scheduled_ns[camera_id]
            if interval_ns and since_last_ns < interval_ns:
                continue
            buffer = self._buffers[camera_id]
            before_stale = buffer.stale_frames
            packet = buffer.read_latest(
                after_sequence=self._last_sequences[camera_id],
                max_age_ms=self._configuration.max_frame_age_ms,
            )
            if buffer.stale_frames > before_stale:
                self.metrics.stale_frames += 1
                self._last_sequences[camera_id] = buffer.sequence
            if packet is None:
                continue
            self._last_sequences[camera_id] = packet.frame_id
            self._last_scheduled_ns[camera_id] = time.monotonic_ns()
            batch.append(packet)
        return batch

    def _put_result(self, result: ScheduledResult) -> None:
        if self._results.full():
            try:
                self._results.get_nowait()
                self.metrics.result_drops += 1
            except queue.Empty:
                pass
        self._results.put_nowait(result)

    def _run_loop(self) -> None:
        idle_seconds = self._configuration.idle_poll_ms / 1_000.0
        while not self._stop.is_set():
            processed = self.run_once()
            if processed == 0:
                self._stop.wait(idle_seconds)
