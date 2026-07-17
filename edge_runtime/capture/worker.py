"""Generic camera capture worker; every camera runs in a separate process."""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass, field
from typing import Protocol

from edge_runtime.capture.camera_health import CameraHealth
from edge_runtime.capture.device_registry import DeviceRegistry
from edge_runtime.capture.latest_frame_buffer import LatestFrameBuffer
from edge_runtime.capture.reconnect_policy import ReconnectPolicy
from shared.enums import CameraState
from shared.schemas import CameraConfig


class _ProcessEvent(Protocol):
    def clear(self) -> None: ...

    def set(self) -> None: ...

    def is_set(self) -> bool: ...

    def wait(self, timeout: float | None = None) -> bool: ...


def _capture_process_main(
    config: CameraConfig,
    buffer: LatestFrameBuffer,
    health: CameraHealth,
    stop_event: _ProcessEvent,
) -> None:
    import cv2

    policy = ReconnectPolicy(config.reconnect)
    frame_interval = 1.0 / config.capture_fps
    device = DeviceRegistry.resolve(config.device_path)
    health.set_state(CameraState.STARTING)

    while not stop_event.is_set():
        capture = cv2.VideoCapture(device)
        if not capture.isOpened():
            health.record_failure()
            health.set_state(CameraState.DISCONNECTED)
            stop_event.wait(policy.next_delay_seconds())
            health.record_reconnect()
            continue

        width, height = config.resolution
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, config.capture_fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        policy.reset()
        window_started = time.monotonic()
        window_frames = 0

        try:
            while not stop_event.is_set():
                cycle_started = time.monotonic()
                ok, frame = capture.read()
                if not ok or frame is None:
                    health.record_failure()
                    break
                actual_height, actual_width = frame.shape[:2]
                try:
                    buffer.publish(frame)
                except ValueError:
                    health.record_failure()
                    health.set_state(CameraState.DEGRADED)
                    break
                window_frames += 1
                elapsed_window = time.monotonic() - window_started
                fps = window_frames / elapsed_window if elapsed_window > 0 else 0.0
                health.record_success(actual_width, actual_height, fps)
                if (actual_width, actual_height) != config.resolution:
                    health.record_resolution_mismatch()
                health.set_dropped_frames(buffer.dropped_frames)
                if elapsed_window >= 2.0:
                    window_started = time.monotonic()
                    window_frames = 0
                remaining = frame_interval - (time.monotonic() - cycle_started)
                if remaining > 0:
                    stop_event.wait(remaining)
        finally:
            capture.release()

        if not stop_event.is_set():
            health.set_state(CameraState.DISCONNECTED)
            stop_event.wait(policy.next_delay_seconds())
            health.record_reconnect()

    health.set_state(CameraState.STOPPED)


@dataclass(slots=True)
class CaptureWorker:
    config: CameraConfig
    buffer: LatestFrameBuffer
    health: CameraHealth
    context: mp.context.BaseContext
    _stop_event: _ProcessEvent = field(init=False, repr=False)
    _process: mp.Process | None = field(init=False, default=None, repr=False)
    started_monotonic_ns: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._stop_event = self.context.Event()

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process else None

    def is_alive(self) -> bool:
        return bool(self._process and self._process.is_alive())

    def start(self) -> None:
        if self.is_alive():
            return
        self._stop_event.clear()
        self._process = self.context.Process(
            target=_capture_process_main,
            args=(self.config, self.buffer, self.health, self._stop_event),
            name=f"capture-{self.config.camera_id}",
            daemon=False,
        )
        self._process.start()
        self.started_monotonic_ns = time.monotonic_ns()

    def stop(self, timeout: float = 3.0) -> None:
        if self._process is None:
            return
        self._stop_event.set()
        self._process.join(timeout)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
        self.health.set_state(CameraState.STOPPED)

    def restart(self) -> None:
        self.stop()
        self.health.record_reconnect()
        self.start()
