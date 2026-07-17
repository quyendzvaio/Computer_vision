"""Own capture processes and restart a hung camera without affecting peers."""

import multiprocessing as mp
import threading
import time
from dataclasses import dataclass

from edge_runtime.capture.camera_health import CameraHealth, CameraHealthSnapshot
from edge_runtime.capture.latest_frame_buffer import LatestFrameBuffer
from edge_runtime.capture.worker import CaptureWorker
from shared.enums import CameraState
from shared.schemas import EdgeConfiguration


@dataclass(frozen=True, slots=True)
class CameraRuntime:
    worker: CaptureWorker
    buffer: LatestFrameBuffer
    health: CameraHealth


class EdgeSupervisor:
    """Lifecycle boundary for isolated capture workers and shared services."""

    def __init__(
        self,
        configuration: EdgeConfiguration,
        *,
        context: mp.context.BaseContext | None = None,
        watchdog_interval_seconds: float = 1.0,
    ) -> None:
        self.configuration = configuration
        self._context = context or mp.get_context("spawn")
        self._watchdog_interval = watchdog_interval_seconds
        self._stop = threading.Event()
        self._watchdog: threading.Thread | None = None
        self.cameras: dict[str, CameraRuntime] = {}

        for camera in configuration.cameras:
            if not camera.enabled:
                continue
            width, height = camera.resolution
            buffer = LatestFrameBuffer(camera.camera_id, width, height, self._context)
            health = CameraHealth(self._context)
            worker = CaptureWorker(camera, buffer, health, self._context)
            self.cameras[camera.camera_id] = CameraRuntime(worker, buffer, health)

    def start(self) -> None:
        self._stop.clear()
        for runtime in self.cameras.values():
            runtime.worker.start()
        self._watchdog = threading.Thread(
            target=self._watchdog_loop,
            name="edge-watchdog",
            daemon=True,
        )
        self._watchdog.start()

    def stop(self) -> None:
        self._stop.set()
        if self._watchdog:
            self._watchdog.join(timeout=2.0)
        for runtime in self.cameras.values():
            runtime.worker.stop()

    def health_snapshots(self) -> dict[str, CameraHealthSnapshot]:
        return {
            camera_id: runtime.health.snapshot()
            for camera_id, runtime in self.cameras.items()
        }

    def _watchdog_loop(self) -> None:
        while not self._stop.wait(self._watchdog_interval):
            now_ns = time.monotonic_ns()
            for runtime in self.cameras.values():
                worker = runtime.worker
                snapshot = runtime.health.snapshot()
                if not worker.is_alive():
                    worker.restart()
                    continue
                if snapshot.state == CameraState.DISCONNECTED:
                    continue
                reference_ns = snapshot.last_success_monotonic_ns or worker.started_monotonic_ns
                timeout_ns = worker.config.reconnect.read_timeout_ms * 1_000_000
                if reference_ns and now_ns - reference_ns > timeout_ns:
                    worker.restart()

    def __enter__(self) -> "EdgeSupervisor":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
