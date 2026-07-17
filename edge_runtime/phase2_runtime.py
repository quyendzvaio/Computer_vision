"""Wires shared detection, per-camera ByteTrack and shared pose inference."""

import queue
import threading
import time

from edge_runtime.inference import ModelRegistry, SharedInferenceScheduler
from edge_runtime.inference.interfaces import PoseRequest
from edge_runtime.inference.pipelines import RTMDetPipeline, RTMPosePipeline
from edge_runtime.supervisor import EdgeSupervisor
from edge_runtime.tracking import PerCameraTrackerManager
from shared.enums import ModelRole, PosePriority
from shared.schemas import EdgeConfiguration, PoseResult, TrackedDetection


class Phase2Runtime:
    """One detector session and one pose session shared by all cameras."""

    def __init__(
        self,
        configuration: EdgeConfiguration,
        supervisor: EdgeSupervisor,
        registry: ModelRegistry,
    ) -> None:
        detector_names = registry.names_for_role(ModelRole.DETECTOR)
        if len(detector_names) != 1:
            raise ValueError("Phase 2 requires exactly one enabled person detector")
        detector_name = detector_names[0]
        detector = RTMDetPipeline(
            registry.get(detector_name),
            registry.configuration(detector_name),
        )
        buffers = {
            camera_id: runtime.buffer for camera_id, runtime in supervisor.cameras.items()
        }
        analytics_fps = {
            camera.camera_id: camera.analytics_fps
            for camera in configuration.cameras
            if camera.enabled
        }
        self.detector_scheduler = SharedInferenceScheduler(
            buffers,
            detector,
            configuration.scheduler,
            analytics_fps=analytics_fps,
            result_queue_size=max(4, len(buffers) * 2),
        )
        self.trackers = PerCameraTrackerManager(list(buffers))
        pose_names = registry.names_for_role(ModelRole.POSE)
        if len(pose_names) > 1:
            raise ValueError("Phase 2 supports one shared pose model")
        self.pose_pipeline = None
        if pose_names:
            pose_name = pose_names[0]
            self.pose_pipeline = RTMPosePipeline(
                registry.get(pose_name),
                registry.configuration(pose_name),
            )
        self.track_results: queue.Queue[list[TrackedDetection]] = queue.Queue(maxsize=16)
        self.pose_results: queue.Queue[PoseResult] = queue.Queue(maxsize=64)
        self._stop = threading.Event()
        self._consumer: threading.Thread | None = None
        self._pose_worker: threading.Thread | None = None
        self._last_pose_request_ns: dict[tuple[str, int], int] = {}
        self._pose_interval_ns = int(1_000_000_000 / configuration.scheduler.pose_fps_medium)

    def start(self) -> None:
        self._stop.clear()
        self.detector_scheduler.start()
        self._consumer = threading.Thread(
            target=self._consume_detections,
            name="detection-tracking-consumer",
            daemon=True,
        )
        self._consumer.start()
        if self.pose_pipeline is not None:
            self._pose_worker = threading.Thread(
                target=self._run_pose,
                name="shared-pose-worker",
                daemon=True,
            )
            self._pose_worker.start()

    def stop(self) -> None:
        self._stop.set()
        self.detector_scheduler.stop()
        if self._consumer:
            self._consumer.join(timeout=2.0)
        if self._pose_worker:
            self._pose_worker.join(timeout=2.0)

    def _consume_detections(self) -> None:
        while not self._stop.is_set():
            result = self.detector_scheduler.get_result(timeout=0.1)
            if result is None or result.frame is None:
                continue
            tracks = self.trackers.update(result.camera_id, list(result.output))
            self._put_latest(self.track_results, tracks)
            if self.pose_pipeline is None:
                continue
            now_ns = time.monotonic_ns()
            for track in tracks:
                identity = track.identity
                if now_ns - self._last_pose_request_ns.get(identity, 0) < self._pose_interval_ns:
                    continue
                accepted = self.detector_scheduler.pose_requests.put(
                    PoseRequest(
                        camera_id=track.camera_id,
                        track_id=track.track_id,
                        frame=result.frame,
                        bbox_xyxy=track.bbox.as_xyxy(),
                        priority=PosePriority.MEDIUM,
                        requested_at_monotonic_ns=now_ns,
                    )
                )
                if accepted:
                    self._last_pose_request_ns[identity] = now_ns

    def _run_pose(self) -> None:
        assert self.pose_pipeline is not None
        while not self._stop.is_set():
            requests: list[PoseRequest] = []
            while len(requests) < 8:
                request = self.detector_scheduler.pose_requests.get()
                if request is None:
                    break
                if request.frame.age_ms <= 1_000:
                    requests.append(request)
            if not requests:
                self._stop.wait(0.005)
                continue
            try:
                results = self.pose_pipeline.infer_requests(requests)
            except Exception:
                continue
            for result in results:
                self._put_latest(self.pose_results, result)

    @staticmethod
    def _put_latest(target: queue.Queue, value: object) -> None:
        if target.full():
            try:
                target.get_nowait()
            except queue.Empty:
                pass
        target.put_nowait(value)
