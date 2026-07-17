"""Creates and isolates one ByteTrack state machine per camera."""

from edge_runtime.tracking.byte_tracker import ByteTracker
from shared.schemas import Detection, TrackedDetection


class PerCameraTrackerManager:
    def __init__(self, camera_ids: list[str], **tracker_options: float | int) -> None:
        if len(camera_ids) != len(set(camera_ids)):
            raise ValueError("camera IDs must be unique")
        self._trackers = {
            camera_id: ByteTracker(camera_id, **tracker_options) for camera_id in camera_ids
        }

    def update(self, camera_id: str, detections: list[Detection]) -> list[TrackedDetection]:
        try:
            tracker = self._trackers[camera_id]
        except KeyError as exc:
            raise KeyError(f"unknown camera: {camera_id}") from exc
        return tracker.update(detections)
