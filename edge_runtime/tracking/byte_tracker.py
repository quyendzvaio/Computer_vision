"""Per-camera ByteTrack with two-stage high/low-score association."""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.optimize import linear_sum_assignment

from shared.schemas import BoundingBox, Detection, TrackedDetection


class TrackState(str, Enum):
    TRACKED = "tracked"
    LOST = "lost"
    REMOVED = "removed"


class KalmanFilterXYAH:
    """Constant-velocity Kalman filter used by the original ByteTrack design."""

    def __init__(self) -> None:
        self.motion = np.eye(8, dtype=np.float64)
        for index in range(4):
            self.motion[index, index + 4] = 1.0
        self.update_matrix = np.eye(4, 8, dtype=np.float64)

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.r_[measurement, np.zeros(4, dtype=np.float64)]
        height = max(float(measurement[3]), 1.0)
        standard_deviation = np.asarray(
            [
                height / 10,
                height / 10,
                1e-2,
                height / 10,
                height / 16,
                height / 16,
                1e-5,
                height / 16,
            ]
        )
        return mean, np.diag(np.square(standard_deviation))

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height = max(float(mean[3]), 1.0)
        deviation = np.asarray(
            [
                height / 20,
                height / 20,
                1e-2,
                height / 20,
                height / 160,
                height / 160,
                1e-5,
                height / 160,
            ]
        )
        motion_covariance = np.diag(np.square(deviation))
        return (
            self.motion @ mean,
            self.motion @ covariance @ self.motion.T + motion_covariance,
        )

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        height = max(float(mean[3]), 1.0)
        innovation_covariance = np.diag(
            np.square(np.asarray([height / 20, height / 20, 0.1, height / 20]))
        )
        projected_mean = self.update_matrix @ mean
        projected_covariance = (
            self.update_matrix @ covariance @ self.update_matrix.T + innovation_covariance
        )
        gain = np.linalg.solve(
            projected_covariance,
            self.update_matrix @ covariance,
        ).T
        innovation = measurement - projected_mean
        updated_mean = mean + gain @ innovation
        updated_covariance = covariance - gain @ projected_covariance @ gain.T
        return updated_mean, updated_covariance


def _xyxy_to_xyah(box: BoundingBox) -> np.ndarray:
    height = max(box.height, 1e-6)
    return np.asarray(
        [(box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2, box.width / height, height],
        dtype=np.float64,
    )


def _xyah_to_bbox(values: np.ndarray) -> BoundingBox:
    center_x, center_y, aspect, height = values[:4]
    width = max(aspect * height, 0.0)
    height = max(height, 0.0)
    return BoundingBox(
        float(center_x - width / 2),
        float(center_y - height / 2),
        float(center_x + width / 2),
        float(center_y + height / 2),
    )


def _iou(first: BoundingBox, second: BoundingBox) -> float:
    width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    intersection = width * height
    union = first.area + second.area - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(slots=True)
class _Track:
    track_id: int
    mean: np.ndarray
    covariance: np.ndarray
    score: float
    class_id: int
    class_name: str
    last_frame: int
    state: TrackState = TrackState.TRACKED

    @property
    def bbox(self) -> BoundingBox:
        return _xyah_to_bbox(self.mean)


class ByteTracker:
    """A tracker instance must belong to exactly one camera."""

    def __init__(
        self,
        camera_id: str,
        *,
        track_threshold: float = 0.5,
        low_threshold: float = 0.1,
        match_iou_threshold: float = 0.3,
        track_buffer: int = 30,
    ) -> None:
        if not camera_id:
            raise ValueError("camera_id is required")
        self.camera_id = camera_id
        self.track_threshold = track_threshold
        self.low_threshold = low_threshold
        self.match_iou_threshold = match_iou_threshold
        self.track_buffer = track_buffer
        self._kalman = KalmanFilterXYAH()
        self._tracks: list[_Track] = []
        self._next_track_id = 1
        self._frame_id = 0

    def update(self, detections: list[Detection]) -> list[TrackedDetection]:
        self._frame_id += 1
        candidates = [
            track for track in self._tracks if track.state != TrackState.REMOVED
        ]
        for track in candidates:
            track.mean, track.covariance = self._kalman.predict(track.mean, track.covariance)

        high = [detection for detection in detections if detection.score >= self.track_threshold]
        low = [
            detection
            for detection in detections
            if self.low_threshold <= detection.score < self.track_threshold
        ]
        unmatched_tracks, unmatched_high = self._associate(candidates, high)
        matched_track_ids = {track.track_id for track in candidates} - {
            track.track_id for track in unmatched_tracks
        }

        # ByteTrack's key property: recover unmatched existing tracks with
        # lower-score detections instead of discarding those observations.
        second_stage_tracks = [
            track for track in unmatched_tracks if track.state == TrackState.TRACKED
        ]
        unmatched_lost = [
            track for track in unmatched_tracks if track.state == TrackState.LOST
        ]
        remaining_second_stage, _ = self._associate(second_stage_tracks, low)
        remaining_tracks = unmatched_lost + remaining_second_stage
        remaining_ids = {track.track_id for track in remaining_tracks}
        matched_track_ids.update(
            track.track_id for track in unmatched_tracks if track.track_id not in remaining_ids
        )

        for track in remaining_tracks:
            if self._frame_id - track.last_frame > self.track_buffer:
                track.state = TrackState.REMOVED
            else:
                track.state = TrackState.LOST

        for detection in unmatched_high:
            mean, covariance = self._kalman.initiate(_xyxy_to_xyah(detection.bbox))
            track = _Track(
                track_id=self._next_track_id,
                mean=mean,
                covariance=covariance,
                score=detection.score,
                class_id=detection.class_id,
                class_name=detection.class_name,
                last_frame=self._frame_id,
            )
            self._next_track_id += 1
            self._tracks.append(track)
            matched_track_ids.add(track.track_id)

        self._tracks = [track for track in self._tracks if track.state != TrackState.REMOVED]
        return [
            TrackedDetection(
                camera_id=self.camera_id,
                track_id=track.track_id,
                bbox=track.bbox,
                score=track.score,
                class_id=track.class_id,
                class_name=track.class_name,
            )
            for track in self._tracks
            if track.state == TrackState.TRACKED and track.track_id in matched_track_ids
        ]

    def _associate(
        self,
        tracks: list[_Track],
        detections: list[Detection],
    ) -> tuple[list[_Track], list[Detection]]:
        if not tracks or not detections:
            return list(tracks), list(detections)
        cost = np.asarray(
            [
                [1.0 - _iou(track.bbox, detection.bbox) for detection in detections]
                for track in tracks
            ]
        )
        rows, columns = linear_sum_assignment(cost)
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for row, column in zip(rows, columns):
            if 1.0 - cost[row, column] < self.match_iou_threshold:
                continue
            track = tracks[int(row)]
            detection = detections[int(column)]
            track.mean, track.covariance = self._kalman.update(
                track.mean,
                track.covariance,
                _xyxy_to_xyah(detection.bbox),
            )
            track.score = detection.score
            track.class_id = detection.class_id
            track.class_name = detection.class_name
            track.last_frame = self._frame_id
            track.state = TrackState.TRACKED
            matched_tracks.add(int(row))
            matched_detections.add(int(column))
        return (
            [track for index, track in enumerate(tracks) if index not in matched_tracks],
            [
                detection
                for index, detection in enumerate(detections)
                if index not in matched_detections
            ],
        )
