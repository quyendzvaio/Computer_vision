"""Coordinate-safe vision results shared by inference, tracking and analytics."""

from dataclasses import dataclass


COCO17_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("bounding box coordinates must be ordered")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass(frozen=True, slots=True)
class Detection:
    bbox: BoundingBox
    score: float
    class_id: int
    class_name: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("detection score must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class TrackedDetection:
    camera_id: str
    track_id: int
    bbox: BoundingBox
    score: float
    class_id: int = 0
    class_name: str = "person"

    @property
    def identity(self) -> tuple[str, int]:
        return self.camera_id, self.track_id


@dataclass(frozen=True, slots=True)
class Keypoint:
    name: str
    x: float | None
    y: float | None
    confidence: float
    visible: bool

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("keypoint confidence must be in [0, 1]")
        if self.visible and (self.x is None or self.y is None):
            raise ValueError("visible keypoints require coordinates")
        if not self.visible and (self.x is not None or self.y is not None):
            raise ValueError("invisible keypoints must not contain fake coordinates")


@dataclass(frozen=True, slots=True)
class PoseResult:
    camera_id: str
    track_id: int
    bbox: BoundingBox
    keypoints: tuple[Keypoint, ...]

    @property
    def identity(self) -> tuple[str, int]:
        return self.camera_id, self.track_id

    def keypoint(self, name: str) -> Keypoint:
        for point in self.keypoints:
            if point.name == name:
                return point
        raise KeyError(name)
