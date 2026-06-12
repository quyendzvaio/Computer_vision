"""Shared data models used across all subsystems.

Coordinate convention: All bounding box and keypoint coordinates are in
PIXEL space (image coordinates), not normalized. x=0,y=0 is top-left.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Literal, Optional, Tuple


@dataclass
class BBox:
    """Bounding box [x1, y1, x2, y2] in pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> Tuple[float, float]:
        """Center point (cx, cy) in pixel coordinates."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def aspect_ratio(self) -> float:
        """width/height ratio. Returns 0.0 if height is 0."""
        return self.width / self.height if self.height > 0 else 0.0

    def to_list(self) -> List[float]:
        """Serialize to [x1, y1, x2, y2]."""
        return [self.x1, self.y1, self.x2, self.y2]

    @classmethod
    def from_list(cls, coords: List[float]) -> "BBox":
        """Deserialize from [x1, y1, x2, y2]."""
        return cls(coords[0], coords[1], coords[2], coords[3])


# Type aliases for violation classification
ViolationType = Literal["FALL", "NO_HELMET", "NO_VEST", "NO_BOOT"]
SeverityLevel = Literal["HIGH", "MEDIUM"]


@dataclass
class DetectedObject:
    """Single object detection result."""
    bbox: BBox
    cls: str          # 'person', 'helmet', 'vest', 'boot'
    conf: float       # confidence 0-1


@dataclass
class Keypoint:
    """Single pose keypoint in pixel coordinates."""
    x: float        # pixel x-coordinate
    y: float        # pixel y-coordinate
    conf: float     # visibility confidence 0-1


@dataclass
class DetectionResult:
    """Output from one inference run for one frame."""
    camera_id: str
    objects: List[DetectedObject] = field(default_factory=list)
    keypoints: Optional[List[List[Keypoint]]] = None  # per-person keypoints (17 each)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Violation:
    """A confirmed safety violation after ROI + cooldown checks."""
    id: Optional[int] = None
    camera_id: str = ""
    type: ViolationType = "NO_HELMET"
    severity: SeverityLevel = "MEDIUM"
    bbox: BBox = field(default_factory=lambda: BBox(0, 0, 0, 0))
    thumbnail_path: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ROIConfig:
    """ROI polygon configuration for one camera.

    Polygon is a list of [x, y] coordinate pairs in pixel coordinates.
    Use `to_point_tuples()` to get Shapely-compatible (x, y) tuples.
    """
    camera_id: str
    polygon: List[List[float]]  # [[x, y], [x, y], ...]
    updated_at: datetime = field(default_factory=datetime.now)

    def to_point_tuples(self) -> List[Tuple[float, float]]:
        """Return polygon as List[Tuple[float, float]] for Shapely."""
        return [(float(p[0]), float(p[1])) for p in self.polygon]
