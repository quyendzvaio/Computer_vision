"""Shared data models used across all subsystems."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple


@dataclass
class BBox:
    """Bounding box [x1, y1, x2, y2] normalized or pixel coords."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0.0


@dataclass
class DetectedObject:
    """Single object detection result."""
    bbox: BBox
    cls: str          # 'person', 'helmet', 'vest', 'boot'
    conf: float       # confidence 0-1


@dataclass
class Keypoint:
    """Single pose keypoint."""
    x: float
    y: float
    conf: float        # visibility confidence


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
    type: str = ""           # FALL | NO_HELMET | NO_VEST | NO_BOOT
    severity: str = "MEDIUM"  # HIGH | MEDIUM
    bbox: List[float] = field(default_factory=list)   # [x1, y1, x2, y2]
    thumbnail_path: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ROIConfig:
    """ROI polygon configuration for one camera."""
    camera_id: str
    polygon: List[Tuple[float, float]]  # [(x,y), (x,y), ...]
    updated_at: datetime = field(default_factory=datetime.now)
