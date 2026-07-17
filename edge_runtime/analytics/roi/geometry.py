"""Resolution-independent ROI and evidence intersection geometry."""

from dataclasses import dataclass
from enum import Enum

from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity

from shared.schemas import BoundingBox, ROIConfig


class EvidenceKind(str, Enum):
    HEAD = "head"
    TORSO = "torso"
    FEET = "feet"
    UPPER_BODY = "upper_body"
    PERSON = "person"
    BARE_HEAD_DETECTION = "bare_head_detection"
    NO_VEST_DETECTION = "no_vest_detection"
    NO_SHOES_DETECTION = "no_shoes_detection"


@dataclass(frozen=True, slots=True)
class EvidenceRegion:
    kind: EvidenceKind
    geometry: BaseGeometry
    source: str

    def __post_init__(self) -> None:
        if self.geometry.is_empty or not self.geometry.is_valid:
            raise ValueError("evidence geometry must be non-empty and valid")


@dataclass(frozen=True, slots=True)
class ROIIntersection:
    intersects: bool
    intersection_area: float
    evidence_overlap_ratio: float


def roi_polygon_pixels(roi: ROIConfig, frame_width: int, frame_height: int) -> Polygon:
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    polygon = Polygon([(x * frame_width, y * frame_height) for x, y in roi.polygon])
    if not polygon.is_valid:
        raise ValueError(f"invalid ROI polygon {roi.roi_id}: {explain_validity(polygon)}")
    if polygon.is_empty or polygon.area <= 0:
        raise ValueError(f"ROI polygon {roi.roi_id} has no area")
    return polygon


def bbox_region(kind: EvidenceKind, bbox: BoundingBox, source: str) -> EvidenceRegion:
    if bbox.area <= 0:
        raise ValueError("evidence bounding box must have area")
    return EvidenceRegion(kind, box(*bbox.as_xyxy()), source)


def intersect_evidence(
    roi_polygon: BaseGeometry,
    evidence: EvidenceRegion,
    *,
    minimum_overlap_ratio: float = 0.0,
) -> ROIIntersection:
    if not 0.0 <= minimum_overlap_ratio <= 1.0:
        raise ValueError("minimum_overlap_ratio must be in [0, 1]")
    intersection = roi_polygon.intersection(evidence.geometry)
    area = float(intersection.area)
    ratio = area / float(evidence.geometry.area) if evidence.geometry.area > 0 else 0.0
    # A zero threshold means any positive-area geometric intersection. Touching
    # a boundary at a single point is deliberately not enough evidence.
    matched = area > 0.0 and ratio >= minimum_overlap_ratio
    return ROIIntersection(matched, area, ratio)
