"""Build rule-specific body evidence from visible COCO-17 keypoints."""

from collections.abc import Iterable

from shapely.geometry import MultiPoint, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from edge_runtime.analytics.roi.geometry import EvidenceKind, EvidenceRegion
from shared.schemas import Keypoint, PoseResult


def _visible_points(pose: PoseResult, names: Iterable[str]) -> list[tuple[float, float]]:
    wanted = set(names)
    return [
        (point.x, point.y)
        for point in pose.keypoints
        if point.name in wanted and point.visible and point.x is not None and point.y is not None
    ]


def _hull_or_buffer(points: list[tuple[float, float]], radius: float) -> BaseGeometry | None:
    if not points:
        return None
    return MultiPoint(points).convex_hull.buffer(radius)


def pose_evidence_regions(pose: PoseResult) -> dict[EvidenceKind, EvidenceRegion]:
    """Return only regions supported by visible keypoints.

    Missing/occluded landmarks produce no region; callers can then emit
    UNVERIFIABLE instead of treating absence as a violation.
    """

    scale = max(pose.bbox.width, pose.bbox.height, 1.0)
    regions: dict[EvidenceKind, EvidenceRegion] = {}

    head_points = _visible_points(
        pose,
        ("nose", "left_eye", "right_eye", "left_ear", "right_ear"),
    )
    head = _hull_or_buffer(head_points, scale * 0.035)
    if head is not None and not head.is_empty:
        regions[EvidenceKind.HEAD] = EvidenceRegion(EvidenceKind.HEAD, head, "pose")

    torso_points = _visible_points(
        pose,
        ("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
    )
    torso = _hull_or_buffer(torso_points, scale * 0.02) if len(torso_points) >= 2 else None
    if torso is not None and not torso.is_empty:
        regions[EvidenceKind.TORSO] = EvidenceRegion(EvidenceKind.TORSO, torso, "pose")

    ankle_points = _visible_points(pose, ("left_ankle", "right_ankle"))
    if ankle_points:
        feet = unary_union([Point(point).buffer(scale * 0.04) for point in ankle_points])
        regions[EvidenceKind.FEET] = EvidenceRegion(EvidenceKind.FEET, feet, "pose")

    upper_parts = [geometry for geometry in (head, torso) if geometry is not None]
    if upper_parts:
        regions[EvidenceKind.UPPER_BODY] = EvidenceRegion(
            EvidenceKind.UPPER_BODY,
            unary_union(upper_parts),
            "pose",
        )
    return regions
