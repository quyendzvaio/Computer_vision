"""Rule-to-evidence mapping used before temporal safety decisions."""

from dataclasses import dataclass

from edge_runtime.analytics.roi.geometry import (
    EvidenceKind,
    EvidenceRegion,
    ROIIntersection,
    intersect_evidence,
    roi_polygon_pixels,
)
from shared.schemas import ROIConfig


RULE_EVIDENCE_KINDS: dict[str, frozenset[EvidenceKind]] = {
    "no_helmet": frozenset({EvidenceKind.HEAD, EvidenceKind.BARE_HEAD_DETECTION}),
    "no_vest": frozenset({EvidenceKind.TORSO, EvidenceKind.NO_VEST_DETECTION}),
    "no_shoes": frozenset({EvidenceKind.FEET, EvidenceKind.NO_SHOES_DETECTION}),
    "fall": frozenset({EvidenceKind.UPPER_BODY, EvidenceKind.PERSON}),
}


@dataclass(frozen=True, slots=True)
class RuleROIEvidence:
    rule: str
    verifiable: bool
    matched: bool
    intersections: tuple[ROIIntersection, ...]


def evaluate_rule_roi(
    rule: str,
    roi: ROIConfig,
    frame_width: int,
    frame_height: int,
    evidence_regions: list[EvidenceRegion],
    *,
    minimum_overlap_ratio: float = 0.0,
) -> RuleROIEvidence:
    try:
        allowed = RULE_EVIDENCE_KINDS[rule]
    except KeyError as exc:
        raise KeyError(f"unknown ROI evidence rule: {rule}") from exc
    relevant = [region for region in evidence_regions if region.kind in allowed]
    if not relevant:
        return RuleROIEvidence(rule, verifiable=False, matched=False, intersections=())
    polygon = roi_polygon_pixels(roi, frame_width, frame_height)
    intersections = tuple(
        intersect_evidence(
            polygon,
            region,
            minimum_overlap_ratio=minimum_overlap_ratio,
        )
        for region in relevant
    )
    return RuleROIEvidence(
        rule,
        verifiable=True,
        matched=any(result.intersects for result in intersections),
        intersections=intersections,
    )
