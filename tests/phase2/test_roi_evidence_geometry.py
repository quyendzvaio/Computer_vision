from edge_runtime.analytics.roi import (
    EvidenceKind,
    EvidenceRegion,
    evaluate_rule_roi,
    pose_evidence_regions,
    roi_polygon_pixels,
)
from shapely.geometry import box

from shared.enums import Severity
from shared.schemas import BoundingBox, Keypoint, PoseResult, ROIConfig


def roi() -> ROIConfig:
    return ROIConfig(
        roi_id="zone-1",
        camera_id="cam-1",
        polygon=[(0.0, 0.0), (0.5, 0.0), (0.5, 0.4), (0.2, 0.4), (0.2, 1.0), (0.0, 1.0)],
        severity=Severity.HIGH,
    )


def test_normalized_concave_roi_scales_with_frame_resolution():
    first = roi_polygon_pixels(roi(), 100, 100)
    second = roi_polygon_pixels(roi(), 200, 200)
    assert second.area == first.area * 4
    assert first.is_valid


def test_rule_uses_corresponding_evidence_not_whole_person_containment():
    head_inside = EvidenceRegion(EvidenceKind.HEAD, box(5, 5, 15, 15), "pose")
    torso_outside = EvidenceRegion(EvidenceKind.TORSO, box(70, 50, 90, 90), "pose")
    helmet = evaluate_rule_roi("no_helmet", roi(), 100, 100, [head_inside, torso_outside])
    vest = evaluate_rule_roi("no_vest", roi(), 100, 100, [head_inside, torso_outside])
    shoes = evaluate_rule_roi("no_shoes", roi(), 100, 100, [head_inside, torso_outside])
    assert helmet.verifiable and helmet.matched
    assert vest.verifiable and not vest.matched
    assert not shoes.verifiable and not shoes.matched


def test_pose_regions_use_only_visible_keypoints():
    visible = {
        "nose": (10, 10),
        "left_shoulder": (8, 30),
        "right_shoulder": (20, 30),
        "left_hip": (9, 55),
        "right_hip": (19, 55),
        "left_ankle": (10, 95),
    }
    names = (
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
        "right_knee", "left_ankle", "right_ankle",
    )
    points = tuple(
        Keypoint(name, *visible[name], 0.9, True)
        if name in visible
        else Keypoint(name, None, None, 0.1, False)
        for name in names
    )
    regions = pose_evidence_regions(PoseResult("cam-1", 1, BoundingBox(0, 0, 30, 100), points))
    assert set(regions) == {
        EvidenceKind.HEAD,
        EvidenceKind.TORSO,
        EvidenceKind.FEET,
        EvidenceKind.UPPER_BODY,
    }
