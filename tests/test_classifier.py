import pytest
from shared.models import DetectionResult, DetectedObject, BBox, Keypoint


def make_result(camera_id, objects=None, keypoints=None):
    return DetectionResult(
        camera_id=camera_id,
        objects=objects or [],
        keypoints=keypoints,
    )


def test_no_person_no_violations():
    """If no person detected, all checks should return empty."""
    from alert.classifier import ViolationClassifier

    vc = ViolationClassifier()
    result = make_result("cam-01", objects=[
        DetectedObject(bbox=BBox(10, 10, 50, 50), cls="helmet", conf=0.8),
    ])
    violations = vc.classify(result)
    assert len(violations) == 0


def test_person_with_helmet_no_violation():
    """Person with overlapping helmet - no NO_HELMET violation."""
    from alert.classifier import ViolationClassifier

    vc = ViolationClassifier()
    result = make_result("cam-01", objects=[
        DetectedObject(bbox=BBox(100, 100, 200, 300), cls="person", conf=0.9),
        DetectedObject(bbox=BBox(110, 100, 180, 160), cls="helmet", conf=0.7),
    ])
    violations = vc.classify(result)
    helmet_violations = [v for v in violations if v.type == "NO_HELMET"]
    assert len(helmet_violations) == 0


def test_person_without_helmet_violation():
    """Person without helmet overlap - NO_HELMET violation."""
    from alert.classifier import ViolationClassifier

    vc = ViolationClassifier()
    result = make_result("cam-01", objects=[
        DetectedObject(bbox=BBox(100, 100, 200, 300), cls="person", conf=0.9),
        # helmet is far away from person
        DetectedObject(bbox=BBox(400, 400, 450, 450), cls="helmet", conf=0.7),
    ])
    violations = vc.classify(result)
    helmet_violations = [v for v in violations if v.type == "NO_HELMET"]
    assert len(helmet_violations) == 1
    assert helmet_violations[0].severity == "HIGH"


def test_person_with_vest_no_violation():
    """Person with overlapping vest - no NO_VEST violation."""
    from alert.classifier import ViolationClassifier

    vc = ViolationClassifier()
    result = make_result("cam-01", objects=[
        DetectedObject(bbox=BBox(100, 100, 200, 300), cls="person", conf=0.9),
        DetectedObject(bbox=BBox(110, 150, 190, 250), cls="vest", conf=0.8),
    ])
    violations = vc.classify(result)
    vest_violations = [v for v in violations if v.type == "NO_VEST"]
    assert len(vest_violations) == 0


def test_person_without_vest_violation():
    """Person without vest overlap - NO_VEST violation."""
    from alert.classifier import ViolationClassifier

    vc = ViolationClassifier()
    result = make_result("cam-01", objects=[
        DetectedObject(bbox=BBox(100, 100, 200, 300), cls="person", conf=0.9),
    ])
    violations = vc.classify(result)
    vest_violations = [v for v in violations if v.type == "NO_VEST"]
    assert len(vest_violations) == 1
    assert vest_violations[0].severity == "MEDIUM"


def test_person_with_boot_no_violation():
    """Person with overlapping boot in lower third - no NO_BOOT violation."""
    from alert.classifier import ViolationClassifier

    vc = ViolationClassifier()
    # Person: y1=100, y2=300, height=200, lower_third starts at 100+200*0.66=232
    result = make_result("cam-01", objects=[
        DetectedObject(bbox=BBox(100, 100, 200, 300), cls="person", conf=0.9),
        # boot in lower third of person
        DetectedObject(bbox=BBox(120, 250, 180, 290), cls="boot", conf=0.8),
    ])
    violations = vc.classify(result)
    boot_violations = [v for v in violations if v.type == "NO_BOOT"]
    assert len(boot_violations) == 0


def test_person_without_boot_violation():
    """Person without boot in lower third - NO_BOOT violation."""
    from alert.classifier import ViolationClassifier

    vc = ViolationClassifier()
    result = make_result("cam-01", objects=[
        DetectedObject(bbox=BBox(100, 100, 200, 300), cls="person", conf=0.9),
        # boot is present but in upper part of person body (not lower 1/3)
        DetectedObject(bbox=BBox(120, 110, 180, 140), cls="boot", conf=0.8),
    ])
    violations = vc.classify(result)
    boot_violations = [v for v in violations if v.type == "NO_BOOT"]
    assert len(boot_violations) == 1
    assert boot_violations[0].severity == "MEDIUM"


def test_fall_detected_by_aspect_ratio():
    """Person with wide aspect ratio (w/h > 1.2) and head below hips -> FALL."""
    from alert.classifier import ViolationClassifier

    vc = ViolationClassifier()
    # Person lying down: wide bbox (w=200, h=50, aspect_ratio=4.0 > 1.2)
    # Head keypoints have high y (lower in image), hip keypoints have low y
    # aspect_ratio 4.0 gives +0.4, head-below-hip gives +0.35 -> total 0.75 > 0.6
    result = make_result("cam-01", objects=[
        DetectedObject(bbox=BBox(100, 200, 300, 250), cls="person", conf=0.9),
    ], keypoints=[
        [
            # 17 COCO keypoints (x, y, conf) for a fallen person
            # Head points: high y (~265) = low in image = head below hips
            Keypoint(200, 265, 0.9),  # 0: nose
            Keypoint(200, 263, 0.9),  # 1: left_eye
            Keypoint(210, 263, 0.9),  # 2: right_eye
            Keypoint(195, 268, 0.9),  # 3: left_ear
            Keypoint(215, 268, 0.9),  # 4: right_ear
            # Shoulders: mid y (~235)
            Keypoint(180, 235, 0.9),  # 5: left_shoulder
            Keypoint(220, 235, 0.9),  # 6: right_shoulder
            # Elbows
            Keypoint(170, 250, 0.9),  # 7: left_elbow
            Keypoint(230, 250, 0.9),  # 8: right_elbow
            # Wrists
            Keypoint(160, 265, 0.9),  # 9: left_wrist
            Keypoint(240, 265, 0.9),  # 10: right_wrist
            # Hips: low y (~215) = high in image = hips above head
            Keypoint(145, 215, 0.9),  # 11: left_hip
            Keypoint(195, 215, 0.9),  # 12: right_hip
            # Knees
            Keypoint(140, 235, 0.9),  # 13: left_knee
            Keypoint(200, 235, 0.9),  # 14: right_knee
            # Ankles
            Keypoint(135, 255, 0.9),  # 15: left_ankle
            Keypoint(205, 255, 0.9),  # 16: right_ankle
        ]
    ])
    violations = vc.classify(result)
    fall_violations = [v for v in violations if v.type == "FALL"]
    # w/h = 200/50 = 4.0 > 1.2 -> +0.4
    # head_y avg = (265+263+263+268+268)/5 = 265.4
    # hip_y avg = (215+215)/2 = 215
    # 265.4 > 215 -> head below hip -> +0.35
    # shoulder_y avg = (235+235)/2 = 235 > 215 hip -> +0.25
    # Total: 0.4 + 0.35 + 0.25 = 1.0 (capped) > 0.6
    assert len(fall_violations) == 1
    assert fall_violations[0].severity == "HIGH"


def test_fall_not_detected_when_standing():
    """Standing person (narrow bbox, head above hips) should NOT trigger FALL."""
    from alert.classifier import ViolationClassifier

    vc = ViolationClassifier()
    result = make_result("cam-01", objects=[
        DetectedObject(bbox=BBox(100, 100, 200, 350), cls="person", conf=0.9),
    ], keypoints=[
        [
            Keypoint(150, 120, 0.9),  # 0: nose (high up)
            Keypoint(140, 115, 0.9),  # 1: left_eye
            Keypoint(160, 115, 0.9),  # 2: right_eye
            Keypoint(135, 125, 0.9),  # 3: left_ear
            Keypoint(165, 125, 0.9),  # 4: right_ear
            Keypoint(120, 160, 0.9),  # 5: left_shoulder
            Keypoint(180, 160, 0.9),  # 6: right_shoulder
            Keypoint(110, 220, 0.9),  # 7: left_elbow
            Keypoint(190, 220, 0.9),  # 8: right_elbow
            Keypoint(100, 280, 0.9),  # 9: left_wrist
            Keypoint(200, 280, 0.9),  # 10: right_wrist
            Keypoint(130, 200, 0.9),  # 11: left_hip
            Keypoint(170, 200, 0.9),  # 12: right_hip
            Keypoint(120, 270, 0.9),  # 13: left_knee
            Keypoint(180, 270, 0.9),  # 14: right_knee
            Keypoint(110, 340, 0.9),  # 15: left_ankle
            Keypoint(190, 340, 0.9),  # 16: right_ankle
        ]
    ])
    violations = vc.classify(result)
    fall_violations = [v for v in violations if v.type == "FALL"]
    # Standing: w/h = 100/250 = 0.4, aspect not > 0.9, no aspect score
    # head_y avg ≈ 120, hip_y avg = 200 -> head above hip -> no keypoint score
    assert len(fall_violations) == 0


def test_multiple_violations_same_person():
    """A person can have multiple violations (no helmet + no vest)."""
    from alert.classifier import ViolationClassifier

    vc = ViolationClassifier()
    result = make_result("cam-01", objects=[
        DetectedObject(bbox=BBox(100, 100, 200, 300), cls="person", conf=0.9),
    ])
    violations = vc.classify(result)
    types = {v.type for v in violations}
    assert "NO_HELMET" in types
    assert "NO_VEST" in types


def test_low_confidence_objects_ignored():
    """Objects below confidence threshold should be ignored."""
    from alert.classifier import ViolationClassifier

    vc = ViolationClassifier(confidence_threshold=0.5)
    result = make_result("cam-01", objects=[
        DetectedObject(bbox=BBox(100, 100, 200, 300), cls="person", conf=0.3),
    ])
    violations = vc.classify(result)
    # Low-confidence person should be ignored entirely
    assert len([v for v in violations if v.type in ("NO_HELMET", "NO_VEST")]) == 0
