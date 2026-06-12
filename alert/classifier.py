"""Violation classifier — converts DetectionResult into a list of Violation objects."""
from typing import List

from shared.models import DetectionResult, DetectedObject, Violation, BBox, Keypoint


def _iou(a: BBox, b: BBox) -> float:
    """Intersection over Union between two bounding boxes."""
    x_left = max(a.x1, b.x1)
    y_top = max(a.y1, b.y1)
    x_right = min(a.x2, b.x2)
    y_bottom = min(a.y2, b.y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    inter = (x_right - x_left) * (y_bottom - y_top)
    area_a = a.width * a.height
    area_b = b.width * b.height
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _compute_fall_score(
    person_box: BBox, keypoints: List[Keypoint]
) -> float:
    """Compute a fall score (0-1) from bbox aspect ratio and keypoint geometry.

    Score > 0.6 triggers FALL alert.
    """
    score = 0.0

    # Factor 1: aspect ratio (wider than tall -> likely fallen)
    if person_box.aspect_ratio > 1.2:
        score += 0.4
    elif person_box.aspect_ratio > 0.9:
        score += 0.2

    # Factor 2: keypoint geometry (head below hip center)
    if keypoints and len(keypoints) >= 17:
        # Head points: nose(0), eyes(1,2), ears(3,4)
        head_ys = [kp.y for i, kp in enumerate(keypoints)
                   if i in (0, 1, 2, 3, 4) and kp.conf > 0.3]
        # Hip points: left_hip(11), right_hip(12)
        hip_ys = [kp.y for i, kp in enumerate(keypoints)
                  if i in (11, 12) and kp.conf > 0.3]

        if head_ys and hip_ys:
            avg_head_y = sum(head_ys) / len(head_ys)
            avg_hip_y = sum(hip_ys) / len(hip_ys)

            # head below or near hip -> likely fallen
            if avg_head_y > avg_hip_y:
                score += 0.35
            elif avg_head_y > avg_hip_y - 30:
                score += 0.15

        # Factor 3: shoulder center vs hip center
        shoulder_ys = [kp.y for i, kp in enumerate(keypoints)
                       if i in (5, 6) and kp.conf > 0.3]
        if shoulder_ys and hip_ys:
            avg_shoulder_y = sum(shoulder_ys) / len(shoulder_ys)
            avg_hip_y = sum(hip_ys) / len(hip_ys)
            if avg_shoulder_y > avg_hip_y:
                score += 0.25

    return min(score, 1.0)


class ViolationClassifier:
    """Classifies DetectionResult into Violation objects for the 4 safety types."""

    IOU_THRESHOLD = 0.1  # Minimum IoU to consider equipment "on" the person

    def __init__(self, confidence_threshold: float = 0.4):
        self.confidence_threshold = confidence_threshold

    def classify(self, result: DetectionResult) -> List[Violation]:
        """Classify all violations in a single detection result."""
        violations: List[Violation] = []

        persons = [o for o in result.objects
                   if o.cls == "person" and o.conf >= self.confidence_threshold]
        helmets = [o for o in result.objects
                   if o.cls == "helmet" and o.conf >= self.confidence_threshold]
        vests = [o for o in result.objects
                 if o.cls == "vest" and o.conf >= self.confidence_threshold]
        boots = [o for o in result.objects
                 if o.cls == "boot" and o.conf >= self.confidence_threshold]

        for i, person in enumerate(persons):
            person_keypoints = None
            if result.keypoints and i < len(result.keypoints):
                person_keypoints = result.keypoints[i]

            # NO_HELMET check
            has_helmet = any(
                _iou(person.bbox, h.bbox) > self.IOU_THRESHOLD
                for h in helmets
            )
            if not has_helmet:
                violations.append(Violation(
                    camera_id=result.camera_id,
                    type="NO_HELMET",
                    severity="HIGH",
                    bbox=person.bbox,
                    timestamp=result.timestamp,
                ))

            # NO_VEST check
            has_vest = any(
                _iou(person.bbox, v.bbox) > self.IOU_THRESHOLD
                for v in vests
            )
            if not has_vest:
                violations.append(Violation(
                    camera_id=result.camera_id,
                    type="NO_VEST",
                    severity="MEDIUM",
                    bbox=person.bbox,
                    timestamp=result.timestamp,
                ))

            # NO_BOOT check — only consider boot detections in lower 1/3 of person box
            person_lower_y = person.bbox.y1 + person.bbox.height * 0.66
            boots_in_lower = [
                b for b in boots
                if b.bbox.y1 >= person_lower_y
                and _iou(person.bbox, b.bbox) > self.IOU_THRESHOLD
            ]
            has_boot = len(boots_in_lower) > 0
            if not has_boot:
                violations.append(Violation(
                    camera_id=result.camera_id,
                    type="NO_BOOT",
                    severity="MEDIUM",
                    bbox=person.bbox,
                    timestamp=result.timestamp,
                ))

            # FALL check (requires keypoints)
            if person_keypoints:
                fall_score = _compute_fall_score(person.bbox, person_keypoints)
                if fall_score > 0.6:
                    violations.append(Violation(
                        camera_id=result.camera_id,
                        type="FALL",
                        severity="HIGH",
                        bbox=person.bbox,
                        timestamp=result.timestamp,
                    ))

        return violations
