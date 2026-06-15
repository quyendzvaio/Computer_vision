"""PPE (Personal Protective Equipment) checker.

Crops body regions from person bbox, runs through MobileNetV3
classifiers to detect helmet/vest/boot presence.
"""
from typing import List

import numpy as np

from shared.models import DetectedObject


HEAD_RATIO = (0.0, 0.2)          # top 20% of bbox
TORSO_RATIO = (0.2, 0.7)        # 20%-70% of bbox
FEET_RATIO = (0.85, 1.0)        # bottom 15% of bbox


def crop_head(frame: np.ndarray, bbox) -> np.ndarray:
    """Crop head region from person bbox. Returns RGB-ready crop."""
    x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
    h = y2 - y1
    cy1 = max(0, y1)
    cy2 = min(frame.shape[0], y1 + int(h * HEAD_RATIO[1]))
    cx1 = max(0, x1)
    cx2 = min(frame.shape[1], x2)
    if cy2 <= cy1 or cx2 <= cx1:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    return frame[cy1:cy2, cx1:cx2]


def crop_torso(frame: np.ndarray, bbox) -> np.ndarray:
    """Crop torso region from person bbox."""
    x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
    h = y2 - y1
    cy1 = max(0, y1 + int(h * TORSO_RATIO[0]))
    cy2 = min(frame.shape[0], y1 + int(h * TORSO_RATIO[1]))
    cx1 = max(0, x1)
    cx2 = min(frame.shape[1], x2)
    if cy2 <= cy1 or cx2 <= cx1:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    return frame[cy1:cy2, cx1:cx2]


def crop_feet(frame: np.ndarray, bbox) -> np.ndarray:
    """Crop feet region from person bbox."""
    x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
    h = y2 - y1
    cy1 = max(0, y1 + int(h * FEET_RATIO[0]))
    cy2 = min(frame.shape[0], y2)
    cx1 = max(0, x1)
    cx2 = min(frame.shape[1], x2)
    if cy2 <= cy1 or cx2 <= cx1:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    return frame[cy1:cy2, cx1:cx2]


class PPEChecker:
    """Coordinate crop + classify for PPE detection per person."""

    def __init__(self, ppe_manager):
        self._ppe = ppe_manager
        self._frame_counter = 0
        self._classify_every_n = 3

    def process_persons(self, frame: np.ndarray,
                        persons: List[DetectedObject],
                        force_classify: bool = False) -> List[dict]:
        """Run PPE check on all detected persons in frame."""
        self._frame_counter += 1
        results = []
        should_classify = force_classify or (self._frame_counter % self._classify_every_n == 0)

        for idx, person in enumerate(persons):
            if should_classify:
                head = crop_head(frame, person.bbox)
                torso = crop_torso(frame, person.bbox)
                feet = crop_feet(frame, person.bbox)

                ppe_result = self._ppe.classify_all(head, torso, feet)

                violations = []
                for item, result in ppe_result.items():
                    if result["label"].startswith("NO_"):
                        violations.append(result["label"])

                if violations:
                    results.append({
                        "person_idx": idx,
                        "bbox": person.bbox,
                        "violations": violations,
                        "ppe": ppe_result,
                    })

        return results
