"""Draw bounding boxes, ROI polygons, and alert labels on frames."""
import json
from typing import List

import cv2
import numpy as np

from shared.models import BBox, DetectedObject


COLORS = {
    "person": (0, 255, 0),
    "HELMET": (0, 255, 0),
    "NO_HELMET": (0, 0, 255),
    "VEST": (0, 255, 0),
    "NO_VEST": (0, 0, 255),
    "BOOT": (0, 255, 0),
    "NO_BOOT": (0, 0, 255),
    "zone": (255, 165, 0),
    "zone_alert": (0, 0, 255),
    "text": (255, 255, 255),
}


def draw_person_bboxes(frame: np.ndarray, persons: List[DetectedObject]) -> np.ndarray:
    """Draw green bounding boxes around detected persons."""
    canvas = frame.copy()
    for person in persons:
        b = person.bbox
        x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLORS["person"], 2)
        label = f"person {person.conf:.2f}"
        cv2.putText(canvas, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text"], 1)
    return canvas


def draw_roi_polygons(frame: np.ndarray, rois: List[dict]) -> np.ndarray:
    """Draw ROI polygons on frame. Alert zones in red, normal in orange."""
    canvas = frame.copy()
    for roi in rois:
        points = np.array(json.loads(roi["points_json"]), dtype=np.int32)
        color = COLORS["zone_alert"] if roi.get("alert_active") else COLORS["zone"]
        cv2.polylines(canvas, [points], isClosed=True, color=color, thickness=2)
        if len(points) > 0:
            cx = int(points[:, 0].mean())
            cy = int(points[:, 1].mean())
            cv2.putText(canvas, roi["zone_name"], (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return canvas


def draw_ppe_labels(frame: np.ndarray, persons: List[DetectedObject],
                    alerts: List[dict]) -> np.ndarray:
    """Draw PPE status per person. Red text for missing items."""
    canvas = frame.copy()
    alert_by_idx = {a["person_idx"]: a["violations"] for a in alerts}

    for idx, person in enumerate(persons):
        b = person.bbox
        x1, y1 = int(b.x1), int(b.y1)
        violations = alert_by_idx.get(idx, [])
        y_offset = y1 - 20
        for vtype in ["HELMET", "VEST", "BOOT"]:
            is_missing = f"NO_{vtype}" in violations
            label = f"{'✗' if is_missing else '✓'} {vtype}"
            color = COLORS[f"NO_{vtype}"] if is_missing else COLORS[vtype]
            cv2.putText(canvas, label, (x1, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset -= 15
    return canvas


def draw_disconnected(frame: np.ndarray) -> np.ndarray:
    """Overlay 'DISCONNECTED' text on frame."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    cv2.putText(canvas, "DISCONNECTED", (w // 2 - 100, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    return canvas


def draw_detection_offline(frame: np.ndarray) -> np.ndarray:
    """Overlay 'Detection Offline' badge."""
    canvas = frame.copy()
    cv2.putText(canvas, "Detection Offline", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return canvas
