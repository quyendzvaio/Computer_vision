"""Detector — wraps ModelManager to produce DetectionResult objects."""
from datetime import datetime
from typing import Optional

import numpy as np

from inference.model_manager import ModelManager
from shared.models import DetectionResult, DetectedObject, BBox


class Detector:
    """Converts raw model detections into structured DetectionResult objects.
    Handles both object detection and pose estimation (via separate model)."""

    def __init__(self, model_manager: ModelManager):
        self.mm = model_manager

    def run(self, jpeg_bytes: bytes, camera_id: str) -> DetectionResult:
        """Run detection on a JPEG frame and return structured result."""
        frame = self.mm.preprocess_jpeg(jpeg_bytes)
        if frame is None:
            return DetectionResult(camera_id=camera_id)

        detections = self.mm.detect(frame)

        objects = [
            DetectedObject(
                bbox=BBox(d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]),
                cls=d.cls_name,
                conf=d.conf,
            )
            for d in detections
        ]

        return DetectionResult(
            camera_id=camera_id,
            objects=objects,
            keypoints=None,  # Pose model integration: Task 17
            timestamp=datetime.now(),
        )
