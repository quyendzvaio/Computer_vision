"""Detector — wraps ModelManager to produce DetectionResult objects."""
from datetime import datetime

from inference.model_manager import ModelManager
from shared.models import DetectionResult, DetectedObject, BBox


class Detector:
    """Converts raw model detections into structured DetectionResult objects."""

    def __init__(self, model_manager: ModelManager):
        self.mm = model_manager

    def run(self, jpeg_bytes: bytes, camera_id: str) -> DetectionResult:
        """Run detection on a JPEG frame and return structured result."""
        try:
            frame = self.mm.preprocess_jpeg(jpeg_bytes)
        except Exception:
            return DetectionResult(camera_id=camera_id)

        if frame is None:
            return DetectionResult(camera_id=camera_id)

        try:
            detections = self.mm.detect(frame)
        except Exception:
            return DetectionResult(camera_id=camera_id)

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
            keypoints=None,
            timestamp=datetime.now(),
        )
