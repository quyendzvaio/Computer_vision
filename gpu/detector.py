"""YOLOv8n ONNX detector wrapper.

Runs inference on raw BGR frame (640×480), returns list of DetectedObject
with 'person' class only (COCO class 0). NMS applied post-inference.
"""
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import onnxruntime

from shared.models import BBox, DetectedObject


MODEL_PATH = Path(__file__).parent / "models" / "yolov8n.onnx"
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
COCO_PERSON_ID = 0


class YOLODetector:
    """YOLOv8n ONNX detector. detects only 'person' class."""

    def __init__(self, model_path: str = str(MODEL_PATH),
                 providers: Optional[List[str]] = None):
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        _, _, self.input_h, self.input_w = self.session.get_inputs()[0].shape

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to model input, normalize, return NCHW tensor."""
        img = cv2.resize(frame, (self.input_w, self.input_h))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = np.expand_dims(img, axis=0)    # → NCHW
        return img

    def postprocess(self, outputs: np.ndarray, orig_shape) -> List[DetectedObject]:
        """Parse YOLOv8 output into DetectedObject list."""
        output = outputs[0]
        output = np.transpose(output, (1, 0))  # (num_boxes, 84)

        boxes, scores = [], []
        for pred in output:
            cls_scores = pred[4:]
            class_id = int(np.argmax(cls_scores))
            score = float(cls_scores[class_id])
            if class_id != COCO_PERSON_ID or score < CONF_THRESHOLD:
                continue

            xc, yc, w, h = pred[0], pred[1], pred[2], pred[3]
            x1 = (xc - w / 2) / self.input_w * orig_shape[1]
            y1 = (yc - h / 2) / self.input_h * orig_shape[0]
            x2 = (xc + w / 2) / self.input_w * orig_shape[1]
            y2 = (yc + h / 2) / self.input_h * orig_shape[0]

            boxes.append([x1, y1, x2, y2])
            scores.append(score)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD)
        results = []
        for i in indices.flatten():
            b = boxes[i]
            results.append(DetectedObject(
                bbox=BBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3]),
                cls="person",
                conf=scores[i],
            ))
        return results

    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        """Run detection on a BGR frame. Returns list of person DetectedObject."""
        orig_h, orig_w = frame.shape[:2]
        input_tensor = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return self.postprocess(outputs, (orig_h, orig_w))
