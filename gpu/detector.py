"""YOLOv8n ONNX detector wrapper.

Runs inference on raw BGR frame (640×480), returns list of DetectedObject
with 'person' class only (COCO class 0). NMS applied post-inference.
"""
from pathlib import Path
from typing import List, Optional, Tuple

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
            available = onnxruntime.get_available_providers()
            providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in available]
        sess_opts = onnxruntime.SessionOptions()
        sess_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = onnxruntime.InferenceSession(model_path, providers=providers, sess_options=sess_opts)
        self.input_name = self.session.get_inputs()[0].name
        _, _, self.input_h, self.input_w = self.session.get_inputs()[0].shape
        self._warmup()

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to model input, normalize, return NCHW tensor."""
        img = cv2.resize(frame, (self.input_w, self.input_h))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = np.expand_dims(img, axis=0)    # → NCHW
        return img

    def postprocess(self, outputs: np.ndarray, orig_shape) -> List[DetectedObject]:
        """Parse YOLOv8 output into DetectedObject list.

        Handles shape variants:
          - (1, 84, 8400)  — ultralytics standard export
          - (1, 8400, 84)  — some export variants
          - (8400, 84)     — already squeezed
        """
        output = outputs[0]
        # Squeeze batch dim if present
        if output.ndim == 3:
            output = output[0] if output.shape[0] == 1 else output.squeeze()
        # Normalize to (num_boxes, 84) via transpose if needed
        if output.shape[0] == 84 and output.shape[1] != 84:
            output = output.T  # (84, 8400) → (8400, 84)

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

    def _warmup(self):
        """CUDA kernel warmup — dummy inference to compile kernels.
        Cuts first-frame latency from ~3s to ~0.3s."""
        dummy = np.zeros((self.input_h, self.input_w, 3), dtype=np.uint8)
        self.detect(dummy)

    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        """Run detection on a BGR frame. Returns list of person DetectedObject."""
        orig_h, orig_w = frame.shape[:2]
        input_tensor = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return self.postprocess(outputs, (orig_h, orig_w))

    def detect_roi(self, frame: np.ndarray,
                   roi_bounds: Optional[Tuple[int, int, int, int]]
                   ) -> List[DetectedObject]:
        """ROI-first detection: crop to ROI → detect → offset bboxes to original frame.

        Args:
            frame: Full BGR frame.
            roi_bounds: (x, y, w, h) of ROI crop, or None for full-frame fallback.

        Returns:
            List of DetectedObject with bboxes in original frame coordinates.
        """
        if roi_bounds is None:
            return self.detect(frame)

        rx, ry, rw, rh = roi_bounds
        if rw < 32 or rh < 32:  # ponytail: too small to detect anything useful
            return []

        crop = frame[ry:ry + rh, rx:rx + rw]
        objects = self.detect(crop)
        for obj in objects:
            obj.bbox.x1 += rx
            obj.bbox.y1 += ry
            obj.bbox.x2 += rx
            obj.bbox.y2 += ry
        return objects
