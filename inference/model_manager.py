"""ModelManager — loads ONNX model and runs CPU inference.

ROI-first optimization: caller crops frame to ROI region BEFORE calling
detect(), so the model sees fewer pixels = faster inference.

Architecture decisions (by user request):
- onnxruntime CPU only (no OpenVINO, no TensorRT)
- Async-friendly: load() is a sync init, but run() is re-entrant
  (callers use threads or asyncio executors to avoid blocking the loop)
"""
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class Detection:
    """One detected object from raw model output."""
    __slots__ = ('bbox', 'cls_id', 'cls_name', 'conf')

    def __init__(self, bbox: List[float], cls_id: int, cls_name: str, conf: float):
        self.bbox = bbox
        self.cls_id = cls_id
        self.cls_name = cls_name
        self.conf = conf

    def __repr__(self):
        return f"Detection({self.cls_name} {self.conf:.2f} {self.bbox})"


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

PERSON_CLASS_ID = 0
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.45
INPUT_SIZE = (416, 416)  # YOLOv8n default input


class ModelManager:
    """ONNX model manager with ROI-first CPU inference."""

    def __init__(self,
                 model_path: str = "models/yolov8n.onnx",
                 input_size: Tuple[int, int] = INPUT_SIZE,
                 conf_threshold: float = CONF_THRESHOLD,
                 nms_threshold: float = IOU_THRESHOLD):
        self.model_path = Path(model_path)
        self.input_w, self.input_h = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self._session = None

    def load(self):
        if ort is None:
            raise RuntimeError("onnxruntime not installed: pip install onnxruntime")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        thread_count = os.cpu_count() or 4
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, thread_count - 1)
        opts.inter_op_num_threads = 2
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        self._session = ort.InferenceSession(
            str(self.model_path),
            providers=['CPUExecutionProvider'],
            sess_options=opts,
        )
        self._warmup()

    @property
    def input_size(self) -> Tuple[int, int]:
        return (self.input_w, self.input_h)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize BGR frame → NCHW float32 tensor (1,3,H,W) normalized [0,1]."""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_w, self.input_h),
                         interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        return np.expand_dims(img, axis=0)

    def preprocess_jpeg(self, jpeg_bytes: bytes) -> Optional[np.ndarray]:
        """Decode JPEG bytes and preprocess. Returns None on failure."""
        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        if frame is None:
            return None
        return self.preprocess(frame)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a preprocessed NCHW tensor. Returns list of Detection."""
        input_tensor = frame if frame.ndim == 4 else self.preprocess(frame)
        outputs = self._session.run(None, {self._session.get_inputs()[0].name: input_tensor})
        return self._postprocess(outputs, frame.shape[:2] if frame.ndim == 3 else None)

    def _postprocess(self, outputs, orig_shape: Optional[Tuple[int, int]] = None) -> List[Detection]:
        """YOLOv8 postprocessing: NMS, filter by class/confidence."""
        preds = outputs[0][0]
        boxes, scores, cls_ids = [], [], []
        for pred in preds.T:
            scores_arr = pred[4:]
            cls_id = int(np.argmax(scores_arr))
            score = float(scores_arr[cls_id])
            if score < self.conf_threshold:
                continue
            xc, yc, w, h = pred[0], pred[1], pred[2], pred[3]
            if orig_shape:
                orig_h, orig_w = orig_shape
                sw, sh = orig_w / self.input_w, orig_h / self.input_h
            else:
                sw = sh = 1.0
            x1 = (xc - w / 2) * sw
            y1 = (yc - h / 2) * sh
            x2 = (xc + w / 2) * sw
            y2 = (yc + h / 2) * sh
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            cls_ids.append(cls_id)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.nms_threshold)
        results = []
        for i in indices.flatten():
            c = cls_ids[i] if i < len(cls_ids) else 0
            results.append(Detection(
                bbox=boxes[i],
                cls_id=c,
                cls_name=COCO_CLASSES.get(c, "?"),
                conf=scores[i],
            ))
        return results

    def _warmup(self):
        """CPU kernel warmup — runs once after model load."""
        dummy = np.zeros((self.input_h, self.input_w, 3), dtype=np.uint8)
        self.detect(dummy)
        print("[ModelManager] Warm-up complete")
