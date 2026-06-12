"""OpenVINO model manager — loads YOLOv8 ONNX→IR models and runs inference."""
import time
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

import numpy as np
import cv2


@dataclass
class Detection:
    """Single YOLO detection."""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    cls: int
    cls_name: str
    conf: float



class ModelManager:
    """Manages YOLO model loading and inference.
    Falls back to ONNX Runtime if OpenVINO is not available."""

    CLASS_NAMES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
        'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
        'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
        'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush',
    ]

    def __init__(
        self,
        model_path: str = "models/yolov8n.onnx",
        input_size: Tuple[int, int] = (416, 416),
        conf_threshold: float = 0.4,
        nms_threshold: float = 0.45,
    ):
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        self._session = None
        self._use_openvino = False

    def load(self):
        """Load the model. Tries OpenVINO first, falls back to ONNX Runtime."""
        try:
            import openvino.runtime as ov
        except ImportError:
            print("[ModelManager] OpenVINO not available, using ONNX Runtime fallback")
            self._load_onnx()
            self._warmup()
            return

        try:
            core = ov.Core()
            ir_path = self.model_path.with_suffix('.xml')
            if ir_path.exists():
                model = core.read_model(str(ir_path))
                self._session = core.compile_model(model, "CPU")
                self._use_openvino = True
                print(f"[ModelManager] Loaded OpenVINO IR model from {ir_path}")
            elif self.model_path.suffix == '.onnx' and self.model_path.exists():
                model = core.read_model(str(self.model_path))
                self._session = core.compile_model(model, "CPU")
                self._use_openvino = True
                print(f"[ModelManager] Loaded ONNX model via OpenVINO from {self.model_path}")
            else:
                raise FileNotFoundError(f"Model not found: {self.model_path}")
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[ModelManager] OpenVINO failed ({e}), falling back to ONNX Runtime")
            self._load_onnx()

        self._warmup()

    def _load_onnx(self):
        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError(
                "Neither OpenVINO nor ONNX Runtime is available. "
                "Install onnxruntime: pip install onnxruntime"
            )
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self._session = ort.InferenceSession(
            str(self.model_path),
            providers=['CPUExecutionProvider'],
        )

    def _warmup(self):
        """Run a dummy inference to warm up the model."""
        dummy = np.random.randn(1, 3, *self.input_size).astype(np.float32)
        if self._use_openvino:
            infer_request = self._session.create_infer_request()
            infer_request.infer([dummy])
        else:
            input_name = self._session.get_inputs()[0].name
            self._session.run(None, {input_name: dummy})
        print("[ModelManager] Warm-up complete")

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Preprocess a BGR frame for YOLO inference.
        Returns (1, 3, H, W) float32 tensor normalized to [0,1]."""
        img = cv2.resize(frame_bgr, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)  # HWC → CHW
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)

    def inference(self, tensor: np.ndarray) -> np.ndarray:
        """Run inference on a preprocessed tensor. Returns raw model output."""
        if self._session is None:
            raise RuntimeError("Model not loaded. Call load() before inference().")
        if self._use_openvino:
            result = self._session([tensor])
            return result[0] if isinstance(result, (list, tuple)) else result
        else:
            input_name = self._session.get_inputs()[0].name
            return self._session.run(None, {input_name: tensor})[0]

    def postprocess(self, output: np.ndarray) -> List[Detection]:
        """Convert YOLOv8 raw output to Detection objects with per-class NMS."""
        if output.ndim == 3:
            output = output[0]
        output = output.transpose(1, 0)  # (8400, 84)

        boxes = output[:, :4]
        scores = output[:, 4:]

        detections: List[Detection] = []
        img_w, img_h = self.input_size

        for cls_id in range(scores.shape[1]):
            cls_scores = scores[:, cls_id]
            mask = cls_scores > self.conf_threshold
            if not mask.any():
                continue

            cls_boxes = boxes[mask]
            cls_confs = cls_scores[mask]

            cls_name = self.CLASS_NAMES[cls_id] if cls_id < len(self.CLASS_NAMES) else str(cls_id)
            cls_detections = []
            for box, conf in zip(cls_boxes, cls_confs):
                cx, cy, w, h = box
                x1 = (cx - w / 2) * img_w
                y1 = (cy - h / 2) * img_h
                x2 = (cx + w / 2) * img_w
                y2 = (cy + h / 2) * img_h

                cls_detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    cls=cls_id,
                    cls_name=cls_name,
                    conf=float(conf),
                ))

            detections.extend(self._nms(cls_detections))

        return detections

    def _nms(self, detections: List[Detection]) -> List[Detection]:
        """Apply Non-Maximum Suppression."""
        if not detections:
            return []

        boxes = np.array([d.bbox for d in detections])
        scores = np.array([d.conf for d in detections])

        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(),
            self.conf_threshold, self.nms_threshold,
        )

        if len(indices) == 0:
            return []

        return [detections[i] for i in indices.flatten()]

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Full pipeline: preprocess → inference → postprocess."""
        if self._session is None:
            raise RuntimeError("Model not loaded. Call load() before detect().")
        tensor = self.preprocess(frame_bgr)
        output = self.inference(tensor)
        return self.postprocess(output)

    def preprocess_jpeg(self, jpeg_bytes: bytes) -> np.ndarray:
        """Decode JPEG bytes to BGR frame."""
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    @property
    def is_loaded(self) -> bool:
        return self._session is not None
