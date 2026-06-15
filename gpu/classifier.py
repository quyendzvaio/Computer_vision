"""MobileNetV3-small ONNX classifiers for PPE detection.

Three binary classifiers: helmet, vest, boot. Each takes a 224×224
RGB crop and returns a label (yes/no) with confidence.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime


MODELS_DIR = Path(__file__).parent / "models"

CLASS_NAMES = {
    "helmet": ["NO_HELMET", "HELMET"],
    "vest": ["NO_VEST", "VEST"],
    "boot": ["NO_BOOT", "BOOT"],
}


class PPEClassifier:
    """Binary classifier for one PPE item (helmet/vest/boot)."""

    def __init__(self, item: str, model_path: Optional[str] = None,
                 providers: Optional[List[str]] = None):
        if item not in CLASS_NAMES:
            raise ValueError(f"Unknown PPE item: {item}. Choose from {list(CLASS_NAMES.keys())}")
        self.item = item
        self.class_names = CLASS_NAMES[item]
        if model_path is None:
            model_path = str(MODELS_DIR / f"{item}.onnx")
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, crop: np.ndarray) -> Tuple[str, float]:
        """Classify a BGR crop (224×224 or will be resized).

        Returns:
            Tuple of (label, confidence).
        """
        img = cv2.resize(crop, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 127.5 - 1.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        outputs = self.session.run(None, {self.input_name: img})
        probs = outputs[0][0]
        class_id = int(np.argmax(probs))
        return self.class_names[class_id], float(probs[class_id])


class PPEManager:
    """Manages all 3 PPE classifiers. Runs them on cropped body regions."""

    def __init__(self):
        self.helmet = PPEClassifier("helmet")
        self.vest = PPEClassifier("vest")
        self.boot = PPEClassifier("boot")

    def classify_all(self, head_crop: np.ndarray, torso_crop: np.ndarray,
                     feet_crop: np.ndarray) -> dict:
        """Run all 3 classifiers on respective body region crops.

        Returns:
            dict: {item: {"label": str, "confidence": float}}
        """
        return {
            "helmet": dict(zip(["label", "confidence"], self.helmet.predict(head_crop))),
            "vest": dict(zip(["label", "confidence"], self.vest.predict(torso_crop))),
            "boot": dict(zip(["label", "confidence"], self.boot.predict(feet_crop))),
        }
