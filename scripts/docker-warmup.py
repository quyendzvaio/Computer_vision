#!/usr/bin/env python3
"""Pre-warm CUDA + ONNX session before main app starts.

Called by entrypoint.sh on container start. Runs dummy inference
to initialize CUDA kernels and ONNX session options, reducing
first-frame latency from ~3s to ~0.3s.
"""
import numpy as np
from gpu.detector import YOLODetector


def warmup():
    print("[Warm-up] Loading YOLOv8n...")
    det = YOLODetector()
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    result = det.detect(dummy)
    print(f"[Warm-up] ONNX session ready (detections: {len(result)})")


if __name__ == "__main__":
    warmup()
