"""Inference worker (Process 2) — runs in a separate process.

Double-buffer protocol:
  1. Find FILLED buffer → LOCK
  2. Read frame, detect
  3. Write result + mark which buffer result is for
  4. P1 reads THAT buffer for exact overlay
  5. P1 releases buffer to FREE
"""
import json
import time
import threading
from datetime import datetime
from typing import List

from inference.model_manager import ModelManager, Detection
from shared.memory import FrameBuffer
from shared.models import DetectionResult, DetectedObject, BBox


class InferenceWorker:
    """Poll loop for a single camera's FrameBuffer."""

    def __init__(
        self,
        buffer: FrameBuffer,
        model: ModelManager,
        target_fps: float = 5.0,
        stop_event: threading.Event = None,
    ):
        self._buffer = buffer
        self._model = model
        self._interval = 1.0 / target_fps if target_fps > 0 else 0.2
        self._stop = stop_event or threading.Event()
        self._last_frame_seq = 0

    def step(self) -> bool:
        """Find FILLED buffer → lock → detect → write result."""
        buf_idx = self._buffer.find_filled()
        if buf_idx is None:
            return False

        # Lock buffer so P1 won't overwrite
        self._buffer.acquire(buf_idx)

        frame = self._buffer.read_frame(buf_idx)
        if frame is None:
            self._buffer.release(buf_idx)
            return False

        # Compute scale for bbox (model-space → frame-space)
        in_h, in_w = frame.shape[:2]
        model_w, model_h = self._model.input_size
        scale_x = in_w / model_w
        scale_y = in_h / model_h

        try:
            detections = self._model.detect(frame)
        except Exception as exc:
            print(f"[InferenceWorker] detect failed: {exc}")
            detections = []

        result = self._build_result(detections, scale_x, scale_y)
        # Write result AND flag which buffer this frame lives in
        self._buffer.write_result(json.dumps(result, default=str), buf_idx)

        # Rate limit before next poll
        time.sleep(self._interval)
        return True

    def _build_result(self, detections: List[Detection],
                       scale_x: float = 1.0, scale_y: float = 1.0) -> dict:
        objects = []
        for d in detections:
            objects.append({
                "cls": d.cls_name,
                "conf": d.conf,
                "bbox": {
                    "x1": float(d.bbox[0]) * scale_x,
                    "y1": float(d.bbox[1]) * scale_y,
                    "x2": float(d.bbox[2]) * scale_x,
                    "y2": float(d.bbox[3]) * scale_y,
                },
            })
        return {
            "objects": objects,
            "timestamp": datetime.now().isoformat(),
        }

    def run_forever(self):
        """Main loop — poll until stop."""
        while not self._stop.is_set():
            self.step()
            if not self._stop.is_set():
                self._stop.wait(0.01)


def inference_worker(config: dict, stop_event: threading.Event):
    """Entry point for P2 process."""
    local_camera_ids = []
    for cam in config.get("cameras", []):
        source = str(cam.get("source", ""))
        if source.isdigit():
            local_camera_ids.append(cam["id"])

    if not local_camera_ids:
        print("[InferenceWorker] No local cameras — nothing to do")
        return

    target_fps = config.get("frame", {}).get("inference_fps", 5.0)
    max_w = config.get("frame", {}).get("max_width", 1920)
    max_h = config.get("frame", {}).get("max_height", 1080)

    print("[InferenceWorker] Loading model...")
    mm = ModelManager(
        model_path=config.get("model", {}).get("path", "models/yolov8n.onnx"),
        input_size=(
            config.get("frame", {}).get("resize_width", 416),
            config.get("frame", {}).get("resize_height", 416),
        ),
        conf_threshold=config.get("model", {}).get("conf_threshold", 0.4),
        nms_threshold=config.get("model", {}).get("nms_threshold", 0.45),
    )
    try:
        mm.load()
    except Exception as e:
        print(f"[InferenceWorker] Model load failed: {e}")
        return

    workers = []
    buffers = []
    for cid in local_camera_ids:
        buf = FrameBuffer(cid, max_width=max_w, max_height=max_h)
        buffers.append(buf)
        worker = InferenceWorker(buf, mm, target_fps=target_fps, stop_event=stop_event)
        workers.append(worker)

    print(f"[InferenceWorker] Running {len(workers)} camera(s) at {target_fps} FPS")

    try:
        threads = []
        for w in workers:
            t = threading.Thread(
                target=w.run_forever, daemon=True,
                name=f"inf-{w._buffer.camera_id}")
            t.start()
            threads.append(t)

        while not stop_event.is_set():
            stop_event.wait(1)
    finally:
        for b in buffers:
            b.close()
        print("[InferenceWorker] Stopped")
