"""Capture + Display + Alert loop (Process 1).

Double-buffer overlay flow:
  1. Check P2 result → if new, get _buffer_index from JSON
  2. Read exact frame from THAT buffer (still LOCKED by P2, untouched)
  3. Draw overlay on exact frame → FrameProcessor → WebSocket
  4. Release result buffer to FREE
  5. Pick FREE buffer → write current frame → P2 reads this
"""
import asyncio
import json
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np

from edge.frame_processor import FrameProcessor
from edge.source_manager import SourceManager
from shared.memory import FrameBuffer
from shared.models import DetectionResult, DetectedObject, BBox


# ── Bbox drawing ──────────────────────────────────────────────

_COLORS = {
    "person":    (0,   255, 0),
    "helmet":    (255, 0,   0),
    "vest":      (0,   0,   255),
    "boot":      (255, 255, 0),
}
_FALLBACK_COLOR = (0, 255, 255)


def _draw_overlay(frame: np.ndarray, objects: List[dict]) -> np.ndarray:
    canvas = frame.copy()
    for obj in objects:
        b = obj.get("bbox", {})
        x1 = int(b.get("x1", 0))
        y1 = int(b.get("y1", 0))
        x2 = int(b.get("x2", 0))
        y2 = int(b.get("y2", 0))
        cls = obj.get("cls", "?")
        conf = obj.get("conf", 0)
        color = _COLORS.get(cls, _FALLBACK_COLOR)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{cls} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return canvas


# ── JSON messages ────────────────────────────────────────────

def _preview_msg(camera_id: str, jpeg_b64: str) -> str:
    return json.dumps({"type": "preview", "camera_id": camera_id, "frame_base64": jpeg_b64})


def _violation_msg(v) -> str:
    return json.dumps({"type": "violation", "violation": {
        "id": v.id, "camera_id": v.camera_id, "type": v.type,
        "severity": v.severity, "bbox": v.bbox.to_list(),
        "thumbnail_path": v.thumbnail_path,
        "timestamp": v.timestamp.isoformat(),
    }}, default=str)


# ── DetectionResult parser ────────────────────────────────────

def _parse_result_json(camera_id: str, data: dict) -> Optional[DetectionResult]:
    """Build DetectionResult from result dict (not raw JSON)."""
    objects = []
    for o in data.get("objects", []):
        b = o.get("bbox", {})
        objects.append(DetectedObject(
            bbox=BBox(
                x1=float(b.get("x1", 0)),
                y1=float(b.get("y1", 0)),
                x2=float(b.get("x2", 0)),
                y2=float(b.get("y2", 0)),
            ),
            cls=o.get("cls", "?"),
            conf=float(o.get("conf", 0)),
        ))

    if not objects:
        return None

    ts = data.get("timestamp")
    return DetectionResult(
        camera_id=camera_id,
        objects=objects,
        timestamp=datetime.fromisoformat(ts) if ts else datetime.now(),
    )


# ── Main class ────────────────────────────────────────────────

class CaptureDisplay:
    """P1: capture + double-buffer IPC + overlay + alert + WebSocket."""

    def __init__(
        self,
        config: dict,
        stop_event: threading.Event,
        loop: asyncio.AbstractEventLoop,
        alert_pipeline,
        ws_manager,
        buffers: Dict[str, FrameBuffer],
    ):
        self._config = config
        self._stop = stop_event
        self._loop = loop
        self._alert_pipeline = alert_pipeline
        self._ws_manager = ws_manager
        self._buffers = buffers

        self._processor = FrameProcessor(
            target_size=(
                config.get("frame", {}).get("resize_width", 416),
                config.get("frame", {}).get("resize_height", 416),
            ),
            motion_threshold=config.get("frame", {}).get("motion_threshold", 0.05),
            jpeg_quality=config.get("frame", {}).get("jpeg_quality", 70),
        )

        self._source_manager = SourceManager(config)

    # ── Helper: fire & forget async WS send ───────────────────

    def _send_ws(self, msg: str):
        asyncio.run_coroutine_threadsafe(
            self._ws_manager.broadcast_async(msg), self._loop,
        )

    # ── Per-frame handler ─────────────────────────────────────

    def _handle_frame(self, camera_id: str, frame: np.ndarray):
        """Called from capture thread.

        Protocol per frame:
          1. Write frame to FREE buffer (P2 reads this)
          2. Check P2 result → if new, overlay on EXACT frame P2 saw
          3. Always send preview (no overlay if no result, keeps dashboard live)
        """
        buf = self._buffers[camera_id]

        # ---- 1. Write current frame to a FREE buffer ----
        free_idx = buf.pick_free()
        if free_idx is not None:
            buf.write_frame(free_idx, frame)

        # ---- 2. Check P2 result ----
        result_payload = buf.read_result()

        if result_payload is not None:
            # We have detection — overlay on EXACT frame P2 processed
            result_buf = result_payload.get("_buffer_index", -1)
            if result_buf in (0, 1):
                exact_frame = buf.read_frame(result_buf)
                if exact_frame is not None:
                    objects = result_payload.get("objects", [])
                    overlay_frame = _draw_overlay(exact_frame, objects) if objects else exact_frame

                    # Encode preview with overlay
                    jpeg_bytes = self._processor.process(overlay_frame)
                    if jpeg_bytes is not None:
                        import base64
                        b64 = base64.b64encode(jpeg_bytes).decode()
                        self._send_ws(_preview_msg(camera_id, b64))

                    # Alert pipeline
                    detection_result = _parse_result_json(camera_id, result_payload)
                    if detection_result:
                        try:
                            violations = self._alert_pipeline.process(
                                detection_result, frame_bgr=overlay_frame)
                            for v in violations:
                                self._send_ws(_violation_msg(v))
                        except Exception as exc:
                            print(f"[CaptureDisplay] Alert error: {exc}")

                # Release result buffer back to FREE
                buf.release(result_buf)
            else:
                # Shouldn't happen, but code path safety
                pass
        else:
            # ---- 3. No result — send bare preview (keeps dashboard live) ----
            jpeg_bytes = self._processor.process(frame)
            if jpeg_bytes is not None:
                import base64
                b64 = base64.b64encode(jpeg_bytes).decode()
                self._send_ws(_preview_msg(camera_id, b64))

    # ── Lifecycle ─────────────────────────────────────────────

    def run(self):
        for cam_id in self._source_manager.cameras:
            try:
                self._source_manager.start(cam_id, self._handle_frame)
            except Exception as exc:
                print(f"[CaptureDisplay] Camera {cam_id} FAILED: {exc}")

        running = sum(1 for c in self._source_manager.cameras
                      if self._source_manager.is_running(c))
        total = len(self._source_manager.cameras)
        print(f"[CaptureDisplay] {running}/{total} cameras running")

        while not self._stop.is_set():
            self._stop.wait(1)

        self._source_manager.stop_all()
        print("[CaptureDisplay] Stopped")
