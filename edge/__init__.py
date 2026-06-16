"""Edge agent — ties together capture, processing, and publishing.

Now delegates frame routing to CaptureDisplay (P1) for local cameras.
Remote (RTSP) cameras still handled via MQTT publisher.

Legacy: EdgeAgent class kept for backward compat with existing tests,
but new CaptureDisplay in capture_loop.py is the new P1 entry point.
"""
from typing import Optional

import numpy as np

from edge.source_manager import SourceManager
from edge.mqtt_publisher import MQTTPublisher


class EdgeAgent:
    """Orchestrates edge-side processing for all configured cameras."""

    def __init__(self, config: dict):
        self.config = config
        self.source_manager = SourceManager(config)
        self.mqtt: Optional[MQTTPublisher] = None

    def start_mqtt(self):
        """Connect to MQTT broker for remote cameras.
        Connection failure is logged, not raised — MQTT is optional."""
        try:
            self.mqtt = MQTTPublisher(self.config)
            self.mqtt.connect()
            remote_ids = [
                c["id"] for c in self.config.get("cameras", [])
                if not str(c.get("source", "")).isdigit()
            ]
            self.mqtt.set_active_cameras(remote_ids)
            print("[EdgeAgent] MQTT connected")
        except ConnectionError as e:
            print(f"[EdgeAgent] MQTT not available ({e}), remote cameras will not work")
            self.mqtt = None

    def start_remote_cameras(self):
        """Start only remote (non-USB) cameras — local cameras handled by CaptureDisplay."""
        started = 0
        failed = 0
        for cam_id, cam in self.source_manager.cameras.items():
            if cam.source.isdigit():
                continue  # Skip local — CaptureDisplay handles these
            try:
                self.source_manager.start(cam_id, self._mqtt_frame_handler)
                started += 1
                print(f"[EdgeAgent] Remote camera {cam_id} started")
            except Exception as e:
                failed += 1
                print(f"[EdgeAgent] Camera {cam_id} FAILED: {e}")
        print(f"[EdgeAgent] {started} remote started, {failed} failed")

    def _mqtt_frame_handler(self, camera_id: str, frame: np.ndarray):
        """Handle frame from remote (RTSP) camera: publish via MQTT."""
        if self.mqtt:
            from edge.frame_processor import FrameProcessor
            cfg = self.config
            proc = FrameProcessor(
                target_size=(
                    cfg.get("frame", {}).get("resize_width", 416),
                    cfg.get("frame", {}).get("resize_height", 416),
                ),
                motion_threshold=cfg.get("frame", {}).get("motion_threshold", 0.05),
                jpeg_quality=cfg.get("frame", {}).get("jpeg_quality", 70),
            )
            cam = self.source_manager.cameras.get(camera_id)
            jpeg_bytes = proc.process(frame, roi_polygon=cam.roi if cam else None)
            if jpeg_bytes:
                self.mqtt.publish_frame(camera_id, jpeg_bytes)

    def stop(self):
        """Stop all capture and disconnect."""
        self.source_manager.stop_all()
        if self.mqtt:
            self.mqtt.disconnect()
