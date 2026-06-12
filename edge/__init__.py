"""Edge agent — ties together capture, processing, and publishing."""
from typing import Optional, Dict

import numpy as np

from edge.source_manager import SourceManager
from edge.frame_processor import FrameProcessor
from edge.mqtt_publisher import MQTTPublisher
from edge.local_bridge import LocalBridge


class EdgeAgent:
    """Orchestrates edge-side processing for all configured cameras."""

    def __init__(self, config: dict, local_bridge: Optional[LocalBridge] = None):
        self.config = config
        self.source_manager = SourceManager(config)
        self.processor = FrameProcessor(
            target_size=(
                config.get("frame", {}).get("resize_width", 416),
                config.get("frame", {}).get("resize_height", 416),
            ),
            motion_threshold=config.get("frame", {}).get("motion_threshold", 0.05),
            jpeg_quality=config.get("frame", {}).get("jpeg_quality", 70),
        )
        self.mqtt: Optional[MQTTPublisher] = None
        self.local_bridge = local_bridge

    def start_mqtt(self):
        """Connect to MQTT broker for remote cameras.
        Connection failure is logged, not raised — MQTT is optional."""
        try:
            self.mqtt = MQTTPublisher(self.config)
            self.mqtt.connect()
            self.mqtt.set_active_cameras(list(self.source_manager.cameras.keys()))
            print("[EdgeAgent] MQTT connected")
        except ConnectionError as e:
            print(f"[EdgeAgent] MQTT not available ({e}), remote cameras will not work")
            self.mqtt = None

    def start_all_cameras(self):
        """Start capturing from all configured cameras.
        Per-camera failures are logged but don't stop other cameras."""
        started = 0
        failed = 0
        for cam_id, cam in self.source_manager.cameras.items():
            is_local = cam.source.isdigit()
            handler = self._local_frame_handler if (is_local and self.local_bridge) else self._mqtt_frame_handler
            try:
                self.source_manager.start(cam_id, handler)
                started += 1
                print(f"[EdgeAgent] Camera {cam_id} started (local={is_local})")
            except Exception as e:
                failed += 1
                print(f"[EdgeAgent] Camera {cam_id} FAILED: {e}")
        print(f"[EdgeAgent] {started} started, {failed} failed, {len(self.source_manager.cameras)} total")

    def _local_frame_handler(self, camera_id: str, frame: np.ndarray):
        """Handle frame from local (USB) camera: push to LocalBridge queue."""
        cam = self.source_manager.cameras.get(camera_id)
        roi = cam.roi if cam else None
        jpeg_bytes = self.processor.process(frame, roi_polygon=roi)
        if jpeg_bytes and self.local_bridge is not None and self.local_bridge._loop is not None:
            import asyncio
            asyncio.run_coroutine_threadsafe(
                self.local_bridge.put_frame(camera_id, jpeg_bytes),
                self.local_bridge._loop,
            )

    def _mqtt_frame_handler(self, camera_id: str, frame: np.ndarray):
        """Handle frame from remote (RTSP) camera: publish via MQTT."""
        cam = self.source_manager.cameras.get(camera_id)
        roi = cam.roi if cam else None
        jpeg_bytes = self.processor.process(frame, roi_polygon=roi)
        if jpeg_bytes and self.mqtt:
            self.mqtt.publish_frame(camera_id, jpeg_bytes)

    def stop(self):
        """Stop all capture and disconnect."""
        self.source_manager.stop_all()
        if self.mqtt:
            self.mqtt.disconnect()
