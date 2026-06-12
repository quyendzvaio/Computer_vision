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
        """Connect to MQTT broker for remote cameras."""
        self.mqtt = MQTTPublisher(self.config)
        self.mqtt.connect()
        self.mqtt.set_active_cameras(list(self.source_manager.cameras.keys()))

    def start_all_cameras(self):
        """Start capturing from all configured cameras."""
        for cam_id, cam in self.source_manager.cameras.items():
            is_local = isinstance(cam.source, int) or cam.source in ("0", "1", "2", "3")
            if is_local and self.local_bridge:
                self.source_manager.start(cam_id, self._local_frame_handler)
            else:
                self.source_manager.start(cam_id, self._mqtt_frame_handler)

    def _local_frame_handler(self, camera_id: str, frame: np.ndarray):
        """Handle frame from local (USB) camera: push to LocalBridge queue."""
        cam = self.source_manager.cameras.get(camera_id)
        roi = cam.roi if cam else None
        jpeg_bytes = self.processor.process(frame, roi_polygon=roi)
        if jpeg_bytes and self.local_bridge:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(
                        self.local_bridge.put_frame(camera_id, jpeg_bytes)
                    )
            except RuntimeError:
                pass

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
