"""MQTT publisher — sends processed frames to the inference server."""
import json
import time
import threading
from typing import Optional

import paho.mqtt.client as mqtt


class MQTTPublisher:
    """Publishes JPEG frame bytes and heartbeat messages to MQTT broker."""

    def __init__(self, config: dict):
        mqtt_cfg = config.get("mqtt", {})
        self._broker = mqtt_cfg.get("broker", "localhost")
        self._port = mqtt_cfg.get("port", 1883)
        self._client_id = mqtt_cfg.get("client_id", "edge-agent-01")
        self._connected = False
        self._client: Optional[mqtt.Client] = None

        topics = config.get("topics", {})
        self._frame_topic_template = topics.get("frame", "cv/{camera_id}/frame")
        self._heartbeat_topic_template = topics.get("heartbeat", "cv/{camera_id}/heartbeat")

        self._heartbeat_interval = 10.0
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._running = False
        self._active_cameras: list = []

    def connect(self):
        """Connect to MQTT broker and start heartbeat thread."""
        self._client = mqtt.Client(client_id=self._client_id)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

        self._client.connect_async(self._broker, self._port, keepalive=30)
        self._client.loop_start()

        timeout = 5
        start = time.time()
        while not self._connected and (time.time() - start) < timeout:
            time.sleep(0.1)

        if not self._connected:
            raise ConnectionError(f"Failed to connect to MQTT broker at {self._broker}:{self._port}")

        self._running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self._heartbeat_thread.start()

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
        else:
            print(f"[MQTT] Connection failed with code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False

    def publish_frame(self, camera_id: str, jpeg_bytes: bytes):
        """Publish a JPEG frame to the camera's MQTT topic."""
        if not self._client or not self._connected:
            return
        topic = self._frame_topic_template.format(camera_id=camera_id)
        self._client.publish(topic, jpeg_bytes, qos=1)

    def _heartbeat_loop(self):
        while self._running:
            for camera_id in self._active_cameras:
                if not self._client or not self._connected:
                    break
                topic = self._heartbeat_topic_template.format(camera_id=camera_id)
                payload = json.dumps({"status": "alive", "timestamp": time.time()})
                self._client.publish(topic, payload, qos=0)
            time.sleep(self._heartbeat_interval)

    def set_active_cameras(self, camera_ids: list):
        self._active_cameras = camera_ids

    def disconnect(self):
        """Disconnect from MQTT broker."""
        self._running = False
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
