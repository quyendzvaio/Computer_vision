"""MQTT subscriber — receives frames from edge agents for inference."""
import threading
from typing import Callable, Optional

import paho.mqtt.client as mqtt


class MQTTSubscriber:
    """Subscribes to MQTT topics for incoming camera frames."""

    def __init__(
        self,
        broker: str = "localhost",
        port: int = 1883,
        frame_topic_pattern: str = "cv/+/frame",
    ):
        self._broker = broker
        self._port = port
        self._frame_topic_pattern = frame_topic_pattern
        self._client: Optional[mqtt.Client] = None
        self._connected: threading.Event = threading.Event()
        self._on_frame: Optional[Callable[[str, bytes], None]] = None

    def connect(self, on_frame: Callable[[str, bytes], None]):
        """Connect to MQTT broker and set frame callback.
        on_frame(camera_id: str, jpeg_bytes: bytes)

        Note: on_frame is called from paho's network thread, not the asyncio event loop."""
        self._on_frame = on_frame
        self._connected.clear()
        self._client = mqtt.Client(client_id="inference-server")
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

        try:
            self._client.connect_async(self._broker, self._port, keepalive=30)
        except Exception:
            raise ConnectionError(
                f"Failed to connect to MQTT broker at {self._broker}:{self._port}"
            )

        self._client.loop_start()

        if not self._connected.wait(timeout=5.0):
            self._client.loop_stop()
            self._client.disconnect()
            raise ConnectionError(
                f"Failed to connect to MQTT broker at {self._broker}:{self._port}"
            )

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected.set()
            client.subscribe(self._frame_topic_pattern, qos=1)
        else:
            print(f"[MQTT Sub] Connection failed with code {rc}")

    def _on_message(self, client, userdata, msg):
        """Parse camera_id from topic and forward frame."""
        # Topic format: cv/{camera_id}/frame
        parts = msg.topic.split('/')
        if len(parts) >= 3:
            camera_id = parts[1]
            if self._on_frame:
                self._on_frame(camera_id, msg.payload)

    def disconnect(self):
        """Disconnect from MQTT broker."""
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
