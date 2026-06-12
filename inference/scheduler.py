"""Round-robin scheduler for multi-camera inference."""
import threading
from collections import deque
from typing import Dict, Optional, Tuple


class Scheduler:
    """Fair round-robin scheduler across multiple cameras.
    Each camera has its own frame queue; poll() returns the next
    (camera_id, jpeg_bytes) pair in round-robin order.
    Thread-safe: all public methods acquire a lock before accessing state."""

    def __init__(self):
        self._lock = threading.Lock()
        self._queues: Dict[str, deque] = {}
        self._order: list[str] = []
        self._index: int = 0

    def register_camera(self, camera_id: str):
        """Register a camera for scheduling."""
        with self._lock:
            if camera_id not in self._queues:
                self._queues[camera_id] = deque(maxlen=5)
                self._order.append(camera_id)

    def add_frame(self, camera_id: str, jpeg_bytes: bytes):
        """Add a frame to a camera's queue. Drops oldest if queue is full.
        Silently registers unknown cameras on first use."""
        with self._lock:
            if camera_id not in self._queues:
                self._queues[camera_id] = deque(maxlen=5)
                self._order.append(camera_id)
            self._queues[camera_id].append(jpeg_bytes)

    def poll(self) -> Optional[Tuple[str, bytes]]:
        """Get the next frame in round-robin order.
        Returns None if all queues are empty."""
        with self._lock:
            if not self._order:
                return None

            checked = 0
            while checked < len(self._order):
                cam_id = self._order[self._index]
                self._index = (self._index + 1) % len(self._order)

                if self._queues.get(cam_id) and len(self._queues[cam_id]) > 0:
                    frame = self._queues[cam_id].popleft()
                    return (cam_id, frame)

                checked += 1

            return None

    @property
    def camera_count(self) -> int:
        with self._lock:
            return len(self._order)
