"""Local bridge — passes frames directly via asyncio.Queue when edge == server."""
import asyncio
from typing import Dict


class LocalBridge:
    """When a USB camera is plugged directly into the server machine,
    frames are passed through an in-process asyncio.Queue instead of MQTT."""

    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = {}

    def get_queue(self, camera_id: str, maxsize: int = 10) -> asyncio.Queue:
        """Get or create a queue for a camera."""
        if camera_id not in self._queues:
            self._queues[camera_id] = asyncio.Queue(maxsize=maxsize)
        return self._queues[camera_id]

    async def put_frame(self, camera_id: str, jpeg_bytes: bytes):
        """Put a frame into the camera's queue (non-blocking, drops oldest if full)."""
        queue = self.get_queue(camera_id)
        if queue.full():
            try:
                queue.get_nowait()  # Drop oldest
            except asyncio.QueueEmpty:
                pass
        await queue.put(jpeg_bytes)

    async def get_frame(self, camera_id: str, timeout: float = 1.0) -> bytes:
        """Get the next frame from the camera's queue."""
        queue = self.get_queue(camera_id)
        return await asyncio.wait_for(queue.get(), timeout=timeout)
