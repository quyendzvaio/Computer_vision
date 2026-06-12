"""Local receiver — reads frames from LocalBridge queues for USB cameras."""
import asyncio
from typing import Callable, Dict

from edge.local_bridge import LocalBridge


class LocalReceiver:
    """Consumes frames from LocalBridge queues and feeds them to the scheduler.
    Each camera gets its own asyncio task."""

    def __init__(self, local_bridge: LocalBridge):
        self._bridge = local_bridge
        self._tasks: Dict[str, asyncio.Task] = {}
        self._running = False

    async def start(
        self,
        camera_ids: list,
        on_frame: Callable[[str, bytes], None],
    ):
        """Start receiving frames for the given camera IDs.
        on_frame(camera_id, jpeg_bytes) is called per frame."""
        self._running = True
        for cid in camera_ids:
            task = asyncio.create_task(
                self._receive_loop(cid, on_frame),
                name=f"local-rx-{cid}",
            )
            self._tasks[cid] = task

    async def _receive_loop(self, camera_id: str, on_frame: Callable):
        """Continuously read from the local bridge queue."""
        while self._running:
            try:
                jpeg_bytes = await self._bridge.get_frame(camera_id, timeout=1.0)
                on_frame(camera_id, jpeg_bytes)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    async def stop(self):
        """Stop all receiver tasks."""
        self._running = False
        for task in self._tasks.values():
            task.cancel()
        self._tasks.clear()
