"""Camera source manager — opens and manages RTSP/USB capture devices."""
import time
import threading
from typing import Dict, Optional, Callable
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraSource:
    """Configuration for one camera source."""
    id: str
    source: str    # File path, RTSP URL, or USB device index (as string)
    roi: list = None  # [(x,y), ...] polygon


class SourceManager:
    """Manages multiple camera capture sources. Each source runs in its own thread,
    calling a callback with every captured frame."""

    def __init__(self, config: dict):
        self.cameras: Dict[str, CameraSource] = {}
        self._captures: Dict[str, cv2.VideoCapture] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._running: Dict[str, bool] = {}
        self._callbacks: Dict[str, Callable] = {}
        self._fps_interval: float = 1.0 / config.get("frame", {}).get("target_fps", 5)

        for cam_cfg in config.get("cameras", []):
            cam = CameraSource(
                id=cam_cfg["id"],
                source=str(cam_cfg["source"]),
                roi=cam_cfg.get("roi"),
            )
            self.cameras[cam.id] = cam

    def _parse_source(self, source: str):
        """Parse source: if numeric string, treat as int (USB index)."""
        try:
            return int(source)
        except ValueError:
            return source

    def start(self, camera_id: str, on_frame: Callable[[str, np.ndarray], None]):
        """Start capturing from a camera. on_frame(camera_id, bgr_frame) called per frame."""
        if camera_id not in self.cameras:
            raise ValueError(f"Unknown camera: {camera_id}")

        if camera_id in self._running and self._running[camera_id]:
            return  # Already running

        cam = self.cameras[camera_id]
        src = self._parse_source(cam.source)
        cap = cv2.VideoCapture(src)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera source: {cam.source}")

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._captures[camera_id] = cap
        self._callbacks[camera_id] = on_frame
        self._running[camera_id] = True

        thread = threading.Thread(
            target=self._capture_loop,
            args=(camera_id,),
            daemon=True,
            name=f"cam-{camera_id}",
        )
        self._threads[camera_id] = thread
        thread.start()

    def _capture_loop(self, camera_id: str):
        cap = self._captures[camera_id]
        callback = self._callbacks[camera_id]
        interval = self._fps_interval

        while self._running.get(camera_id, False):
            start = time.time()

            ret, frame = cap.read()
            if not ret:
                time.sleep(1)
                cap.release()
                cam = self.cameras[camera_id]
                src = self._parse_source(cam.source)
                cap = cv2.VideoCapture(src)
                self._captures[camera_id] = cap
                continue

            try:
                callback(camera_id, frame)
            except Exception:
                pass

            elapsed = time.time() - start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self, camera_id: str):
        """Stop capturing from a camera."""
        self._running[camera_id] = False
        if camera_id in self._captures:
            self._captures[camera_id].release()
            del self._captures[camera_id]

    def stop_all(self):
        """Stop all cameras."""
        for cid in list(self._running.keys()):
            self.stop(cid)

    def is_running(self, camera_id: str) -> bool:
        return self._running.get(camera_id, False)
