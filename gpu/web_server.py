"""FastAPI web server thread for secondary viewing (history, admin, 1fps preview).

Runs in a QThread with uvicorn.Server for clean lifecycle.
"""
import asyncio
import base64
import threading
from queue import Queue
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PyQt5.QtCore import QThread
import uvicorn

from gpu.database import get_cameras, get_rois, get_violations, save_roi


class ConnectionManager:
    """WebSocket connection manager for alert broadcasts."""

    def __init__(self):
        self._connections: list[WebSocket] = []
        self._lock = threading.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        with self._lock:
            self._connections.append(ws)

    def disconnect(self, ws: WebSocket):
        with self._lock:
            if ws in self._connections:
                self._connections.remove(ws)

    async def broadcast(self, message: dict):
        dead = []
        with self._lock:
            for ws in self._connections:
                try:
                    await ws.send_json(message)
                except Exception:
                    dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


ws_manager = ConnectionManager()
preview_queue: Queue = Queue(maxsize=2)


def push_preview(camera_id: str, frame_bgr: np.ndarray):
    """Push a preview frame for WebSocket broadcast (called from main thread)."""
    if preview_queue.full():
        try:
            preview_queue.get_nowait()
        except Exception:
            pass
    ret, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 50])
    if ret:
        b64 = base64.b64encode(buf.tobytes()).decode()
        preview_queue.put_nowait({"camera_id": camera_id, "frame_base64": b64})


class WebServer(QThread):
    """FastAPI web server running in its own QThread with asyncio event loop."""

    def __init__(self, db_conn, alert_manager, host: str = "0.0.0.0", port: int = 8080):
        super().__init__()
        self._host = host
        self._port = port
        self._db = db_conn
        self._alert_manager = alert_manager

    def run(self):
        app = FastAPI(title="CV Safety Monitor v2")

        @app.on_event("startup")
        def startup():
            app.state.db = self._db

        @app.get("/api/cameras")
        async def list_cameras():
            return {"cameras": get_cameras(self._db)}

        @app.get("/api/roi/{camera_id}")
        async def get_roi_api(camera_id: str):
            return {"rois": get_rois(self._db, camera_id)}

        @app.put("/api/roi/{camera_id}")
        async def save_roi_api(camera_id: str, data: dict):
            save_roi(self._db, camera_id, data["zone_name"],
                     data["points"], data.get("color", "#ff0000"))
            return {"status": "ok"}

        @app.get("/api/violations")
        async def list_violations(limit: int = 50, offset: int = 0, camera_id: str = ""):
            return {
                "violations": get_violations(
                    self._db, limit=limit, offset=offset,
                    camera_id=camera_id or None,
                )
            }

        @app.websocket("/ws/dashboard")
        async def dashboard_ws(ws: WebSocket):
            await ws_manager.connect(ws)
            try:
                while True:
                    if not preview_queue.empty():
                        try:
                            preview = preview_queue.get_nowait()
                            await ws.send_json({
                                "type": "preview",
                                "camera_id": preview["camera_id"],
                                "frame_base64": preview["frame_base64"],
                            })
                        except Exception:
                            pass
                    await asyncio.sleep(1)
            except WebSocketDisconnect:
                ws_manager.disconnect(ws)

        config = uvicorn.Config(app, host=self._host, port=self._port, log_level="warning")
        server = uvicorn.Server(config)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())
