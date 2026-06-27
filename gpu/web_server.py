"""FastAPI web server thread for secondary viewing (history, admin, 1fps preview).

Runs in a QThread with uvicorn.Server for clean lifecycle.
"""
import asyncio
import base64
import json
import threading
from pathlib import Path
from queue import Queue, Empty
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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
_preview_event = threading.Event()


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
        _preview_event.set()


class WebServer(QThread):
    """FastAPI web server running in its own QThread with asyncio event loop."""

    def __init__(self, db_conn, alert_manager, host: str = "0.0.0.0", port: int = 8080):
        super().__init__()
        self._host = host
        self._port = port
        self._db = db_conn
        self._alert_manager = alert_manager
        self.on_roi_saved = None  # callback: on_roi_saved(camera_id)

    def run(self):
        app = FastAPI(title="CV Safety Monitor v2")

        static_dir = Path(__file__).parent.parent / "dashboard" / "static"
        if static_dir.exists():
            app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

            @app.get("/")
            async def root():
                return FileResponse(str(static_dir / "index.html"))

            @app.get("/admin.html")
            async def admin_page():
                return FileResponse(str(static_dir / "admin.html"))

            @app.get("/history.html")
            async def history_page():
                return FileResponse(str(static_dir / "history.html"))
        else:
            @app.get("/")
            async def root():
                return {
                    "app": "CV Safety Monitor v2",
                    "version": "0.1.0",
                    "endpoints": {
                        "cameras": "/api/cameras",
                        "roi": "/api/roi/{camera_id}",
                        "violations": "/api/violations",
                        "websocket": "/ws/dashboard",
                        "docs": "/docs",
                    }
                }

        @app.on_event("startup")
        def startup():
            app.state.db = self._db

        @app.get("/api/cameras")
        async def list_cameras():
            return get_cameras(self._db)

        @app.get("/api/roi/{camera_id}")
        async def get_roi_api(camera_id: str):
            rois = get_rois(self._db, camera_id)
            if not rois:
                return {"polygon": []}
            # Return first enabled ROI's polygon as flat coords
            points = json.loads(rois[0]["points_json"])
            return {"polygon": points}

        @app.put("/api/roi/{camera_id}")
        async def save_roi_api(camera_id: str, data: dict):
            polygon = data.get("polygon", data.get("points", []))
            save_roi(self._db, camera_id, "default-zone",
                     polygon, data.get("color", "#ff0000"))
            # Reload ROI into the running camera thread
            cb = self.on_roi_saved
            if cb:
                cb(camera_id)
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
            loop = asyncio.get_running_loop()
            try:
                while True:
                    # Wait for preview event in thread-safe way
                    await loop.run_in_executor(None, _preview_event.wait)
                    _preview_event.clear()
                    # Drain queue
                    while True:
                        try:
                            preview = preview_queue.get_nowait()
                            await ws.send_json({
                                "type": "preview",
                                "camera_id": preview["camera_id"],
                                "frame_base64": preview["frame_base64"],
                            })
                        except Empty:
                            break
            except WebSocketDisconnect:
                ws_manager.disconnect(ws)

        config = uvicorn.Config(app, host=self._host, port=self._port, log_level="warning")
        server = uvicorn.Server(config)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())
