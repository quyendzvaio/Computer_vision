"""FastAPI dashboard server: REST API + WebSocket + static file serving."""
import asyncio
import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from alert.db import get_violations, get_roi, save_roi, get_all_rois

STATIC_DIR = str(Path(__file__).parent / "static")
THUMBNAIL_DIR = str(Path("data/thumbnails"))


# --- WebSocket Connection Manager ---

class ConnectionManager:
    """Manages active WebSocket connections for realtime broadcast."""

    def __init__(self):
        self._connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self._connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self._connections:
            self._connections.remove(websocket)

    def broadcast(self, message: str):
        """Schedule broadcast to all connected clients.

        Safe to call from sync context (e.g., Dispatcher).
        In sync context, schedules via asyncio.create_task if a loop is running.
        """
        try:
            loop = asyncio.get_running_loop()
            for ws in self._connections[:]:
                loop.create_task(self._safe_send(ws, message))
        except RuntimeError:
            # No running event loop -- tests or app not started
            pass

    async def broadcast_async(self, message: str):
        """Broadcast from within an async context."""
        for ws in self._connections[:]:
            await self._safe_send(ws, message)

    async def _safe_send(self, ws: WebSocket, message: str):
        try:
            await ws.send_text(message)
        except Exception:
            self.disconnect(ws)


ws_manager = ConnectionManager()
_roi_matcher = None


# --- Pydantic Schemas ---

class ROIPayload(BaseModel):
    polygon: List[List[float]]


# --- FastAPI App ---

app = FastAPI(title="CV Safety Monitor", version="0.1.0")

# Mount static files
static_path = Path(STATIC_DIR)
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


# --- REST API ---

@app.get("/api/cameras")
async def api_get_cameras():
    """Return list of cameras from edge config."""
    cameras = []
    config_path = Path("edge/config.yaml")
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        for cam in config.get("cameras", []):
            cameras.append({
                "id": cam["id"],
                "source": str(cam["source"]),
            })
    return cameras


@app.get("/api/roi/{camera_id}")
async def api_get_roi(camera_id: str):
    """Get ROI polygon for a camera."""
    row = get_roi(camera_id)
    if row is None:
        raise HTTPException(status_code=404, detail="ROI not found for this camera")
    return {
        "camera_id": row["camera_id"],
        "polygon": json.loads(row["polygon"]),
        "updated_at": row["updated_at"],
    }


@app.put("/api/roi/{camera_id}")
async def api_put_roi(camera_id: str, payload: ROIPayload):
    """Save or update ROI polygon for a camera."""
    polygon = [(p[0], p[1]) for p in payload.polygon]
    save_roi(camera_id, polygon)
    if _roi_matcher is not None:
        _roi_matcher.invalidate(camera_id)
    return {"status": "ok", "camera_id": camera_id}


@app.get("/api/violations")
async def api_get_violations(
    camera_id: Optional[str] = Query(None),
    type: Optional[str] = Query(None, alias="type"),
    from_time: Optional[str] = Query(None),
    to_time: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Query violation history with optional filters."""
    rows = get_violations(
        camera_id=camera_id,
        violation_type=type,
        from_time=from_time,
        to_time=to_time,
        limit=limit,
        offset=offset,
    )
    return rows


@app.get("/api/violations/{violation_id}/thumbnail")
async def api_get_thumbnail(violation_id: int):
    """Serve thumbnail image for a violation."""
    rows = get_violations(limit=1000)
    match = None
    for r in rows:
        if r["id"] == violation_id:
            match = r
            break
    if match is None or not match.get("thumbnail_path"):
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    path = Path(match["thumbnail_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail file missing")

    return FileResponse(str(path), media_type="image/jpeg")


# --- WebSocket ---

@app.websocket("/ws/dashboard")
async def ws_dashboard(websocket: WebSocket):
    """Realtime dashboard WebSocket endpoint."""
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


# --- Static HTML pages ---

@app.get("/")
async def root():
    index_path = Path(STATIC_DIR) / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "CV Safety Monitor API", "docs": "/docs"}


@app.get("/admin.html")
async def admin_page():
    return FileResponse(str(Path(STATIC_DIR) / "admin.html"))


@app.get("/history.html")
async def history_page():
    return FileResponse(str(Path(STATIC_DIR) / "history.html"))
