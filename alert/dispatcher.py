"""Dispatcher: persists violation to DB, saves thumbnail, broadcasts to dashboard."""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from shared.models import Violation


class Dispatcher:
    """Handles the final step of the alert pipeline:
    1. Save thumbnail to disk
    2. INSERT violation into SQLite
    3. Broadcast to all WebSocket dashboard clients
    """

    def __init__(self, db, ws_manager=None, thumbnail_dir: str = "data/thumbnails"):
        self._db = db
        self._ws_manager = ws_manager
        self._thumbnail_dir = Path(thumbnail_dir)
        self._thumbnail_dir.mkdir(parents=True, exist_ok=True)

    def dispatch(self, violation: Violation, frame_bgr: Optional[np.ndarray] = None) -> None:
        """Process and persist a violation, then broadcast to dashboard."""
        # 1. Save thumbnail
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{violation.camera_id}_{violation.type}_{ts}.jpg"
        thumbnail_path = str(self._thumbnail_dir / filename)

        if frame_bgr is not None:
            cv2.imwrite(thumbnail_path, frame_bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 75])
        violation.thumbnail_path = thumbnail_path

        # 2. Insert into DB
        vid = self._db.insert_violation(
            violation.camera_id,
            violation.type,
            violation.severity,
            violation.bbox.to_list(),
            thumbnail_path,
        )
        violation.id = vid

        # 3. Broadcast to dashboard via WebSocket
        if self._ws_manager is not None:
            message = json.dumps({
                "type": "violation",
                "violation": {
                    "id": violation.id,
                    "camera_id": violation.camera_id,
                    "type": violation.type,
                    "severity": violation.severity,
                    "bbox": violation.bbox.to_list(),
                    "thumbnail_path": violation.thumbnail_path,
                    "timestamp": violation.timestamp.isoformat(),
                },
            }, default=str)
            self._ws_manager.broadcast(message)
