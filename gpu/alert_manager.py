"""Alert manager: cooldown dedup, SQLite logging, WebSocket broadcast."""
import os
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from shared.models import BBox


class CooldownManager:
    """Tracks cooldown per (zone_name, person_idx, violation_type) tuple."""

    def __init__(self, default_cooldown: float = 30.0):
        self._default = default_cooldown
        self._timers: Dict[Tuple[str, int, str], float] = {}

    def can_alert(self, key: Tuple[str, int, str]) -> bool:
        """Check if enough time has passed since last alert for this key.

        Returns:
            True if should fire alert, False if still in cooldown.
        """
        now = time.time()
        last = self._timers.get(key, 0.0)
        if now - last >= self._default:
            self._timers[key] = now
            return True
        return False


class AlertManager:
    """Manages alert lifecycle: cooldown -> SQLite -> callback."""

    def __init__(self, conn: sqlite3.Connection,
                 on_alert: Optional[Callable[..., None]] = None,
                 cooldown: float = 30.0,
                 thumbnail_dir: str = "data/thumbnails"):
        self._conn = conn
        self._callback = on_alert
        self._cooldown = CooldownManager(default_cooldown=cooldown)
        self._thumbnail_dir = thumbnail_dir

    def process_violations(self, alerts: List[dict], camera_id: str,
                           frame: Optional[np.ndarray] = None,
                           force: bool = False) -> List[str]:
        """Process alerts through cooldown, save to DB, invoke callback.

        Returns:
            List of violation IDs that were actually fired.
        """
        from gpu.database import save_violation

        fired_ids = []
        for alert in alerts:
            vtype = alert.get("type", "PERSON_IN_ZONE")
            zone = alert.get("zone_name", "")
            person_idx = alert.get("person_idx", 0)
            bbox = alert.get("bbox")

            key = (zone or camera_id, person_idx, vtype)
            if not force and not self._cooldown.can_alert(key):
                continue

            thumbnail_path = ""
            if frame is not None and bbox is not None:
                thumbnail_path = self._save_thumbnail(frame, bbox, vtype)

            bbox_json = bbox.to_list() if bbox else ""

            sev = "HIGH" if vtype in ("PERSON_IN_ZONE",) else "MEDIUM"
            vid = save_violation(self._conn, camera_id, vtype, sev,
                                 zone_name=zone, bbox_json=str(bbox_json),
                                 thumbnail_path=thumbnail_path)

            if self._callback:
                self._callback(vid, camera_id, vtype, zone, person_idx, timestamp=datetime.now())

            fired_ids.append(vid)

        return fired_ids

    def process_zone_alerts(self, camera_id: str, alerts: List[dict],
                            frame: Optional[np.ndarray] = None,
                            force: bool = False) -> List[str]:
        """Process ROI zone alerts."""
        return self.process_violations(alerts, camera_id, frame, force)

    def _save_thumbnail(self, frame: np.ndarray, bbox: BBox, vtype: str) -> str:
        """Save a thumbnail crop of the violation region."""
        Path(self._thumbnail_dir).mkdir(parents=True, exist_ok=True)
        x1 = max(0, int(bbox.x1))
        y1 = max(0, int(bbox.y1))
        x2 = min(frame.shape[1], int(bbox.x2))
        y2 = min(frame.shape[0], int(bbox.y2))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return ""

        fname = f"{vtype}_{uuid.uuid4().hex[:8]}.jpg"
        path = str(Path(self._thumbnail_dir) / fname)
        cv2.imwrite(path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        return path
