"""SQLite database helper for CV Safety Monitor v2.

Manages cameras, ROI configs, violations, and settings tables.
Thread-safe via check_same_thread=False + per-call cursor.
"""
import json
import sqlite3
import uuid
from pathlib import Path
from typing import List, Optional

DB_PATH = Path(__file__).parent.parent / "data" / "cv.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_connection()
    conn.executescript("""
        DROP TABLE IF EXISTS violations;
        DROP TABLE IF EXISTS roi_config;
        DROP TABLE IF EXISTS cameras;
        DROP TABLE IF EXISTS settings;

        CREATE TABLE IF NOT EXISTS cameras (
            id TEXT PRIMARY KEY,
            zmq_port INTEGER NOT NULL,
            device_path TEXT,
            enabled BOOLEAN DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS roi_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT NOT NULL REFERENCES cameras(id),
            zone_name TEXT NOT NULL,
            points_json TEXT NOT NULL,
            color TEXT NOT NULL DEFAULT '#ff0000',
            enabled BOOLEAN DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS violations (
            id TEXT PRIMARY KEY,
            camera_id TEXT NOT NULL,
            type TEXT NOT NULL,
            severity TEXT NOT NULL DEFAULT 'MEDIUM',
            zone_name TEXT,
            person_idx INTEGER,
            bbox_json TEXT,
            thumbnail_path TEXT,
            acknowledged BOOLEAN DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()
    return get_connection()


def upsert_camera(conn: sqlite3.Connection, cam_id: str, zmq_port: int, device_path: str = ""):
    conn.execute(
        "INSERT OR REPLACE INTO cameras (id, zmq_port, device_path) VALUES (?, ?, ?)",
        (cam_id, zmq_port, device_path),
    )
    conn.commit()


def get_cameras(conn: sqlite3.Connection) -> List[dict]:
    rows = conn.execute("SELECT * FROM cameras WHERE enabled=1").fetchall()
    return [dict(r) for r in rows]


# --- ROI ---

def save_roi(conn: sqlite3.Connection, camera_id: str, zone_name: str, points: list, color: str = "#ff0000"):
    conn.execute(
        "INSERT OR REPLACE INTO roi_config (camera_id, zone_name, points_json, color) VALUES (?, ?, ?, ?)",
        (camera_id, zone_name, json.dumps(points), color),
    )
    conn.commit()


def get_rois(conn: sqlite3.Connection, camera_id: str) -> List[dict]:
    rows = conn.execute(
        "SELECT * FROM roi_config WHERE camera_id=? AND enabled=1", (camera_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def delete_roi(conn: sqlite3.Connection, roi_id: int):
    conn.execute("DELETE FROM roi_config WHERE id=?", (roi_id,))
    conn.commit()


# --- Violations ---

def save_violation(conn: sqlite3.Connection, camera_id: str, vtype: str, severity: str,
                   zone_name: str = "", bbox_json: str = "", thumbnail_path: str = "") -> str:
    vid = str(uuid.uuid4())[:8]
    conn.execute(
        "INSERT INTO violations (id, camera_id, type, severity, zone_name, bbox_json, thumbnail_path) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (vid, camera_id, vtype, severity, zone_name, bbox_json, thumbnail_path),
    )
    conn.commit()
    return vid


def get_violations(conn: sqlite3.Connection, limit: int = 50, offset: int = 0,
                   camera_id: Optional[str] = None) -> List[dict]:
    if camera_id:
        rows = conn.execute(
            "SELECT * FROM violations WHERE camera_id=? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (camera_id, limit, offset),
        )
    else:
        rows = conn.execute(
            "SELECT * FROM violations ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
    return [dict(r) for r in rows]
