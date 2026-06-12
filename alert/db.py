"""SQLite database layer for violations and ROI configs."""
import json
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

DB_PATH = Path("data/cv.db")


def get_db() -> sqlite3.Connection:
    """Get a database connection with row factory enabled."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS violations (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id       TEXT    NOT NULL,
            type            TEXT    NOT NULL,
            severity        TEXT    NOT NULL DEFAULT 'MEDIUM',
            bbox            TEXT,
            thumbnail_path  TEXT,
            created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS roi_configs (
            camera_id   TEXT PRIMARY KEY,
            polygon     TEXT NOT NULL,
            updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_violations_camera
            ON violations(camera_id);
        CREATE INDEX IF NOT EXISTS idx_violations_type
            ON violations(type);
        CREATE INDEX IF NOT EXISTS idx_violations_created
            ON violations(created_at);
    """)
    conn.commit()
    conn.close()


def insert_violation(
    camera_id: str,
    violation_type: str,
    severity: str,
    bbox: List[float],
    thumbnail_path: str,
) -> int:
    """Insert a violation record and return its ID."""
    conn = get_db()
    cursor = conn.execute(
        "INSERT INTO violations (camera_id, type, severity, bbox, thumbnail_path) "
        "VALUES (?, ?, ?, ?, ?)",
        (camera_id, violation_type, severity, json.dumps(bbox), thumbnail_path),
    )
    conn.commit()
    vid = cursor.lastrowid
    conn.close()
    return vid


def get_violations(
    camera_id: Optional[str] = None,
    violation_type: Optional[str] = None,
    from_time: Optional[str] = None,
    to_time: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Query violations with optional filters. Returns list of dicts."""
    conn = get_db()
    query = "SELECT * FROM violations WHERE 1=1"
    params: List[Any] = []

    if camera_id:
        query += " AND camera_id = ?"
        params.append(camera_id)
    if violation_type:
        query += " AND type = ?"
        params.append(violation_type)
    if from_time:
        query += " AND created_at >= ?"
        params.append(from_time)
    if to_time:
        query += " AND created_at <= ?"
        params.append(to_time)

    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    rows = [dict(row) for row in conn.execute(query, params).fetchall()]
    conn.close()
    return rows


def save_roi(camera_id: str, polygon: List[Tuple[float, float]]):
    """Insert or update ROI polygon for a camera.

    polygon can be a list of tuples or lists of coordinate pairs.
    """
    conn = get_db()
    # Convert tuples to lists for consistent JSON storage
    polygon_list = [[float(p[0]), float(p[1])] for p in polygon]
    polygon_json = json.dumps(polygon_list)
    conn.execute(
        "INSERT INTO roi_configs (camera_id, polygon, updated_at) "
        "VALUES (?, ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(camera_id) DO UPDATE SET polygon=excluded.polygon, "
        "updated_at=CURRENT_TIMESTAMP",
        (camera_id, polygon_json),
    )
    conn.commit()
    conn.close()


def get_roi(camera_id: str) -> Optional[Dict[str, Any]]:
    """Get ROI config for a camera, or None if not set."""
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM roi_configs WHERE camera_id = ?", (camera_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_rois() -> List[Dict[str, Any]]:
    """Get all ROI configs."""
    conn = get_db()
    rows = [dict(row) for row in conn.execute("SELECT * FROM roi_configs").fetchall()]
    conn.close()
    return rows
