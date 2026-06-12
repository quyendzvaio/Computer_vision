import json
import pytest
from pathlib import Path


def test_init_db_creates_tables(temp_dir):
    """init_db() should create violations and roi_configs tables."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from alert.db import init_db, get_db

    db_path = temp_dir / "test.db"
    import alert.db as db_module
    original_path = db_module.DB_PATH
    db_module.DB_PATH = db_path

    try:
        init_db()
        conn = get_db()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        assert "violations" in tables
        assert "roi_configs" in tables
    finally:
        db_module.DB_PATH = original_path


def test_insert_and_query_violation(temp_dir):
    """Insert a violation and read it back."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from alert.db import init_db, insert_violation, get_violations

    import alert.db as db_module
    original_path = db_module.DB_PATH
    db_module.DB_PATH = temp_dir / "test.db"

    try:
        init_db()
        vid = insert_violation("cam-01", "NO_HELMET", "HIGH", [10, 20, 30, 40], "/tmp/thumb.jpg")
        assert vid > 0

        rows = get_violations()
        assert len(rows) == 1
        assert rows[0]["camera_id"] == "cam-01"
        assert rows[0]["type"] == "NO_HELMET"
        assert rows[0]["severity"] == "HIGH"
    finally:
        db_module.DB_PATH = original_path


def test_get_violations_with_filters(temp_dir):
    """get_violations should filter by camera_id, type, and time range."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from alert.db import init_db, insert_violation, get_violations

    import alert.db as db_module
    original_path = db_module.DB_PATH
    db_module.DB_PATH = temp_dir / "test.db"

    try:
        init_db()
        insert_violation("cam-01", "NO_HELMET", "HIGH", [10, 20, 30, 40], "/t1.jpg")
        insert_violation("cam-01", "FALL", "HIGH", [50, 60, 70, 80], "/t2.jpg")
        insert_violation("cam-02", "NO_VEST", "MEDIUM", [90, 100, 110, 120], "/t3.jpg")

        # Filter by camera
        rows = get_violations(camera_id="cam-01")
        assert len(rows) == 2

        # Filter by type
        rows = get_violations(violation_type="FALL")
        assert len(rows) == 1
        assert rows[0]["type"] == "FALL"

        # Filter by camera + type
        rows = get_violations(camera_id="cam-01", violation_type="NO_HELMET")
        assert len(rows) == 1
    finally:
        db_module.DB_PATH = original_path


def test_roi_config_crud(temp_dir):
    """Save and load ROI polygon for a camera."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from alert.db import init_db, save_roi, get_roi

    import alert.db as db_module
    original_path = db_module.DB_PATH
    db_module.DB_PATH = temp_dir / "test.db"

    try:
        init_db()
        polygon = [(100.0, 200.0), (300.0, 200.0), (300.0, 400.0), (100.0, 400.0)]
        save_roi("cam-01", polygon)

        loaded = get_roi("cam-01")
        assert loaded is not None
        assert loaded["camera_id"] == "cam-01"
        parsed_polygon = json.loads(loaded["polygon"])
        assert len(parsed_polygon) == 4
        assert parsed_polygon[0] == [100.0, 200.0]

        # Non-existent camera returns None
        assert get_roi("nonexistent") is None
    finally:
        db_module.DB_PATH = original_path


def test_get_all_rois(temp_dir):
    """get_all_rois() should return all saved ROI configs with proper camera_ids."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from alert.db import init_db, save_roi, get_all_rois

    import alert.db as db_module
    original_path = db_module.DB_PATH
    db_module.DB_PATH = temp_dir / "test.db"

    try:
        init_db()

        # Save 3 camera ROIs
        save_roi("cam-01", [(0.0, 0.0), (100.0, 100.0)])
        save_roi("cam-02", [(200.0, 200.0), (300.0, 300.0)])
        save_roi("cam-03", [(400.0, 400.0), (500.0, 500.0)])

        all_rois = get_all_rois()
        assert len(all_rois) == 3

        camera_ids = {r["camera_id"] for r in all_rois}
        assert camera_ids == {"cam-01", "cam-02", "cam-03"}
    finally:
        db_module.DB_PATH = original_path


def test_roi_upsert_update(temp_dir):
    """save_roi should update an existing camera's polygon, not create a new row."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from alert.db import init_db, save_roi, get_roi, get_all_rois

    import alert.db as db_module
    original_path = db_module.DB_PATH
    db_module.DB_PATH = temp_dir / "test.db"

    try:
        init_db()

        # Save initial polygon for cam-01
        polygon1 = [(0.0, 0.0), (10.0, 10.0)]
        save_roi("cam-01", polygon1)

        # Save a DIFFERENT polygon for same cam-01
        polygon2 = [(20.0, 20.0), (30.0, 30.0), (40.0, 40.0)]
        save_roi("cam-01", polygon2)

        # Read back — should be the new polygon, not the old one
        loaded = get_roi("cam-01")
        assert loaded is not None
        assert loaded["camera_id"] == "cam-01"
        parsed = json.loads(loaded["polygon"])
        assert len(parsed) == 3  # polygon2 has 3 points, polygon1 had 2
        assert parsed[0] == [20.0, 20.0]

        # Verify only 1 row exists for cam-01
        all_rois = get_all_rois()
        assert len(all_rois) == 1
        assert all_rois[0]["camera_id"] == "cam-01"
    finally:
        db_module.DB_PATH = original_path


def test_time_range_filters(temp_dir):
    """get_violations should filter by from_time and to_time."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from alert.db import init_db, insert_violation, get_violations, get_db

    import alert.db as db_module
    original_path = db_module.DB_PATH
    db_module.DB_PATH = temp_dir / "test.db"

    try:
        init_db()

        # Insert 3 violations
        v1 = insert_violation("cam-01", "NO_HELMET", "HIGH", [10, 20, 30, 40], "/t1.jpg")
        v2 = insert_violation("cam-02", "FALL", "HIGH", [50, 60, 70, 80], "/t2.jpg")
        v3 = insert_violation("cam-03", "NO_VEST", "MEDIUM", [90, 100, 110, 120], "/t3.jpg")

        # Set explicit timestamps so the filter is deterministic
        # SQLite CURRENT_TIMESTAMP has only second resolution, and all
        # three inserts land in the same second without this.
        conn = get_db()
        conn.execute("UPDATE violations SET created_at = '2024-01-01 10:00:00' WHERE id = ?", (v1,))
        conn.execute("UPDATE violations SET created_at = '2024-01-01 10:05:00' WHERE id = ?", (v2,))
        conn.execute("UPDATE violations SET created_at = '2024-01-01 10:10:00' WHERE id = ?", (v3,))
        conn.commit()
        conn.close()

        # Filter from 10:05 onward — should get v2 and v3
        rows_from = get_violations(from_time="2024-01-01 10:05:00")
        assert len(rows_from) == 2
        cam_ids = {r["camera_id"] for r in rows_from}
        assert cam_ids == {"cam-02", "cam-03"}

        # Filter up to 10:05 — should get v1 and v2
        rows_to = get_violations(to_time="2024-01-01 10:05:00")
        assert len(rows_to) == 2
        cam_ids = {r["camera_id"] for r in rows_to}
        assert cam_ids == {"cam-01", "cam-02"}

        # Filter narrow window — only v2
        rows_window = get_violations(
            from_time="2024-01-01 10:04:00",
            to_time="2024-01-01 10:06:00",
        )
        assert len(rows_window) == 1
        assert rows_window[0]["camera_id"] == "cam-02"
    finally:
        db_module.DB_PATH = original_path


def test_limit_offset(temp_dir):
    """get_violations should respect limit and offset parameters."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from alert.db import init_db, insert_violation, get_violations

    import alert.db as db_module
    original_path = db_module.DB_PATH
    db_module.DB_PATH = temp_dir / "test.db"

    try:
        init_db()

        # Insert 5 violations
        for i in range(5):
            insert_violation(
                f"cam-{i:02d}", "NO_HELMET", "HIGH",
                [10, 20, 30, 40], f"/t{i}.jpg"
            )

        # Query with limit=2 — should return exactly 2
        rows = get_violations(limit=2)
        assert len(rows) == 2

        # Query with limit=2 offset=2 — should return different rows
        rows_page2 = get_violations(limit=2, offset=2)
        assert len(rows_page2) == 2

        # Verify the two pages don't overlap
        ids_page1 = {r["id"] for r in rows}
        ids_page2 = {r["id"] for r in rows_page2}
        assert ids_page1.isdisjoint(ids_page2)
    finally:
        db_module.DB_PATH = original_path
