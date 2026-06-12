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
