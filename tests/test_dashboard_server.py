"""Tests for dashboard FastAPI server."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(temp_dir):
    """Create TestClient with isolated DB and temp static dir."""
    from alert.db import init_db
    import alert.db as db_module

    original_path = db_module.DB_PATH
    db_path = temp_dir / "test_dashboard.db"
    db_module.DB_PATH = db_path
    init_db()

    # Create temp static dir
    static_dir = temp_dir / "static"
    static_dir.mkdir()

    with patch('dashboard.server.STATIC_DIR', str(static_dir)):
        with patch('dashboard.server.THUMBNAIL_DIR', str(temp_dir / "thumbnails")):
            from dashboard.server import app
            yield TestClient(app)

    db_module.DB_PATH = original_path


def test_get_cameras_empty(client):
    """GET /api/cameras returns list (may be empty or from config)."""
    resp = client.get("/api/cameras")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


def test_get_roi_not_found(client):
    """GET /api/roi/unknown should return 404."""
    resp = client.get("/api/roi/nonexistent")
    assert resp.status_code == 404


def test_put_and_get_roi(client):
    """PUT then GET /api/roi/{camera_id} should round-trip."""
    polygon = [[100, 200], [300, 200], [300, 400], [100, 400]]
    resp = client.put("/api/roi/cam-01", json={"polygon": polygon})
    assert resp.status_code == 200

    resp = client.get("/api/roi/cam-01")
    assert resp.status_code == 200
    data = resp.json()
    assert data["camera_id"] == "cam-01"
    assert data["polygon"] == polygon


def test_get_violations_empty(client):
    """GET /api/violations returns empty list initially."""
    resp = client.get("/api/violations")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 0


def test_get_violations_with_data(client):
    """GET /api/violations returns violations after inserts."""
    from alert.db import insert_violation

    insert_violation("cam-01", "NO_HELMET", "HIGH", [10, 20, 30, 40], "/tmp/t.jpg")
    insert_violation("cam-02", "FALL", "HIGH", [50, 60, 70, 80], "/tmp/t2.jpg")

    resp = client.get("/api/violations")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2

    # Test filtering by camera
    resp = client.get("/api/violations?camera_id=cam-01")
    data = resp.json()
    assert len(data) == 1
    assert data[0]["camera_id"] == "cam-01"


def test_get_thumbnail_not_found(client):
    """GET /api/violations/999/thumbnail should return 404."""
    resp = client.get("/api/violations/999/thumbnail")
    assert resp.status_code == 404
