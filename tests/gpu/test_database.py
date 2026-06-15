"""Test SQLite database operations."""
import pytest
from gpu.database import get_connection, init_db, upsert_camera, get_cameras, save_roi, get_rois, save_violation, get_violations


@pytest.fixture
def conn():
    c = init_db()
    yield c
    c.close()


def test_camera_crud(conn):
    upsert_camera(conn, "test-cam", 5555, "/dev/video0")
    cams = get_cameras(conn)
    assert any(c["id"] == "test-cam" for c in cams)


def test_roi_crud(conn):
    upsert_camera(conn, "cam1", 5555)
    save_roi(conn, "cam1", "Zone A", [[0, 0], [100, 0], [100, 100], [0, 100]])
    rois = get_rois(conn, "cam1")
    assert len(rois) == 1
    assert rois[0]["zone_name"] == "Zone A"


def test_violation_crud(conn):
    upsert_camera(conn, "cam1", 5555)
    vid = save_violation(conn, "cam1", "PERSON_IN_ZONE", "HIGH", zone_name="Zone A")
    assert len(vid) == 8
    violations = get_violations(conn)
    assert len(violations) == 1
    assert violations[0]["type"] == "PERSON_IN_ZONE"
