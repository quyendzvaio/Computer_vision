"""Tests for alert.roi_matcher — ROI point-in-polygon matching with caching."""
import json
from unittest.mock import MagicMock


def make_fake_db(roi_data=None):
    """Create a fake db module-like object that returns preset ROI data.

    Mocks alert.db.get_roi() at the function level, returning a plain dict
    or None directly.
    """
    mock = MagicMock()
    if roi_data:
        mock.get_roi.return_value = roi_data
    else:
        mock.get_roi.return_value = None
    return mock


def test_is_in_roi_center_inside():
    """Point center (300,300) should be inside a square ROI."""
    from alert.roi_matcher import ROIMatcher

    polygon = [(100, 100), (500, 100), (500, 500), (100, 500)]
    fake_db = make_fake_db({
        "camera_id": "cam-01",
        "polygon": json.dumps(polygon),
    })
    matcher = ROIMatcher(fake_db)

    # bbox center at (300, 300) — clearly inside
    assert matcher.is_in_roi("cam-01", [250, 250, 350, 350]) is True


def test_is_in_roi_center_outside():
    """Point center (50,50) should be outside the ROI square."""
    from alert.roi_matcher import ROIMatcher

    polygon = [(100, 100), (500, 100), (500, 500), (100, 500)]
    fake_db = make_fake_db({
        "camera_id": "cam-01",
        "polygon": json.dumps(polygon),
    })
    matcher = ROIMatcher(fake_db)

    # bbox center at (50, 50) — clearly outside
    assert matcher.is_in_roi("cam-01", [25, 25, 75, 75]) is False


def test_is_in_roi_on_boundary():
    """Point on the polygon boundary should be considered inside (Shapely default)."""
    from alert.roi_matcher import ROIMatcher

    polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
    fake_db = make_fake_db({
        "camera_id": "cam-02",
        "polygon": json.dumps(polygon),
    })
    matcher = ROIMatcher(fake_db)

    # bbox center on left edge
    assert matcher.is_in_roi("cam-02", [0, 40, 0, 60]) is True


def test_is_in_roi_no_config():
    """When no ROI is configured, should return True (allow all)."""
    from alert.roi_matcher import ROIMatcher

    fake_db = make_fake_db(None)  # No ROI row
    matcher = ROIMatcher(fake_db)

    assert matcher.is_in_roi("unknown-cam", [0, 0, 10, 10]) is True


def test_roi_cache_reuse():
    """ROI polygon should be cached after first load."""
    from alert.roi_matcher import ROIMatcher

    polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
    fake_db = make_fake_db({
        "camera_id": "cam-01",
        "polygon": json.dumps(polygon),
    })
    matcher = ROIMatcher(fake_db)

    # First call — loads from DB
    matcher.is_in_roi("cam-01", [50, 50, 60, 60])
    # Second call — should use cache (db.get_roi not called again)
    matcher.is_in_roi("cam-01", [30, 30, 40, 40])

    # DB should have been queried exactly once for this camera
    assert fake_db.get_roi.call_count == 1


def test_invalidate_cache():
    """invalidate() should force reload on next check."""
    from alert.roi_matcher import ROIMatcher

    polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
    fake_db = make_fake_db({
        "camera_id": "cam-01",
        "polygon": json.dumps(polygon),
    })
    matcher = ROIMatcher(fake_db)

    matcher.is_in_roi("cam-01", [50, 50, 60, 60])
    matcher.invalidate("cam-01")
    matcher.is_in_roi("cam-01", [30, 30, 40, 40])

    # DB should have been queried twice (cache cleared between)
    assert fake_db.get_roi.call_count == 2
