"""Test ROI checker point-in-polygon logic."""
import pytest
from gpu.roi_checker import point_in_polygon, ROIChecker


def test_point_inside_square():
    poly = [[0, 0], [100, 0], [100, 100], [0, 100]]
    assert point_in_polygon((50, 50), poly) is True


def test_point_outside_square():
    poly = [[0, 0], [100, 0], [100, 100], [0, 100]]
    assert point_in_polygon((200, 200), poly) is False


def test_point_on_edge():
    poly = [[0, 0], [100, 0], [100, 100], [0, 100]]
    assert point_in_polygon((50, 0), poly) is True


def test_point_on_vertex():
    poly = [[0, 0], [100, 0], [100, 100], [0, 100]]
    assert point_in_polygon((0, 0), poly) is True


def test_point_inside_triangle():
    poly = [[0, 0], [100, 50], [0, 100]]
    assert point_in_polygon((30, 50), poly) is True


def test_roi_checker_empty_rois():
    checker = ROIChecker([])
    assert checker.check_person((50, 50)) == []


def test_roi_checker_inside():
    checker = ROIChecker([
        {"zone_name": "Zone A", "color": "#ff0000", "enabled": True,
         "points_json": "[[0,0],[100,0],[100,100],[0,100]]"},
    ])
    zones = checker.check_person((50, 50))
    assert len(zones) == 1
    assert zones[0]["zone_name"] == "Zone A"


def test_roi_checker_disabled_zone():
    checker = ROIChecker([
        {"zone_name": "Zone A", "color": "#ff0000", "enabled": False,
         "points_json": "[[0,0],[100,0],[100,100],[0,100]]"},
    ])
    assert checker.check_person((50, 50)) == []
