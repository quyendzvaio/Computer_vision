"""ROI (Region of Interest) checker using ray-casting point-in-polygon."""
import json
from typing import List, Optional, Tuple


def point_in_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> bool:
    """Ray-casting algorithm. Returns True if point is inside polygon."""
    x, y = point
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        # Check if point is on an edge
        edge_dist = abs((xj - xi) * (yi - y) - (xi - x) * (yj - yi))
        edge_len = ((xj - xi) ** 2 + (yj - yi) ** 2) ** 0.5
        if edge_len > 0 and edge_dist / edge_len < 1.0:
            dot = ((x - xi) * (xj - xi) + (y - yi) * (yj - yi))
            if 0 <= dot <= edge_len * edge_len:
                return True

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside

        j = i

    return inside


class ROIChecker:
    """Check if detected persons are inside any ROI zone."""

    def __init__(self, rois: Optional[List[dict]] = None):
        self._rois = rois or []

    def reload(self, rois: List[dict]):
        self._rois = rois

    def check_person(self, foot_point: Tuple[float, float]) -> List[dict]:
        """Check if a foot point falls inside any ROI.

        Returns:
            List of ROI dicts that contain the point. Empty if not in any zone.
        """
        inside_zones = []
        for roi in self._rois:
            if not roi.get("enabled", True):
                continue
            polygon = json.loads(roi["points_json"])
            if point_in_polygon(foot_point, polygon):
                inside_zones.append(roi)
        return inside_zones
