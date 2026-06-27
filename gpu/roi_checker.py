"""ROI (Region of Interest) checker using ray-casting point-in-polygon."""
import json
from typing import List, Optional, Tuple


def compute_roi_bounds(rois: List[dict], margin: int = 16,
                       frame_size: Optional[Tuple[int, int]] = None
                       ) -> Optional[Tuple[int, int, int, int]]:
    """Compute axis-aligned bounding box enclosing all enabled ROI polygons.

    Returns (x, y, w, h) in pixel coords, or None if no enabled ROI.
    Margin added around the bounds, clamped to frame_size if provided.
    """
    xs, ys = [], []
    for roi in rois:
        if not roi.get("enabled", True):
            continue
        polygon = json.loads(roi["points_json"])
        for pt in polygon:
            xs.append(pt[0])
            ys.append(pt[1])
    if not xs:
        return None

    x1 = max(0, min(xs) - margin)
    y1 = max(0, min(ys) - margin)
    x2 = max(xs) + margin
    y2 = max(ys) + margin

    if frame_size:
        x2 = min(frame_size[0], x2)
        y2 = min(frame_size[1], y2)

    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))


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

    def __init__(self, rois: Optional[List[dict]] = None,
                 frame_size: Optional[Tuple[int, int]] = None):
        self._rois = rois or []
        self._frame_size = frame_size

    def reload(self, rois: List[dict]):
        self._rois = rois

    def get_bounds(self, margin: int = 16) -> Optional[Tuple[int, int, int, int]]:
        """Cached bounding box covering all enabled ROI polygons.

        Returns (x, y, w, h) or None if no ROI configured.
        """
        return compute_roi_bounds(self._rois, margin, self._frame_size)

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
