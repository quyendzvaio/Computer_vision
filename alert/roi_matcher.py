"""ROI point-in-polygon matching using Shapely."""
import json
from typing import Dict, List, Optional

from shapely.geometry import Point, Polygon


class ROIMatcher:
    """Checks whether a detection's bounding box center falls within a camera's
    configured ROI polygon. Results are cached per camera; call invalidate()
    when the ROI is updated via the admin UI."""

    def __init__(self, db):
        """db: alert.db module (or compatible mock with get_roi method)."""
        self._db = db
        self._cache: Dict[str, Optional[Polygon]] = {}

    def _load_polygon(self, camera_id: str) -> Optional[Polygon]:
        """Load ROI polygon from DB for a camera. Returns None if not configured."""
        row = self._db.get_roi(camera_id)
        if row is None:
            return None
        points = json.loads(row["polygon"])  # [[x,y], [x,y], ...]
        if len(points) < 3:
            return None
        return Polygon(points)

    def _get_polygon(self, camera_id: str) -> Optional[Polygon]:
        """Get cached polygon or load from DB."""
        if camera_id not in self._cache:
            self._cache[camera_id] = self._load_polygon(camera_id)
        return self._cache[camera_id]

    def is_in_roi(self, camera_id: str, bbox: List[float]) -> bool:
        """Check if bbox [x1,y1,x2,y2] center is inside the camera's ROI polygon.
        If no ROI is configured for this camera, returns True (allow all)."""
        polygon = self._get_polygon(camera_id)
        if polygon is None:
            return True  # No ROI configured = allow everything

        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        point = Point(center_x, center_y)
        return polygon.contains(point) or polygon.touches(point)

    def invalidate(self, camera_id: str) -> None:
        """Clear cache for a camera (call after ROI update)."""
        self._cache.pop(camera_id, None)

    def invalidate_all(self) -> None:
        """Clear entire cache."""
        self._cache.clear()
