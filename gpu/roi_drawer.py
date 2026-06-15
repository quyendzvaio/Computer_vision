"""ROI polygon drawing interaction on CameraWidget."""
from typing import Callable, List, Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMouseEvent


class ROIDrawer:
    """Manages ROI drawing interaction state.

    Modes:
        - 'view': normal viewing, no editing
        - 'draw': drawing new polygon (click to add vertices)
        - 'edit': editing existing polygon
    """

    def __init__(self, widget, on_save: Callable[[str, list], None]):
        self._widget = widget
        self._on_save = on_save
        self._mode = "view"
        self._current_polygon: List[Tuple[float, float]] = []
        self._current_zone_name = "Zone A"

    def set_mode(self, mode: str):
        self._mode = mode

    def set_zone_name(self, name: str):
        self._current_zone_name = name

    def get_polygon_data(self) -> list:
        return self._current_polygon

    def mouse_press_event(self, event: QMouseEvent,
                          widget_coords: Tuple[float, float]) -> bool:
        """Handle mouse press. Returns True if event was consumed."""
        if self._mode != "draw":
            return False

        x, y = widget_coords
        if event.button() == Qt.LeftButton:
            self._current_polygon.append([x, y])
            self._widget.update()
            return True
        elif event.button() == Qt.RightButton:
            if len(self._current_polygon) >= 3:
                self._on_save(self._current_zone_name, self._current_polygon)
                self._current_polygon = []
            return True

        return False

    def mouse_double_click_event(self, event: QMouseEvent,
                                 widget_coords: Tuple[float, float]) -> bool:
        """Double-click closes and saves polygon."""
        if self._mode != "draw":
            return False
        if len(self._current_polygon) >= 3:
            self._on_save(self._current_zone_name, self._current_polygon)
            self._current_polygon = []
        return True
