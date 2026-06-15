"""PyQt QWidget for displaying camera video with overlay.

Zero-copy: QImage wraps numpy array directly without copy.
"""
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPainter, QPixmap
from PyQt5.QtWidgets import QWidget


class CameraWidget(QWidget):
    """Widget that displays one camera feed with overlay."""

    def __init__(self, camera_id: str, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self._pixmap: Optional[QPixmap] = None
        self.setMinimumSize(320, 240)

    def update_frame(self, frame: np.ndarray):
        """Set current frame and trigger repaint."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def paintEvent(self, event):
        if self._pixmap is None:
            return
        painter = QPainter(self)
        scaled = self._pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
        painter.end()
