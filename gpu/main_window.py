"""PyQt5 MainWindow for CV Safety Monitor v2.

Toolbar: camera tabs, ROI draw mode, settings.
Layout: side-by-side camera widgets, status bar for alert log.
"""
import sqlite3
from datetime import datetime
from typing import Dict, Optional

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDockWidget, QHBoxLayout, QListWidget, QListWidgetItem,
    QMainWindow, QPushButton, QVBoxLayout, QWidget,
)

from gpu.camera_widget import CameraWidget
from gpu.roi_drawer import ROIDrawer


class MainWindow(QMainWindow):
    """Main application window with toolbar, camera feeds, status bar."""

    def __init__(self, alert_manager, db_conn: sqlite3.Connection):
        super().__init__()
        self._alert_manager = alert_manager
        self._db = db_conn

        self._drawer_mode = False
        self._drawers: Dict[str, ROIDrawer] = {}
        self._cams: Dict[str, CameraWidget] = {}

        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("CV Safety Monitor v2")
        self.setMinimumSize(1280, 720)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        for cam_id in ["cam1", "cam2"]:
            widget = CameraWidget(cam_id)
            self._cams[cam_id] = widget
            layout.addWidget(widget)

        # Toolbar
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)

        self._draw_btn = QPushButton("✎ ROI Draw")
        self._draw_btn.setCheckable(True)
        self._draw_btn.toggled.connect(self._toggle_draw_mode)
        toolbar.addWidget(self._draw_btn)

        self._status_label = QPushButton("Ready")
        self._status_label.setEnabled(False)
        toolbar.addWidget(self._status_label)

        # Status bar as dock
        self._alert_list = QListWidget()
        status_dock = QDockWidget("Alerts", self)
        status_dock.setWidget(self._alert_list)
        self.addDockWidget(Qt.BottomDockWidgetArea, status_dock)

    def _toggle_draw_mode(self, enabled: bool):
        self._drawer_mode = enabled
        for cam_id, drawer in self._drawers.items():
            drawer.set_mode("draw" if enabled else "view")

    def register_drawer(self, cam_id: str, drawer: ROIDrawer):
        self._drawers[cam_id] = drawer

    def get_camera_widget(self, cam_id: str) -> Optional[CameraWidget]:
        return self._cams.get(cam_id)

    def add_alert_entry(self, text: str):
        item = QListWidgetItem(text)
        self._alert_list.insertItem(0, item)
        while self._alert_list.count() > 100:
            self._alert_list.takeItem(self._alert_list.count() - 1)

    def set_status(self, text: str):
        self._status_label.setText(text)
