"""Entry point for CV Safety Monitor v2 GPU machine.

Initializes all components: database, detector, classifiers,
alert manager, camera threads, UI, and FastAPI web server.
"""
import sys
from datetime import datetime

import numpy as np
from PyQt5.QtWidgets import QApplication

from gpu.alert_manager import AlertManager
from gpu.cam1_thread import Cam1Thread
from gpu.cam2_thread import Cam2Thread
from gpu.classifier import PPEManager
from gpu.database import get_connection, init_db, get_rois, get_cameras, upsert_camera
from gpu.detector import YOLODetector
from gpu.main_window import MainWindow
from gpu.roi_checker import ROIChecker
from gpu.roi_drawer import ROIDrawer
from gpu.web_server import WebServer, push_preview, ws_manager


class CVApp:
    """Application coordinator. Wires all components together."""

    def __init__(self):
        self._qapp = QApplication(sys.argv)
        self._qapp.setApplicationName("CV Safety Monitor v2")

        # Database
        self._db = init_db()
        self._init_cameras()
        self._init_default_rois()

        # Models
        self._detector_cam1 = YOLODetector()
        self._detector_cam2 = YOLODetector()
        self._ppe_manager = PPEManager()

        # ROI checker
        rois_cam1 = get_rois(self._db, "cam1")
        self._roi_checker = ROIChecker(rois_cam1)

        # Alert manager
        self._alert_manager = AlertManager(self._db, on_alert=self._on_alert_fired)

        # Main window
        self._window = MainWindow(self._alert_manager, self._db)

        for cam_id in ["cam1", "cam2"]:
            def make_save(rid):
                return lambda zone, pts: self._save_roi(rid, zone, pts)
            drawer = ROIDrawer(
                self._window.get_camera_widget(cam_id),
                on_save=make_save(cam_id),
            )
            self._window.register_drawer(cam_id, drawer)

        # Camera threads
        self._cameras = get_cameras(self._db)
        self._threads = []
        for cam in self._cameras:
            cam_id = cam["id"]
            port = cam["zmq_port"]
            if cam_id == "cam1":
                t = Cam1Thread(port, self._detector_cam1, self._roi_checker)
                t.frame_ready.connect(self._on_frame_ready)
                t.alert.connect(self._on_alert)
                self._threads.append(t)
            elif cam_id == "cam2":
                t = Cam2Thread(port, self._detector_cam2, self._ppe_manager)
                t.frame_ready.connect(self._on_frame_ready)
                t.alert.connect(self._on_alert)
                self._threads.append(t)

        # Start camera threads
        for t in self._threads:
            t.start()

        # Web server
        self._web = WebServer(self._db, self._alert_manager)
        self._web.start()

        self._window.show()

    def _init_cameras(self):
        if not get_cameras(self._db):
            upsert_camera(self._db, "cam1", 5555, "/dev/v4l/by-id/usb-cam1")
            upsert_camera(self._db, "cam2", 5556, "/dev/v4l/by-id/usb-cam2")
            self._db.commit()

    def _init_default_rois(self):
        import json
        from gpu.database import save_roi
        for cam_id in ["cam1", "cam2"]:
            if not get_rois(self._db, cam_id):
                # Define a default center-zone covering ~60% of frame
                zone = [[40, 30], [280, 30], [280, 210], [40, 210]]
                save_roi(self._db, cam_id, "default-zone", zone, "#ff0000")
        self._db.commit()

    def _on_frame_ready(self, camera_id: str, frame: np.ndarray):
        widget = self._window.get_camera_widget(camera_id)
        if widget is not None:
            widget.update_frame(frame)
        push_preview(camera_id, frame)

    def _on_alert(self, alert_dict: dict):
        vtype = alert_dict.get("type", "")
        zone = alert_dict.get("zone_name", "")
        camera_id = alert_dict.get("camera_id", vtype)
        text = f"[{datetime.now().strftime('%H:%M:%S')}] {vtype} - {camera_id}"
        if zone:
            text += f" - {zone}"
        self._window.add_alert_entry(text)

        # Broadcast alert to dashboard via WebSocket
        import asyncio
        import json
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(ws_manager.broadcast(json.dumps({
                "type": "violation",
                "violation": {
                    "type": vtype,
                    "camera_id": camera_id,
                    "zone": zone,
                    "timestamp": datetime.now().isoformat(),
                }
            })))
        except RuntimeError:
            pass

    def _on_alert_fired(self, vid, cam, vtype, zone, pidx, timestamp):
        pass

    def _save_roi(self, camera_id: str, zone_name: str, points: list):
        from gpu.database import save_roi
        save_roi(self._db, camera_id, zone_name, points)

    def run(self):
        sys.exit(self._qapp.exec_())


def main():
    app = CVApp()
    app.run()


if __name__ == "__main__":
    main()
