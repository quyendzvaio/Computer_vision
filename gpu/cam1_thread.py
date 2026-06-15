"""CAM1 QThread: ZMQ SUB -> YOLOv8 detect -> ROI check -> overlay -> emit.

Self-paced with frame budget. Latest frame strategy via ZMQ RCVHWM.
"""
import time
from typing import List, Optional

import cv2
import numpy as np
import zmq
from PyQt5.QtCore import QThread, pyqtSignal

from gpu.detector import YOLODetector
from gpu.roi_checker import ROIChecker
from gpu.overlay import draw_person_bboxes, draw_roi_polygons, draw_disconnected


FRAME_BUDGET_MS = 33  # ~30fps target


class Cam1Thread(QThread):
    """CAM1 pipeline: receive frame -> detect persons -> check ROI -> overlay.

    Signals:
        frame_ready: (camera_id: str, overlay_frame: np.ndarray)
        alert: (alert_dict: dict)
    """
    frame_ready = pyqtSignal(str, np.ndarray)
    alert = pyqtSignal(dict)

    def __init__(self, zmq_port: int, detector: YOLODetector,
                 roi_checker: ROIChecker, parent=None):
        super().__init__(parent)
        self._port = zmq_port
        self._detector = detector
        self._roi_checker = roi_checker
        self._running = True
        self._disconnected = False

    def stop(self):
        self._running = False

    def update_rois(self, rois: List[dict]):
        self._roi_checker.reload(rois)

    def run(self):
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.setsockopt(zmq.RCVHWM, 2)
        sub.bind(f"tcp://*:{self._port}")
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        sub.setsockopt(zmq.RCVTIMEO, 1000)

        while self._running:
            t0 = time.perf_counter()

            try:
                data = sub.recv()
            except zmq.Again:
                if not self._disconnected:
                    self._disconnected = True
                    d = draw_disconnected(np.zeros((480, 640, 3), dtype=np.uint8))
                    self.frame_ready.emit("cam1", d)
                continue

            self._disconnected = False
            frame = np.frombuffer(data, dtype=np.uint8).reshape((480, 640, 3))

            persons = self._detector.detect(frame)

            alerts = []
            for person in persons:
                foot_point = (person.bbox.x1 + person.bbox.width / 2, person.bbox.y2)
                zones = self._roi_checker.check_person(foot_point)
                for zone in zones:
                    alerts.append({
                        "type": "PERSON_IN_ZONE",
                        "zone_name": zone["zone_name"],
                        "person_idx": id(person) & 0xFFFF,
                        "bbox": person.bbox,
                    })

            overlay = draw_person_bboxes(frame, persons)
            overlay = draw_roi_polygons(overlay, self._roi_checker._rois)

            self.frame_ready.emit("cam1", overlay)
            for alert_dict in alerts:
                self.alert.emit(alert_dict)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            if elapsed_ms > FRAME_BUDGET_MS:
                pass  # HWM drops stale frames

        sub.close()
        ctx.term()
