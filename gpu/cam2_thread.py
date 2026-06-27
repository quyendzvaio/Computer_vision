"""CAM2 QThread: ZMQ SUB -> YOLOv8 detect -> crop -> PPE classify -> ROI -> overlay.

Two-level skip:
  1. Motion detection: skip YOLOv8 if no motion.
  2. Classify skip: run classifiers every N frames even with motion.
"""
import time
from typing import List, Optional

import cv2
import numpy as np
import zmq
from PyQt5.QtCore import QThread, pyqtSignal

from gpu.detector import YOLODetector
from gpu.classifier import PPEManager
from gpu.ppe_checker import PPEChecker
from gpu.roi_checker import ROIChecker
from gpu.overlay import (
    draw_person_bboxes, draw_ppe_labels, draw_roi_polygons,
    draw_disconnected, draw_detection_offline,
)


FRAME_BUDGET_MS = 50  # ~20fps target (CAM2 is heavier)
MOTION_THRESHOLD = 30.0


class Cam2Thread(QThread):
    """CAM2 pipeline: receive frame -> detect persons -> PPE classify -> ROI check -> overlay."""
    frame_ready = pyqtSignal(str, np.ndarray)
    alert = pyqtSignal(dict)

    def __init__(self, zmq_port: int, detector: YOLODetector,
                 ppe_manager: PPEManager, roi_checker: ROIChecker, parent=None):
        super().__init__(parent)
        self._port = zmq_port
        self._detector = detector
        self._ppe_manager = ppe_manager
        self._ppe_checker = PPEChecker(ppe_manager)
        self._roi_checker = roi_checker
        self._running = True
        self._prev_gray: Optional[np.ndarray] = None
        self._disconnected = False
        self._frame_skip = 3
        self._frame_count = 0

    def stop(self):
        self._running = False

    def update_rois(self, rois: List[dict]):
        self._roi_checker.reload(rois)

    def _has_motion(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if self._prev_gray is None:
            self._prev_gray = gray
            return True
        diff = cv2.absdiff(self._prev_gray, gray)
        mean_diff = diff.mean()
        self._prev_gray = gray
        return mean_diff > MOTION_THRESHOLD

    def run(self):
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.setsockopt(zmq.RCVHWM, 2)
        sub.bind(f"tcp://*:{self._port}")
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        sub.setsockopt(zmq.RCVTIMEO, 1000)

        model_ok = True

        while self._running:
            t0 = time.perf_counter()

            try:
                data = sub.recv()
            except zmq.Again:
                if not self._disconnected:
                    self._disconnected = True
                    d = draw_disconnected(np.zeros((240, 320, 3), dtype=np.uint8))
                    self.frame_ready.emit("cam2", d)
                continue

            self._disconnected = False
            # Decode JPEG frame
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            self._frame_count += 1
            if self._frame_count % self._frame_skip != 0:
                self.frame_ready.emit("cam2", frame)
                continue

            t_start = time.perf_counter()

            # Level 1: motion skip
            if not self._has_motion(frame):
                self.frame_ready.emit("cam2", frame)
                continue

            try:
                roi_bounds = self._roi_checker.get_bounds()
                persons = self._detector.detect_roi(frame, roi_bounds)
                model_ok = True
            except Exception:
                if model_ok:
                    model_ok = False
                    self.frame_ready.emit("cam2", draw_detection_offline(frame))
                continue

            # ROI check — persons inside restricted zones
            zone_alerts = []
            for person in persons:
                foot_point = (person.bbox.x1 + person.bbox.width / 2, person.bbox.y2)
                zones = self._roi_checker.check_person(foot_point)
                for zone in zones:
                    zone_alerts.append({
                        "type": "PERSON_IN_ZONE",
                        "zone_name": zone["zone_name"],
                        "person_idx": id(person) & 0xFFFF,
                        "bbox": person.bbox,
                    })

            # Level 2: classify skip (internal counter)
            alerts = self._ppe_checker.process_persons(frame, persons)

            overlay = draw_person_bboxes(frame, persons)
            overlay = draw_ppe_labels(overlay, persons, alerts)
            overlay = draw_roi_polygons(overlay, self._roi_checker._rois)

            self.frame_ready.emit("cam2", overlay)

            # Emit zone alerts (PERSON_IN_ZONE)
            for alert_dict in zone_alerts:
                self.alert.emit(alert_dict)

            # Emit PPE alerts
            for alert_dict in alerts:
                for v in alert_dict["violations"]:
                    self.alert.emit({
                        "type": v,
                        "zone_name": "",
                        "person_idx": alert_dict["person_idx"],
                        "bbox": alert_dict["bbox"],
                    })

            elapsed_ms = (time.perf_counter() - t0) * 1000
            if elapsed_ms > FRAME_BUDGET_MS:
                pass

        sub.close()
        ctx.term()
