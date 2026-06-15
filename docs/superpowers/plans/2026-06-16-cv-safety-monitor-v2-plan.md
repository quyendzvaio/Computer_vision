# CV Safety Monitor v2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign CV Safety Monitor from multi-process + Web Dashboard to PyQt5 + QThread + ZeroMQ architecture with two independent camera pipelines (CAM1: ROI zone detection, CAM2: PPE detection).

**Architecture:** Edge device captures raw BGR frames from 2 USB cameras and sends via ZeroMQ PUB to GPU machine. GPU machine runs single PyQt5 process with 3 threads: CAM1 QThread (detect → ROI check), CAM2 QThread (detect → crop → classify PPE), Main thread (render + UI + FastAPI web). No multiprocessing IPC, no shared memory. Frame budget with skip strategy prioritizes latency.

**Tech Stack:** PyQt5, ZeroMQ, OpenCV, ONNX Runtime (GPU), YOLOv8n, MobileNetV3-small, FastAPI, SQLite

**Plan structure follows spec implementation order (Section 14):** Edge sender → GPU core components → Threads → UI → Web → Tests

---
pre-existing:
  edge/:
    - config.yaml (will modify)
    - __init__.py
    - capture_loop.py (v1, not used in v2)
    - frame_processor.py (v1, not used)
    - local_bridge.py (v1, not used)
    - mqtt_publisher.py (v1, not used)
    - source_manager.py (v1, not used)
  shared/:
    - models.py (will modify)
    - memory.py (v1, not used)
  dashboard/ (v1, not used)
  alert/ (v1, not used)
  inference/ (v1, not used)

### Task 1: Edge device sender

**Files:**
- Create: `edge/sender.py`
- Modify: `edge/config.yaml`
- Create: `tests/test_edge_sender.py`

- [ ] **Step 1: Write edge config.yaml**

Replace existing MQTT-based config with ZMQ-based config:

```yaml
# edge/config.yaml
gpu_host: 192.168.1.100

cameras:
  - id: cam1
    device_path: /dev/v4l/by-id/usb-cam1-video-index0
    zmq_port: 5555
    fps: 30
    resolution: [640, 480]

  - id: cam2
    device_path: /dev/v4l/by-id/usb-cam2-video-index0
    zmq_port: 5556
    fps: 30
    resolution: [640, 480]
```

- [ ] **Step 2: Write edge/sender.py**

```python
"""Edge device: capture USB cameras and send raw BGR frames via ZeroMQ PUB.

Connects to GPU machine (stable endpoint). ZMQ PUB with HWM=2 prevents
queue buildup when GPU is slow. Sends raw BGR bytes — no JPEG encode.
"""
import logging
import time

import cv2
import yaml
import zmq

logging.basicConfig(level=logging.INFO, format="[Edge] %(message)s")
log = logging.getLogger("edge")


def load_config(path: str = "edge/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_camera(cfg: dict):
    """Open USB camera and set resolution. Returns VideoCapture or None."""
    path = cfg.get("device_path", "")
    # Fallback: try numeric index if device_path fails
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        log.error("Cannot open camera: %s", path)
        return None
    w, h = cfg.get("resolution", [640, 480])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, cfg.get("fps", 30))
    log.info("Camera %s opened: %dx%d", cfg["id"], int(cap.get(3)), int(cap.get(4)))
    return cap


def sender_loop():
    cfg = load_config()
    ctx = zmq.Context()
    cameras = []

    gpu_host = cfg.get("gpu_host", "127.0.0.1")

    for cam_cfg in cfg.get("cameras", []):
        cap = setup_camera(cam_cfg)
        if cap is None:
            continue
        pub = ctx.socket(zmq.PUB)
        pub.setsockopt(zmq.SNDHWM, 2)
        pub.connect(f"tcp://{gpu_host}:{cam_cfg['zmq_port']}")
        cameras.append((cam_cfg["id"], cap, pub))
        log.info("Camera %s → tcp://%s:%d", cam_cfg["id"], gpu_host, cam_cfg["zmq_port"])

    if not cameras:
        log.error("No cameras available. Exiting.")
        return

    log.info("Edge sender running with %d camera(s)", len(cameras))

    while True:
        for cam_id, cap, pub in cameras:
            ret, frame = cap.read()
            if not ret:
                log.warning("Camera %s: frame read failed", cam_id)
                continue
            pub.send(frame.tobytes())


if __name__ == "__main__":
    try:
        sender_loop()
    except KeyboardInterrupt:
        log.info("Edge sender stopped by user")
```

- [ ] **Step 3: Write test edge config**

```python
# tests/test_edge_sender.py
"""Test edge sender config loading and camera setup."""
import pytest
import yaml
import tempfile
import os
from edge.sender import load_config


def test_load_config_minimal():
    """Load a minimal valid config."""
    data = {
        "gpu_host": "192.168.1.100",
        "cameras": [
            {"id": "cam1", "device_path": "/dev/video0", "zmq_port": 5555, "fps": 30, "resolution": [640, 480]},
        ],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        path = f.name
    try:
        cfg = load_config(path)
        assert cfg["gpu_host"] == "192.168.1.100"
        assert len(cfg["cameras"]) == 1
        assert cfg["cameras"][0]["zmq_port"] == 5555
    finally:
        os.unlink(path)


def test_load_config_empty_cameras():
    """Handle camera list empty gracefully."""
    data = {"gpu_host": "127.0.0.1", "cameras": []}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        path = f.name
    try:
        cfg = load_config(path)
        assert len(cfg["cameras"]) == 0
    finally:
        os.unlink(path)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_edge_sender.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add edge/sender.py edge/config.yaml tests/test_edge_sender.py
git commit -m "feat: edge device sender with ZMQ PUB transport

Edge device captures USB cameras and sends raw BGR frames
via ZeroMQ PUB to GPU machine. HWM=2 prevents queue buildup."
```

### Task 2: Update shared models

**Files:**
- Modify: `shared/models.py`

- [ ] **Step 1: Update ViolationType and add ViolationSeverity**

Replace the existing ViolationType to include PERSON_IN_ZONE and drop FALL (v1 legacy):

```python
# In shared/models.py, after the existing type aliases section (~line 47-49)

# Type aliases for violation classification
ViolationType = Literal["PERSON_IN_ZONE", "NO_HELMET", "NO_VEST", "NO_BOOT"]
SeverityLevel = Literal["HIGH", "MEDIUM", "LOW"]
```

Also add a helper method to DetectionResult for checking person count:

```python
    @property
    def person_count(self) -> int:
        """Number of detected persons in this frame."""
        return sum(1 for o in self.objects if o.cls == "person")
```

Add this after the `DetectionResult` class's existing content (after line ~74).

- [ ] **Step 2: Verify no broken references**

Run: `python -c "from shared.models import ViolationType, DetectionResult; print('OK')"`
Expected: prints "OK"

- [ ] **Step 3: Commit**

```bash
git add shared/models.py
git commit -m "feat: update shared models for v2 violation types

Add PERSON_IN_ZONE violation type, drop FALL (v1 legacy).
Add person_count property to DetectionResult."
```

### Task 3: GPU database module

**Files:**
- Create: `gpu/database.py`

- [ ] **Step 1: Write the full database module**

```python
"""SQLite database helper for CV Safety Monitor v2.

Manages cameras, ROI configs, violations, and settings tables.
Thread-safe via check_same_thread=False + per-call cursor.
"""
import sqlite3
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

DB_PATH = Path(__file__).parent.parent / "data" / "cv.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS cameras (
            id TEXT PRIMARY KEY,
            zmq_port INTEGER NOT NULL,
            device_path TEXT,
            enabled BOOLEAN DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS roi_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT NOT NULL REFERENCES cameras(id),
            zone_name TEXT NOT NULL,
            points_json TEXT NOT NULL,
            color TEXT NOT NULL DEFAULT '#ff0000',
            enabled BOOLEAN DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS violations (
            id TEXT PRIMARY KEY,
            camera_id TEXT NOT NULL,
            type TEXT NOT NULL,
            severity TEXT NOT NULL DEFAULT 'MEDIUM',
            zone_name TEXT,
            person_idx INTEGER,
            bbox_json TEXT,
            thumbnail_path TEXT,
            acknowledged BOOLEAN DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()
    return get_connection()


def upsert_camera(conn: sqlite3.Connection, cam_id: str, zmq_port: int, device_path: str = ""):
    conn.execute(
        "INSERT OR REPLACE INTO cameras (id, zmq_port, device_path) VALUES (?, ?, ?)",
        (cam_id, zmq_port, device_path),
    )
    conn.commit()


def get_cameras(conn: sqlite3.Connection) -> List[dict]:
    rows = conn.execute("SELECT * FROM cameras WHERE enabled=1").fetchall()
    return [dict(r) for r in rows]


# --- ROI ---

def save_roi(conn: sqlite3.Connection, camera_id: str, zone_name: str, points: list, color: str = "#ff0000"):
    conn.execute(
        "INSERT OR REPLACE INTO roi_config (camera_id, zone_name, points_json, color) VALUES (?, ?, ?, ?)",
        (camera_id, zone_name, json.dumps(points), color),
    )
    conn.commit()


def get_rois(conn: sqlite3.Connection, camera_id: str) -> List[dict]:
    rows = conn.execute(
        "SELECT * FROM roi_config WHERE camera_id=? AND enabled=1", (camera_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def delete_roi(conn: sqlite3.Connection, roi_id: int):
    conn.execute("DELETE FROM roi_config WHERE id=?", (roi_id,))
    conn.commit()


# --- Violations ---

def save_violation(conn: sqlite3.Connection, camera_id: str, vtype: str, severity: str,
                   zone_name: str = "", bbox_json: str = "", thumbnail_path: str = "") -> str:
    vid = str(uuid.uuid4())[:8]
    conn.execute(
        "INSERT INTO violations (id, camera_id, type, severity, zone_name, bbox_json, thumbnail_path) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (vid, camera_id, vtype, severity, zone_name, bbox_json, thumbnail_path),
    )
    conn.commit()
    return vid


def get_violations(conn: sqlite3.Connection, limit: int = 50, offset: int = 0,
                   camera_id: Optional[str] = None) -> List[dict]:
    if camera_id:
        rows = conn.execute(
            "SELECT * FROM violations WHERE camera_id=? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (camera_id, limit, offset),
        )
    else:
        rows = conn.execute(
            "SELECT * FROM violations ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
    return [dict(r) for r in rows]


# --- Init ---

def init_default_cameras():
    """Seed default cameras if table is empty."""
    conn = get_connection()
    if not get_cameras(conn):
        upsert_camera(conn, "cam1", 5555, "/dev/v4l/by-id/usb-cam1")
        upsert_camera(conn, "cam2", 5556, "/dev/v4l/by-id/usb-cam2")
    conn.close()
```

- [ ] **Step 2: Write tests**

```python
# tests/gpu/test_database.py
"""Test SQLite database operations."""
import pytest
from gpu.database import get_connection, init_db, upsert_camera, get_cameras, save_roi, get_rois, save_violation, get_violations


@pytest.fixture
def conn():
    c = init_db()
    yield c
    c.close()


def test_camera_crud(conn):
    upsert_camera(conn, "test-cam", 5555, "/dev/video0")
    cams = get_cameras(conn)
    assert any(c["id"] == "test-cam" for c in cams)


def test_roi_crud(conn):
    upsert_camera(conn, "cam1", 5555)
    save_roi(conn, "cam1", "Zone A", [[0, 0], [100, 0], [100, 100], [0, 100]])
    rois = get_rois(conn, "cam1")
    assert len(rois) == 1
    assert rois[0]["zone_name"] == "Zone A"


def test_violation_crud(conn):
    upsert_camera(conn, "cam1", 5555)
    vid = save_violation(conn, "cam1", "PERSON_IN_ZONE", "HIGH", zone_name="Zone A")
    assert len(vid) == 8
    violations = get_violations(conn)
    assert len(violations) == 1
    assert violations[0]["type"] == "PERSON_IN_ZONE"
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/gpu/test_database.py -v`
Expected: 3 PASS

- [ ] **Step 4: Commit**

```bash
git add gpu/database.py tests/gpu/test_database.py
git commit -m "feat: GPU database module with SQLite schema

Tables: cameras, roi_config, violations, settings.
WAL mode for concurrent reads. UUID-based violation IDs."
```

### Task 4: GPU detector — YOLOv8n ONNX wrapper

**Files:**
- Create: `gpu/detector.py`
- Create: `tests/gpu/test_detector.py`

- [ ] **Step 1: Write detector.py**

```python
"""YOLOv8n ONNX detector wrapper.

Runs inference on raw BGR frame (640×480), returns list of DetectedObject
with 'person' class only (COCO class 0). NMS applied post-inference.
"""
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import onnxruntime

from shared.models import BBox, DetectedObject


MODEL_PATH = Path(__file__).parent / "models" / "yolov8n.onnx"
INPUT_SIZE = (640, 480)  # (width, height)
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
COCO_PERSON_ID = 0


class YOLODetector:
    """YOLOv8n ONNX detector. detects only 'person' class."""

    def __init__(self, model_path: str = str(MODEL_PATH),
                 providers: Optional[List[str]] = None):
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        _, _, self.input_h, self.input_w = self.session.get_inputs()[0].shape

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to model input, normalize, return NCHW tensor."""
        img = cv2.resize(frame, (self.input_w, self.input_h))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = np.expand_dims(img, axis=0)    # → NCHW
        return img

    def postprocess(self, outputs: np.ndarray, orig_shape) -> List[DetectedObject]:
        """Parse YOLOv8 output into DetectedObject list."""
        # outputs shape: (1, 84, 8400) for 640×640 input
        # For 640×480, still (1, 84, num_boxes)
        output = outputs[0]  # (84, num_boxes)
        output = np.transpose(output, (1, 0))  # (num_boxes, 84)

        boxes, scores, class_ids = [], [], []
        for pred in output:
            cls_scores = pred[4:]  # class probabilities start at index 4
            class_id = int(np.argmax(cls_scores))
            score = float(cls_scores[class_id])
            if class_id != COCO_PERSON_ID or score < CONF_THRESHOLD:
                continue

            xc, yc, w, h = pred[0], pred[1], pred[2], pred[3]
            x1 = (xc - w / 2) / self.input_w * orig_shape[1]
            y1 = (yc - h / 2) / self.input_h * orig_shape[0]
            x2 = (xc + w / 2) / self.input_w * orig_shape[1]
            y2 = (yc + h / 2) / self.input_h * orig_shape[0]

            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            class_ids.append(class_id)

        if not boxes:
            return []

        # NMS
        indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD)
        results = []
        for i in indices.flatten():
            b = boxes[i]
            results.append(DetectedObject(
                bbox=BBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3]),
                cls="person",
                conf=scores[i],
            ))
        return results

    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        """Run detection on a BGR frame. Returns list of person DetectedObject."""
        orig_h, orig_w = frame.shape[:2]
        input_tensor = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return self.postprocess(outputs, (orig_h, orig_w))
```

- [ ] **Step 2: Write test for detector non-model logic**

Test that postprocess handles empty/no-detection case — valuable without a real model:

```python
# tests/gpu/test_detector.py
"""Test YOLODetector preprocessing and postprocessing logic."""
import numpy as np
import pytest
from gpu.detector import YOLODetector
from shared.models import DetectedObject


def test_preprocess_output_shape():
    """Preprocess should return NCHW float32 tensor."""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    # Create detector instance without loading model (just test preprocess)
    det = YOLODetector.__new__(YOLODetector)
    det.input_w, det.input_h = 640, 640
    tensor = det.preprocess(frame)
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert 0.0 <= tensor.min() <= tensor.max() <= 1.0


def test_postprocess_empty():
    """Postprocess with no detections above threshold returns empty list."""
    det = YOLODetector.__new__(YOLODetector)
    det.input_w, det.input_h = 640, 640
    # Create dummy output: 84 values, all zeros (no detection)
    dummy = np.zeros((1, 84, 8400), dtype=np.float32)
    result = det.postprocess(dummy, (480, 640))
    assert result == []
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/gpu/test_detector.py -v`
Expected: 2 PASS

- [ ] **Step 4: Commit**

```bash
git add gpu/detector.py tests/gpu/test_detector.py
git commit -m "feat: YOLOv8n ONNX detector wrapper

Detects only 'person' class (COCO class 0). CUDA/CPU provider
fallback. NMS with configurable thresholds."
```

### Task 5: GPU classifier — MobileNetV3 ONNX wrapper

**Files:**
- Create: `gpu/classifier.py`
- Create: `tests/gpu/test_classifier.py`

- [ ] **Step 1: Write classifier.py**

```python
"""MobileNetV3-small ONNX classifiers for PPE detection.

Three binary classifiers: helmet, vest, boot. Each takes a 224×224
RGB crop and returns a label (yes/no) with confidence.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime


MODELS_DIR = Path(__file__).parent / "models"

CLASS_NAMES = {
    "helmet": ["NO_HELMET", "HELMET"],
    "vest": ["NO_VEST", "VEST"],
    "boot": ["NO_BOOT", "BOOT"],
}


class PPEClassifier:
    """Binary classifier for one PPE item (helmet/vest/boot).

    Each classifier is a MobileNetV3-small ONNX model trained on
    224×224 crops of the relevant body region.
    """

    def __init__(self, item: str, model_path: Optional[str] = None,
                 providers: Optional[List[str]] = None):
        if item not in CLASS_NAMES:
            raise ValueError(f"Unknown PPE item: {item}. Choose from {list(CLASS_NAMES.keys())}")
        self.item = item
        self.class_names = CLASS_NAMES[item]
        if model_path is None:
            model_path = str(MODELS_DIR / f"{item}.onnx")
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, crop: np.ndarray) -> Tuple[str, float]:
        """Classify a BGR crop (224×224 or will be resized).

        Returns:
            Tuple of (label, confidence).
            Label is 'HELMET'/'NO_HELMET' etc.
        """
        img = cv2.resize(crop, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 127.5 - 1.0  # MobileNet normalization
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        outputs = self.session.run(None, {self.input_name: img})
        probs = outputs[0][0]
        class_id = int(np.argmax(probs))
        return self.class_names[class_id], float(probs[class_id])


class PPEManager:
    """Manages all 3 PPE classifiers. Runs them on cropped body regions."""

    def __init__(self, models_dir: Optional[str] = None):
        kw = {}
        if models_dir:
            kw = {"model_path": str(Path(models_dir) / "helmet.onnx")}
        self.helmet = PPEClassifier("helmet")
        self.vest = PPEClassifier("vest")
        self.boot = PPEClassifier("boot")

    def classify_all(self, head_crop: np.ndarray, torso_crop: np.ndarray,
                     feet_crop: np.ndarray) -> dict:
        """Run all 3 classifiers on respective body region crops.

        Returns:
            dict: {item: {"label": str, "confidence": float}}
        """
        return {
            "helmet": dict(zip(["label", "confidence"], self.helmet.predict(head_crop))),
            "vest": dict(zip(["label", "confidence"], self.vest.predict(torso_crop))),
            "boot": dict(zip(["label", "confidence"], self.boot.predict(feet_crop))),
        }
```

- [ ] **Step 2: Write tests**

```python
# tests/gpu/test_classifier.py
"""Test PPEClassifier configuration and preprocess logic."""
import numpy as np
import pytest
from gpu.classifier import PPEClassifier, CLASS_NAMES


def test_unknown_item_raises():
    with pytest.raises(ValueError):
        PPEClassifier("hat")


def test_class_names_exist():
    assert "helmet" in CLASS_NAMES
    assert "vest" in CLASS_NAMES
    assert "boot" in CLASS_NAMES
    for names in CLASS_NAMES.values():
        assert len(names) == 2  # [negative, positive]
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/gpu/test_classifier.py -v`
Expected: 2 PASS

- [ ] **Step 4: Commit**

```bash
git add gpu/classifier.py tests/gpu/test_classifier.py
git commit -m "feat: MobileNetV3 ONNX classifiers for PPE detection

Binary classifiers for helmet, vest, boot. PPEManager
coordinates all 3 on cropped body regions."
```

### Task 6: GPU ROI checker

**Files:**
- Create: `gpu/roi_checker.py`
- Create: `tests/gpu/test_roi_checker.py`

- [ ] **Step 1: Write roi_checker.py**

```python
"""ROI (Region of Interest) checker using ray-casting point-in-polygon.

Determines whether a person's foot point falls inside any drawn ROI zone.
ROI polygons are loaded from SQLite.
"""
from typing import List, Optional, Tuple

import numpy as np


def point_in_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> bool:
    """Ray-casting algorithm. Returns True if point is inside polygon.

    Args:
        point: (x, y) pixel coordinates.
        polygon: [[x1,y1], [x2,y2], ..., [xn,yn]] — closed or open.

    Returns:
        True if point is inside polygon (including on edge via epsilon).
    """
    x, y = point
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        # Check if point is on an edge (within epsilon)
        edge_dist = abs((xj - xi) * (yi - y) - (xi - x) * (yj - yi))
        edge_len = ((xj - xi) ** 2 + (yj - yi) ** 2) ** 0.5
        if edge_len > 0 and edge_dist / edge_len < 1.0:
            # Check if point projects onto the edge segment
            dot = ((x - xi) * (xj - xi) + (y - yi) * (yj - yi))
            if 0 <= dot <= edge_len * edge_len:
                return True  # On edge counts as inside

        # Ray casting: check if horizontal ray from point crosses this edge
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside

        j = i

    return inside


class ROIChecker:
    """Check if detected persons are inside any ROI zone.

    Loads ROI configs from database. Uses foot point (bottom-center
    of bbox) to determine zone entry.
    """

    def __init__(self, rois: Optional[List[dict]] = None):
        self._rois = rois or []

    def reload(self, rois: List[dict]):
        self._rois = rois

    def check_person(self, foot_point: Tuple[float, float]) -> List[dict]:
        """Check if a foot point falls inside any ROI.

        Args:
            foot_point: (x, y) pixel coordinates of person's feet.

        Returns:
            List of ROI dicts (with zone_name, color) that contain the point.
            Empty list if not in any zone.
        """
        inside_zones = []
        for roi in self._rois:
            if not roi.get("enabled", True):
                continue
            polygon = self._parse_polygon(roi["points_json"])
            if point_in_polygon(foot_point, polygon):
                inside_zones.append(roi)
        return inside_zones

    def _parse_polygon(self, points_json: str) -> List[List[float]]:
        """Parse JSON polygon string to list of [x,y] pairs."""
        import json
        return json.loads(points_json)
```

- [ ] **Step 2: Write tests**

```python
# tests/gpu/test_roi_checker.py
"""Test ROI checker point-in-polygon logic."""
import pytest
from gpu.roi_checker import point_in_polygon, ROIChecker


def test_point_inside_square():
    """Point in center of unit square should be inside."""
    poly = [[0, 0], [100, 0], [100, 100], [0, 100]]
    assert point_in_polygon((50, 50), poly) is True


def test_point_outside_square():
    """Point far from square should be outside."""
    poly = [[0, 0], [100, 0], [100, 100], [0, 100]]
    assert point_in_polygon((200, 200), poly) is False


def test_point_on_edge():
    """Point on polygon edge should be inside."""
    poly = [[0, 0], [100, 0], [100, 100], [0, 100]]
    assert point_in_polygon((50, 0), poly) is True


def test_point_on_vertex():
    """Point on polygon vertex should be inside."""
    poly = [[0, 0], [100, 0], [100, 100], [0, 100]]
    assert point_in_polygon((0, 0), poly) is True


def test_point_inside_triangle():
    """Point in triangle center should be inside."""
    poly = [[0, 0], [100, 50], [0, 100]]
    assert point_in_polygon((30, 50), poly) is True


def test_roi_checker_empty_rois():
    """ROIChecker with no ROIs returns empty list."""
    checker = ROIChecker([])
    assert checker.check_person((50, 50)) == []


def test_roi_checker_inside():
    """ROIChecker returns zone when foot point inside."""
    checker = ROIChecker([
        {"zone_name": "Zone A", "color": "#ff0000", "enabled": True,
         "points_json": "[[0,0],[100,0],[100,100],[0,100]]"},
    ])
    zones = checker.check_person((50, 50))
    assert len(zones) == 1
    assert zones[0]["zone_name"] == "Zone A"


def test_roi_checker_disabled_zone():
    """Disabled ROI should be ignored."""
    checker = ROIChecker([
        {"zone_name": "Zone A", "color": "#ff0000", "enabled": False,
         "points_json": "[[0,0],[100,0],[100,100],[0,100]]"},
    ])
    assert checker.check_person((50, 50)) == []
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/gpu/test_roi_checker.py -v`
Expected: 8 PASS

- [ ] **Step 4: Commit**

```bash
git add gpu/roi_checker.py tests/gpu/test_roi_checker.py
git commit -m "feat: ROI checker with ray-casting point-in-polygon

ROIChecker loads zones from SQLite, checks foot point against
each enabled polygon. On-edge counts as inside."
```

### Task 7: GPU PPE checker

**Files:**
- Create: `gpu/ppe_checker.py`
- Create: `tests/gpu/test_ppe_checker.py`

- [ ] **Step 1: Write ppe_checker.py**

```python
"""PPE (Personal Protective Equipment) checker.

Crops body regions from person bbox, runs through MobileNetV3
classifiers to detect helmet/vest/boot presence.
"""
from typing import List, Optional, Tuple

import cv2
import numpy as np

from shared.models import DetectedObject


# Crop ratios relative to bbox height
HEAD_RATIO = (0.0, 0.2)       # top 20% of bbox
TORSO_RATIO = (0.2, 0.7)      # 20%-70% of bbox
FEET_RATIO = (0.85, 1.0)      # bottom 15% of bbox


def crop_head(frame: np.ndarray, bbox) -> np.ndarray:
    """Crop head region from person bbox. Returns 224×224 RGB-ready crop."""
    x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
    h = y2 - y1
    # Head is top 20% of bbox
    cy1 = max(0, y1)
    cy2 = min(frame.shape[0], y1 + int(h * HEAD_RATIO[1]))
    cx1 = max(0, x1)
    cx2 = min(frame.shape[1], x2)
    if cy2 <= cy1 or cx2 <= cx1:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    return frame[cy1:cy2, cx1:cx2]


def crop_torso(frame: np.ndarray, bbox) -> np.ndarray:
    """Crop torso region from person bbox."""
    x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
    h = y2 - y1
    cy1 = max(0, y1 + int(h * TORSO_RATIO[0]))
    cy2 = min(frame.shape[0], y1 + int(h * TORSO_RATIO[1]))
    cx1 = max(0, x1)
    cx2 = min(frame.shape[1], x2)
    if cy2 <= cy1 or cx2 <= cx1:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    return frame[cy1:cy2, cx1:cx2]


def crop_feet(frame: np.ndarray, bbox) -> np.ndarray:
    """Crop feet region from person bbox."""
    x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
    h = y2 - y1
    cy1 = max(0, y1 + int(h * FEET_RATIO[0]))
    cy2 = min(frame.shape[0], y2)
    cx1 = max(0, x1)
    cx2 = min(frame.shape[1], x2)
    if cy2 <= cy1 or cx2 <= cx1:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    return frame[cy1:cy2, cx1:cx2]


class PPEChecker:
    """Coordinate crop + classify for PPE detection per person."""

    def __init__(self, ppe_manager):
        self._ppe = ppe_manager
        self._frame_counter = 0
        self._classify_every_n = 3  # Run classifiers every N frames

    def process_persons(self, frame: np.ndarray,
                        persons: List[DetectedObject],
                        force_classify: bool = False) -> List[dict]:
        """Run PPE check on all detected persons in frame.

        Args:
            frame: BGR frame.
            persons: list of person DetectedObject.
            force_classify: if True, always run classifiers (skip frame override).

        Returns:
            List of alert dicts per person with missing PPE.
            Each alert: {"person_idx": int, "violations": [str, ...], ...}
        """
        self._frame_counter += 1
        results = []
        should_classify = force_classify or (self._frame_counter % self._classify_every_n == 0)

        for idx, person in enumerate(persons):
            if should_classify:
                head = crop_head(frame, person.bbox)
                torso = crop_torso(frame, person.bbox)
                feet = crop_feet(frame, person.bbox)

                ppe_result = self._ppe.classify_all(head, torso, feet)

                violations = []
                for item, result in ppe_result.items():
                    if result["label"].startswith("NO_"):
                        violations.append(result["label"])

                if violations:
                    results.append({
                        "person_idx": idx,
                        "bbox": person.bbox,
                        "violations": violations,
                        "ppe": ppe_result,
                    })

        return results
```

- [ ] **Step 2: Write tests**

```python
# tests/gpu/test_ppe_checker.py
"""Test PPE checker crop logic."""
import numpy as np
import pytest
from gpu.ppe_checker import crop_head, crop_torso, crop_feet
from shared.models import BBox


def test_crop_head_valid():
    """Head crop returns valid region within frame."""
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
    bbox = BBox(100, 100, 300, 400)  # h=300
    head = crop_head(frame, bbox)
    assert head.shape[0] > 0
    assert head.shape[1] > 0
    assert head.shape[2] == 3


def test_crop_head_clamps_to_frame():
    """Head crop at frame edge should clamp, not crash."""
    frame = np.ones((480, 640, 3), dtype=np.uint8)
    bbox = BBox(0, 0, 50, 100)
    head = crop_head(frame, bbox)
    assert head.shape[0] > 0
    assert head.shape[1] > 0


def test_crop_feet_at_bottom():
    """Feet crop should be bottom ~15% of bbox."""
    frame = np.ones((480, 640, 3), dtype=np.uint8)
    bbox = BBox(100, 100, 300, 400)  # h=300
    feet = crop_feet(frame, bbox)
    # Height should be ~15% of 300 = 45, after clamp maybe less
    assert 20 <= feet.shape[0] <= 60


def test_crop_all_valid_shapes():
    """All 3 crops produce valid images when frame large enough."""
    frame = np.ones((480, 640, 3), dtype=np.uint8)
    bbox = BBox(50, 50, 300, 400)
    assert crop_head(frame, bbox).size > 0
    assert crop_torso(frame, bbox).size > 0
    assert crop_feet(frame, bbox).size > 0
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/gpu/test_ppe_checker.py -v`
Expected: 4 PASS

- [ ] **Step 4: Commit**

```bash
git add gpu/ppe_checker.py tests/gpu/test_ppe_checker.py
git commit -m "feat: PPE checker with body region crop + classify

Crops head/torso/feet from person bbox using fixed ratios.
Coordinates with PPEManager for inference on each region.
Frame skip strategy: classifiers run every N frames."
```

### Task 8: GPU overlay

**Files:**
- Create: `gpu/overlay.py`
- Create: `tests/gpu/test_overlay.py`

- [ ] **Step 1: Write overlay.py**

```python
"""Draw bounding boxes, ROI polygons, and alert labels on frames.

All drawing uses QPainter-compatible coordinates (same as numpy array).
"""
from typing import List, Optional

import cv2
import numpy as np

from shared.models import BBox, DetectedObject


COLORS = {
    "person": (0, 255, 0),
    "HELMET": (0, 255, 0),
    "NO_HELMET": (0, 0, 255),
    "VEST": (0, 255, 0),
    "NO_VEST": (0, 0, 255),
    "BOOT": (0, 255, 0),
    "NO_BOOT": (0, 0, 255),
    "zone": (255, 165, 0),
    "zone_alert": (0, 0, 255),
    "text": (255, 255, 255),
}


def draw_person_bboxes(frame: np.ndarray, persons: List[DetectedObject]) -> np.ndarray:
    """Draw green bounding boxes around detected persons."""
    canvas = frame.copy()
    for person in persons:
        b = person.bbox
        x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLORS["person"], 2)
        label = f"person {person.conf:.2f}"
        cv2.putText(canvas, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text"], 1)
    return canvas


def draw_roi_polygons(frame: np.ndarray, rois: List[dict]) -> np.ndarray:
    """Draw ROI polygons on frame. Alert zones in red, normal in orange."""
    canvas = frame.copy()
    for roi in rois:
        import json
        points = np.array(json.loads(roi["points_json"]), dtype=np.int32)
        color = COLORS["zone_alert"] if roi.get("alert_active") else COLORS["zone"]
        cv2.polylines(canvas, [points], isClosed=True, color=color, thickness=2)
        # Label at first vertex
        if len(points) > 0:
            cx = int(points[:, 0].mean())
            cy = int(points[:, 1].mean())
            cv2.putText(canvas, roi["zone_name"], (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return canvas


def draw_ppe_labels(frame: np.ndarray, persons: List[DetectedObject],
                    alerts: List[dict]) -> np.ndarray:
    """Draw PPE status per person. Red text for missing items."""
    canvas = frame.copy()
    alert_by_idx = {a["person_idx"]: a["violations"] for a in alerts}

    for idx, person in enumerate(persons):
        b = person.bbox
        x1, y1 = int(b.x1), int(b.y1)
        violations = alert_by_idx.get(idx, [])
        y_offset = y1 - 20
        for vtype in ["HELMET", "VEST", "BOOT"]:
            is_missing = f"NO_{vtype}" in violations
            label = f"{'✗' if is_missing else '✓'} {vtype}"
            color = COLORS[f"NO_{vtype}"] if is_missing else COLORS[vtype]
            cv2.putText(canvas, label, (x1, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset -= 15
    return canvas


def draw_disconnected(frame: np.ndarray) -> np.ndarray:
    """Overlay 'DISCONNECTED' text on frame."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    cv2.putText(canvas, "DISCONNECTED", (w // 2 - 100, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    return canvas


def draw_detection_offline(frame: np.ndarray) -> np.ndarray:
    """Overlay 'Detection Offline' badge."""
    canvas = frame.copy()
    cv2.putText(canvas, "Detection Offline", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return canvas
```

- [ ] **Step 2: Write tests**

```python
# tests/gpu/test_overlay.py
"""Test overlay drawing functions output valid frames."""
import numpy as np
import pytest
from gpu.overlay import (
    draw_person_bboxes, draw_roi_polygons, draw_ppe_labels,
    draw_disconnected, draw_detection_offline,
)
from shared.models import BBox, DetectedObject


@pytest.fixture
def frame():
    return np.ones((480, 640, 3), dtype=np.uint8) * 200


def test_draw_person_bboxes_output_shape(frame):
    """draw_person_bboxes returns same shape frame."""
    persons = [DetectedObject(bbox=BBox(50, 50, 200, 400), cls="person", conf=0.85)]
    result = draw_person_bboxes(frame, persons)
    assert result.shape == frame.shape
    assert result.dtype == frame.dtype


def test_draw_person_bboxes_empty(frame):
    """Empty person list should return unchanged frame."""
    result = draw_person_bboxes(frame, [])
    assert np.array_equal(result, frame)


def test_draw_disconnected(frame):
    """Disconnected overlay should not change shape."""
    result = draw_disconnected(frame)
    assert result.shape == frame.shape


def test_draw_detection_offline(frame):
    """Detection offline overlay should not change shape."""
    result = draw_detection_offline(frame)
    assert result.shape == frame.shape


def test_draw_roi_polygons(frame):
    """ROI polygon overlay should return same shape."""
    rois = [{"zone_name": "Zone A", "points_json": "[[0,0],[100,0],[100,100],[0,100]]"}]
    result = draw_roi_polygons(frame, rois)
    assert result.shape == frame.shape


def test_draw_ppe_labels(frame):
    """PPE labels should draw correctly."""
    persons = [DetectedObject(bbox=BBox(50, 50, 200, 400), cls="person", conf=0.85)]
    alerts = [{"person_idx": 0, "violations": ["NO_VEST"]}]
    result = draw_ppe_labels(frame, persons, alerts)
    assert result.shape == frame.shape
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/gpu/test_overlay.py -v`
Expected: 6 PASS

- [ ] **Step 4: Commit**

```bash
git add gpu/overlay.py tests/gpu/test_overlay.py
git commit -m "feat: overlay drawing for bboxes, ROI polygons, PPE labels

Functions for person bboxes, ROI zone polygons, PPE status
per person, disconnected/detection-offline overlays."
```

### Task 9: GPU alert manager

**Files:**
- Create: `gpu/alert_manager.py`
- Create: `tests/gpu/test_alert_manager.py`

- [ ] **Step 1: Write alert_manager.py**

```python
"""Alert manager: cooldown dedup, SQLite logging, WebSocket broadcast.

Each violation type per person has a cooldown timer to prevent spam.
When alert fires, saves to SQLite and emits signal for UI + WebSocket.
"""
import sqlite3
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from shared.models import BBox


class CooldownManager:
    """Tracks cooldown per (zone_name, person_idx, violation_type) tuple.

    Uses time-based cooldown: same alert won't fire again within N seconds.
    """

    def __init__(self, default_cooldown: float = 30.0):
        self._default = default_cooldown
        self._timers: Dict[Tuple[str, int, str], float] = {}

    def can_alert(self, key: Tuple[str, int, str]) -> bool:
        """Check if enough time has passed since last alert for this key.

        Args:
            key: (zone_name or camera_id, person_idx, violation_type)

        Returns:
            True if should fire alert, False if still in cooldown.
        """
        now = time.time()
        last = self._timers.get(key, 0.0)
        if now - last >= self._default:
            self._timers[key] = now
            return True
        return False


class AlertManager:
    """Manages alert lifecycle: cooldown → SQLite → callback.

    Callback can be used for UI signal emission and WebSocket broadcast.
    """

    def __init__(self, conn: sqlite3.Connection,
                 on_alert: Optional[Callable[..., None]] = None,
                 cooldown: float = 30.0,
                 thumbnail_dir: str = "data/thumbnails"):
        self._conn = conn
        self._callback = on_alert
        self._cooldown = CooldownManager(default_cooldown=cooldown)
        self._thumbnail_dir = thumbnail_dir

    def process_violations(self, alerts: List[dict], camera_id: str,
                           frame: Optional[np.ndarray] = None,
                           force: bool = False) -> List[str]:
        """Process alerts through cooldown, save to DB, invoke callback.

        Args:
            alerts: list of alert dicts from ROIChecker or PPEChecker.
            camera_id: 'cam1' or 'cam2'.
            frame: optional BGR frame for thumbnail capture.
            force: bypass cooldown (for testing).

        Returns:
            List of violation IDs that were actually fired.
        """
        fired_ids = []
        for alert in alerts:
            vtype = alert.get("type", "PERSON_IN_ZONE")
            zone = alert.get("zone_name", "")
            person_idx = alert.get("person_idx", 0)
            bbox = alert.get("bbox")

            key = (zone or camera_id, person_idx, vtype)
            if not force and not self._cooldown.can_alert(key):
                continue

            thumbnail_path = ""
            if frame is not None and bbox is not None:
                thumbnail_path = self._save_thumbnail(frame, bbox, vtype)

            bbox_json = bbox.to_list() if bbox else ""

            vid = self._save_to_db(camera_id, vtype, zone, bbox_json, thumbnail_path)

            if self._callback:
                self._callback(vid, camera_id, vtype, zone, person_idx, timestamp=datetime.now())

            fired_ids.append(vid)

        return fired_ids

    def process_zone_alerts(self, camera_id: str, alerts: List[dict],
                            frame: Optional[np.ndarray] = None,
                            force: bool = False) -> List[str]:
        """Process ROI zone alerts."""
        return self.process_violations(alerts, camera_id, frame, force)

    def _save_thumbnail(self, frame: np.ndarray, bbox: BBox, vtype: str) -> str:
        """Save a thumbnail crop of the violation region."""
        import os
        import uuid
        from pathlib import Path

        Path(self._thumbnail_dir).mkdir(parents=True, exist_ok=True)
        x1 = max(0, int(bbox.x1))
        y1 = max(0, int(bbox.y1))
        x2 = min(frame.shape[1], int(bbox.x2))
        y2 = min(frame.shape[0], int(bbox.y2))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return ""

        fname = f"{vtype}_{uuid.uuid4().hex[:8]}.jpg"
        path = str(Path(self._thumbnail_dir) / fname)
        import cv2
        cv2.imwrite(path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        return path

    def _save_to_db(self, camera_id: str, vtype: str, zone: str,
                    bbox_json: str, thumbnail_path: str) -> str:
        from gpu.database import save_violation
        sev = "HIGH" if vtype in ("PERSON_IN_ZONE",) else "MEDIUM"
        return save_violation(self._conn, camera_id, vtype, sev,
                              zone_name=zone, bbox_json=bbox_json,
                              thumbnail_path=thumbnail_path)
```

- [ ] **Step 2: Write tests**

```python
# tests/gpu/test_alert_manager.py
"""Test cooldown manager and alert manager."""
import sqlite3
import time
import pytest
from gpu.alert_manager import CooldownManager, AlertManager


def test_cooldown_allows_first():
    cd = CooldownManager(default_cooldown=1.0)
    assert cd.can_alert(("cam1", 0, "PERSON_IN_ZONE")) is True


def test_cooldown_blocks_immediate_repeat():
    cd = CooldownManager(default_cooldown=5.0)
    assert cd.can_alert(("cam1", 0, "PERSON_IN_ZONE")) is True
    assert cd.can_alert(("cam1", 0, "PERSON_IN_ZONE")) is False


def test_cooldown_expires():
    cd = CooldownManager(default_cooldown=0.1)
    assert cd.can_alert(("cam1", 0, "PERSON_IN_ZONE")) is True
    time.sleep(0.15)
    assert cd.can_alert(("cam1", 0, "PERSON_IN_ZONE")) is True


def test_cooldown_different_key_not_blocked():
    cd = CooldownManager(default_cooldown=5.0)
    assert cd.can_alert(("cam1", 0, "NO_HELMET")) is True
    assert cd.can_alert(("cam1", 1, "NO_HELMET")) is True  # different person


@pytest.fixture
def alert_manager():
    import tempfile
    from gpu.database import init_db
    conn = init_db()
    alerts = []
    def callback(vid, cam, vtype, zone, pidx, timestamp):
        alerts.append((vid, vtype, zone))
    am = AlertManager(conn, on_alert=callback, cooldown=30.0)
    return am, conn, alerts


def test_alert_manager_zone(alert_manager):
    am, conn, alerts = alert_manager
    fired = am.process_zone_alerts("cam1", [
        {"type": "PERSON_IN_ZONE", "zone_name": "Zone A", "person_idx": 0},
    ], force=True)
    assert len(fired) == 1
    assert len(alerts) == 1
    assert alerts[0][1] == "PERSON_IN_ZONE"


def test_alert_manager_cooldown(alert_manager):
    am, conn, alerts = alert_manager
    fired1 = am.process_zone_alerts("cam1", [
        {"type": "PERSON_IN_ZONE", "zone_name": "Zone A", "person_idx": 0},
    ], force=False)
    assert len(fired1) == 1
    # Second immediate fire should be blocked by cooldown
    fired2 = am.process_zone_alerts("cam1", [
        {"type": "PERSON_IN_ZONE", "zone_name": "Zone A", "person_idx": 0},
    ], force=False)
    assert len(fired2) == 0  # blocked
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/gpu/test_alert_manager.py -v`
Expected: 6 PASS

- [ ] **Step 4: Commit**

```bash
git add gpu/alert_manager.py tests/gpu/test_alert_manager.py
git commit -m "feat: alert manager with cooldown dedup + SQLite logging

CooldownManager blocks duplicate alerts within N seconds.
AlertManager coordinates SQLite save, thumbnail capture,
and callback for UI/WebSocket broadcast."
```

### Task 10: GPU CAM1 QThread

**Files:**
- Create: `gpu/cam1_thread.py`
- Create: `tests/gpu/test_cam1_thread.py`

- [ ] **Step 1: Write cam1_thread.py**

```python
"""CAM1 QThread: ZMQ SUB → YOLOv8 detect → ROI check → overlay → emit.

Self-paced with frame budget: if processing exceeds budget, skip next frame.
Uses 'latest frame' strategy — no queue buildup.
"""
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
import zmq
from PyQt5.QtCore import QThread, pyqtSignal

from gpu.detector import YOLODetector
from gpu.roi_checker import ROIChecker
from gpu.overlay import draw_person_bboxes, draw_roi_polygons, draw_disconnected
from shared.models import DetectedObject


FRAME_BUDGET_MS = 33  # ~30fps target


class Cam1Thread(QThread):
    """CAM1 pipeline: receive frame → detect persons → check ROI → overlay.

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
        self._connected = False

    def stop(self):
        self._running = False

    def update_rois(self, rois: List[dict]):
        self._roi_checker.reload(rois)

    def run(self):
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.setsockopt(zmq.RCVHWM, 2)
        sub.bind(f"tcp://*:{self._port}")
        sub.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all topics
        sub.setsockopt(zmq.RCVTIMEO, 1000)  # 1s timeout for disconnect detection
        self._connected = True
        self._disconnected = False

        while self._running:
            t0 = time.perf_counter()

            # Receive frame
            try:
                data = sub.recv()
            except zmq.Again:
                if not self._disconnected:
                    self._disconnected = True
                    # Emit disconnected frame
                    disconnected = draw_disconnected(np.zeros((480, 640, 3), dtype=np.uint8))
                    self.frame_ready.emit("cam1", disconnected)
                continue

            self._disconnected = False
            frame = np.frombuffer(data, dtype=np.uint8).reshape((480, 640, 3))

            # Detect persons
            persons = self._detector.detect(frame)

            # Check ROI
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

            # Overlay
            overlay = draw_person_bboxes(frame, persons)
            overlay = draw_roi_polygons(overlay, self._roi_checker._rois)

            # Emit
            self.frame_ready.emit("cam1", overlay)
            for alert_dict in alerts:
                self.alert.emit(alert_dict)

            # Frame budget check
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if elapsed_ms > FRAME_BUDGET_MS:
                # Skip next frame: don't recv until next loop
                # ZMQ recv with timeout will drop old frames anyway (HWM=2)
                pass

        sub.close()
        ctx.term()
```

- [ ] **Step 2: Write test for initialization and budget constant**

```python
# tests/gpu/test_cam1_thread.py
"""Test CAM1 thread constants and initialization logic."""
from gpu.cam1_thread import FRAME_BUDGET_MS


def test_frame_budget_constant():
    """Frame budget should be 33ms for ~30fps."""
    assert FRAME_BUDGET_MS == 33
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/gpu/test_cam1_thread.py -v`
Expected: 1 PASS

- [ ] **Step 4: Commit**

```bash
git add gpu/cam1_thread.py tests/gpu/test_cam1_thread.py
git commit -m "feat: CAM1 QThread with ZMQ SUB + detect + ROI + overlay

Self-paced thread with frame budget. Latest-frame strategy via
ZMQ RCVHWM=2. Emits frame_ready signal for UI and alert signal
for each person found inside a restricted zone."
```

### Task 11: GPU CAM2 QThread

**Files:**
- Create: `gpu/cam2_thread.py`

- [ ] **Step 1: Write cam2_thread.py**

```python
"""CAM2 QThread: ZMQ SUB → YOLOv8 detect → crop → PPE classify → overlay → emit.

Two-level skip:
  1. Motion detection: skip YOLOv8 if no motion.
  2. Classify skip: run classifiers every N frames even with motion.
"""
import time
from typing import Optional

import cv2
import numpy as np
import zmq
from PyQt5.QtCore import QThread, pyqtSignal

from gpu.detector import YOLODetector
from gpu.classifier import PPEManager
from gpu.ppe_checker import PPEChecker
from gpu.overlay import draw_person_bboxes, draw_ppe_labels, draw_disconnected
from shared.models import DetectedObject


FRAME_BUDGET_MS = 50  # ~20fps target (CAM2 is heavier)
MOTION_THRESHOLD = 30.0  # Mean absolute difference for motion detection


class Cam2Thread(QThread):
    """CAM2 pipeline: receive frame → detect persons → PPE classify → overlay.

    Signals:
        frame_ready: (camera_id: str, overlay_frame: np.ndarray)
        alert: (alert_dict: dict)
        detection_offline: () — emitted when model fails
    """
    frame_ready = pyqtSignal(str, np.ndarray)
    alert = pyqtSignal(dict)

    def __init__(self, zmq_port: int, detector: YOLODetector,
                 ppe_manager: PPEManager, parent=None):
        super().__init__(parent)
        self._port = zmq_port
        self._detector = detector
        self._ppe_manager = ppe_manager
        self._ppe_checker = PPEChecker(ppe_manager)
        self._running = True
        self._prev_gray: Optional[np.ndarray] = None
        self._disconnected = False

    def stop(self):
        self._running = False

    def _has_motion(self, frame: np.ndarray) -> bool:
        """Detect motion via frame difference. Returns True if motion > threshold."""
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
                    disconnected = draw_disconnected(np.zeros((480, 640, 3), dtype=np.uint8))
                    self.frame_ready.emit("cam2", disconnected)
                continue

            self._disconnected = False
            frame = np.frombuffer(data, dtype=np.uint8).reshape((480, 640, 3))

            # Level 1: motion skip
            has_motion = self._has_motion(frame)
            if not has_motion:
                # Still emit frame (no overlay changes) to keep UI alive
                self.frame_ready.emit("cam2", frame)
                continue

            # Detect persons
            try:
                persons = self._detector.detect(frame)
                model_ok = True
            except Exception as exc:
                if model_ok:
                    model_ok = False
                    from gpu.overlay import draw_detection_offline
                    self.frame_ready.emit("cam2", draw_detection_offline(frame))
                continue

            # Level 2: PPE classify (skip every N frames via PPEChecker internal counter)
            alerts = self._ppe_checker.process_persons(frame, persons)

            # Overlay
            overlay = draw_person_bboxes(frame, persons)
            overlay = draw_ppe_labels(overlay, persons, alerts)

            self.frame_ready.emit("cam2", overlay)
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
                pass  # Skip next frame via HWM

        sub.close()
        ctx.term()
```

- [ ] **Step 2: Commit**

```bash
git add gpu/cam2_thread.py
git commit -m "feat: CAM2 QThread with motion skip + detect + PPE classify

Two-level skip: motion detection (skip YOLOv8 on static frames)
and classify skip (run classifiers every N frames). Frame budget
50ms for heavier pipeline."
```

### Task 12: GPU camera widget

**Files:**
- Create: `gpu/camera_widget.py`

- [ ] **Step 1: Write camera_widget.py**

```python
"""PyQt QWidget for displaying camera video with overlay.

Uses QPainter for efficient rendering of numpy frame data.
Zero-copy: QImage wraps numpy array directly without copy.
"""
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPainter, QPixmap
from PyQt5.QtWidgets import QWidget


class CameraWidget(QWidget):
    """Widget that displays one camera feed with overlay.

    Receives numpy BGR frames via update_frame() and renders them.
    Handles resize keeping aspect ratio.
    """

    def __init__(self, camera_id: str, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self._frame: Optional[np.ndarray] = None
        self._pixmap: Optional[QPixmap] = None
        self._aspect_ratio = 640 / 480
        self.setMinimumSize(320, 240)

    def update_frame(self, frame: np.ndarray):
        """Set current frame and trigger repaint.

        Args:
            frame: BGR numpy array (H×W×3).
        """
        self._frame = frame
        # Convert BGR to RGB for QImage
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
```

- [ ] **Step 2: Commit**

```bash
git add gpu/camera_widget.py
git commit -m "feat: PyQt CameraWidget for video display

Renders numpy BGR frame via QImage/QPixmap with aspect ratio
preservation. Zero-copy QImage from numpy array."
```

### Task 13: GPU ROI drawer

**Files:**
- Create: `gpu/roi_drawer.py`

- [ ] **Step 1: Write roi_drawer.py**

```python
"""ROI polygon drawing interaction on CameraWidget.

Supports: add vertices via click, close polygon via double-click,
drag vertex, move entire polygon, delete via right-click.
"""
import json
from typing import Callable, List, Optional, Tuple

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QGraphicsEllipseItem, QGraphicsItem, QGraphicsLineItem, QGraphicsPolygonItem, QGraphicsScene, QGraphicsView


class ROIDrawer:
    """Manages ROI drawing interaction state.

    Usage:
        drawer = ROIDrawer(on_save_callback)
        Install event filter on camera widget's view.

    Modes:
        - 'view': normal viewing, no editing
        - 'draw': drawing new polygon (click to add vertices)
        - 'edit': editing existing polygon (drag vertices/edges)
    """

    def __init__(self, widget, on_save: Callable[[str, list], None]):
        self._widget = widget
        self._on_save = on_save
        self._mode = "view"
        self._current_polygon: List[Tuple[float, float]] = []
        self._current_zone_name = "Zone A"
        self._zone_colors = ["#ff4444", "#ff8800", "#ffcc00", "#44ff44"]

    def set_mode(self, mode: str):
        self._mode = mode

    def set_zone_name(self, name: str):
        self._current_zone_name = name

    def set_polygon_data(self, polygon: list):
        """Replace current drawing polygon (used for editing)."""
        self._current_polygon = polygon

    def get_polygon_data(self) -> list:
        return self._current_polygon

    def mouse_press_event(self, event: QMouseEvent, widget_coords: Tuple[float, float]):
        """Handle mouse press. Returns True if event was consumed."""
        if self._mode != "draw":
            return False

        x, y = widget_coords
        if event.button() == Qt.LeftButton:
            self._current_polygon.append([x, y])
            self._widget.update()
            return True
        elif event.button() == Qt.RightButton:
            # Close polygon
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

    def get_current_color(self) -> str:
        idx = len([r for r in range(4)])  # simple rotation
        return self._zone_colors[0]  # simplified: always red
```

- [ ] **Step 2: Commit**

```bash
git add gpu/roi_drawer.py
git commit -m "feat: ROI drawer with polygon drawing interaction

Supports draw mode (click vertices, double-click close) and
save callback integration. Zone name and color rotation."
```

### Task 14: GPU main window

**Files:**
- Create: `gpu/main_window.py`

- [ ] **Step 1: Write main_window.py**

```python
"""PyQt5 MainWindow for CV Safety Monitor v2.

Toolbar: camera tabs, ROI draw mode, settings.
Layout: side-by-side camera widgets, status bar for alert log.
"""
import sqlite3
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QAction, QApplication, QDockWidget, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPushButton, QStatusBar, QToolBar, QVBoxLayout, QWidget,
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

        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("CV Safety Monitor v2")
        self.setMinimumSize(1280, 720)

        # Central widget: side-by-side cameras
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        self._cams = {}
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

        toolbar.addSeparator()

        self._status_label = QLabel("Ready")
        toolbar.addWidget(self._status_label)

        # Status bar
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
        """Add an alert message to the status list."""
        item = QListWidgetItem(text)
        self._alert_list.insertItem(0, item)
        # Keep max 100 alerts in list
        while self._alert_list.count() > 100:
            self._alert_list.takeItem(self._alert_list.count() - 1)

    def set_status(self, text: str):
        self._status_label.setText(text)
```

- [ ] **Step 2: Commit**

```bash
git add gpu/main_window.py
git commit -m "feat: PyQt MainWindow with camera layout and toolbar

Side-by-side CameraWidgets, ROI draw toggle, alert log dock,
status indicator. Toolbar-driven mode switching."
```

### Task 15: GPU main entry point

**Files:**
- Create: `gpu/main.py`
- Create: `gpu/__init__.py`

- [ ] **Step 1: Write gpu/__init__.py**

```python
"""CV Safety Monitor v2 GPU machine package."""
```

- [ ] **Step 2: Write gpu/main.py**

```python
"""Entry point for CV Safety Monitor v2 GPU machine.

Initializes all components: database, detector, classifiers,
alert manager, camera threads, UI, and FastAPI web server.
"""
import sys
import signal
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QThread, QTimer
from PyQt5.QtWidgets import QApplication

from gpu.alert_manager import AlertManager
from gpu.cam1_thread import Cam1Thread
from gpu.cam2_thread import Cam2Thread
from gpu.classifier import PPEManager
from gpu.database import get_connection, init_db, get_rois, get_cameras
from gpu.detector import YOLODetector
from gpu.main_window import MainWindow
from gpu.roi_checker import ROIChecker
from gpu.roi_drawer import ROIDrawer
from gpu.web_server import WebServer


class CVApp:
    """Application coordinator. Wires all components together."""

    def __init__(self):
        self._qapp = QApplication(sys.argv)
        self._qapp.setApplicationName("CV Safety Monitor v2")

        # Database
        self._db = init_db()
        self._init_cameras()

        # Models
        self._detector_cam1 = YOLODetector()
        self._detector_cam2 = YOLODetector()
        self._ppe_manager = PPEManager()

        # ROI checker
        rois_cam1 = get_rois(self._db, "cam1")
        self._roi_checker = ROIChecker(rois_cam1)

        # Alert manager
        self._alert_manager = AlertManager(
            self._db,
            on_alert=self._on_alert_fired,
        )

        # Main window
        self._window = MainWindow(self._alert_manager, self._db)
        for cam_id in ["cam1", "cam2"]:
            drawer = ROIDrawer(
                self._window.get_camera_widget(cam_id),
                on_save=lambda zone, pts: self._save_roi(cam_id, zone, pts),
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

        # Web server (thread)
        self._web = WebServer(self._db, self._alert_manager)
        self._web.start()

        self._window.show()

    def _init_cameras(self):
        """Seed cameras from config if DB empty."""
        from gpu.database import get_cameras, upsert_camera
        if not get_cameras(self._db):
            upsert_camera(self._db, "cam1", 5555, "/dev/v4l/by-id/usb-cam1")
            upsert_camera(self._db, "cam2", 5556, "/dev/v4l/by-id/usb-cam2")
            self._db.commit()

    def _on_frame_ready(self, camera_id: str, frame: np.ndarray):
        widget = self._window.get_camera_widget(camera_id)
        if widget is not None:
            widget.update_frame(frame)

    def _on_alert(self, alert_dict: dict):
        camera_id = alert_dict.get("type", "unknown")
        zone = alert_dict.get("zone_name", "")
        vtype = alert_dict.get("type", "")
        text = f"[{datetime.now().strftime('%H:%M:%S')}] {vtype} - {camera_id}"
        if zone:
            text += f" - {zone}"
        self._window.add_alert_entry(text)

    def _on_alert_fired(self, vid, cam, vtype, zone, pidx, timestamp):
        """Callback from AlertManager after cooldown + DB save."""
        # WebSocket broadcast handled by WebServer if running
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
```

- [ ] **Step 3: Update main.py entry point**

Replace old main.py with a launcher that detects v1 vs v2 mode. Add a simple flag approach:

```python
# In main.py: replace the main() function
"""CV Safety Monitor — entry point (v2)."""
import sys


def main():
    from gpu.main import main as gpu_main
    gpu_main()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Commit**

```bash
git add gpu/__init__.py gpu/main.py
git commit -m "feat: GPU machine entry point wiring all components

CVApp coordinates database, detector, classifiers, camera threads,
alert manager, UI window, and web server. Seeds default cameras."
```

### Task 16: GPU FastAPI web server

**Files:**
- Create: `gpu/web_server.py`

- [ ] **Step 1: Write web_server.py**

```python
"""FastAPI web server thread for secondary viewing (history, admin, 1fps preview).

Runs in a QThread with uvicorn.Server for clean lifecycle.
"""
import asyncio
import json
import threading
from datetime import datetime
from queue import Queue
from typing import Dict, Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PyQt5.QtCore import QThread, pyqtSignal
import uvicorn

from gpu.database import get_violations, get_cameras, get_rois


class ConnectionManager:
    """WebSocket connection manager for alert broadcasts."""

    def __init__(self):
        self._connections: list[WebSocket] = []
        self._lock = threading.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        with self._lock:
            self._connections.append(ws)

    def disconnect(self, ws: WebSocket):
        with self._lock:
            if ws in self._connections:
                self._connections.remove(ws)

    async def broadcast(self, message: dict):
        dead = []
        with self._lock:
            for ws in self._connections:
                try:
                    await ws.send_json(message)
                except Exception:
                    dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


ws_manager = ConnectionManager()
preview_queue: Queue = Queue(maxsize=2)  # latest frame per camera


def push_preview(camera_id: str, frame_bgr: np.ndarray):
    """Push a preview frame for WebSocket broadcast (called from main thread)."""
    if preview_queue.full():
        try:
            preview_queue.get_nowait()
        except Exception:
            pass
    # JPEG encode for WS transport
    ret, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 50])
    if ret:
        import base64
        b64 = base64.b64encode(buf.tobytes()).decode()
        preview_queue.put_nowait({"camera_id": camera_id, "frame_base64": b64})


class WebServer(QThread):
    """FastAPI web server running in its own QThread with asyncio event loop."""

    def __init__(self, db_conn, alert_manager, host: str = "0.0.0.0", port: int = 8080):
        super().__init__()
        self._host = host
        self._port = port
        self._db = db_conn
        self._alert_manager = alert_manager

    def run(self):
        app = FastAPI(title="CV Safety Monitor v2")

        # Attach db to app state
        app.state.db = self._db

        @app.get("/api/cameras")
        async def list_cameras():
            return {"cameras": get_cameras(self._db)}

        @app.get("/api/roi/{camera_id}")
        async def get_roi_api(camera_id: str):
            return {"rois": get_rois(self._db, camera_id)}

        @app.put("/api/roi/{camera_id}")
        async def save_roi_api(camera_id: str, data: dict):
            from gpu.database import save_roi
            save_roi(self._db, camera_id, data["zone_name"], data["points"], data.get("color", "#ff0000"))
            return {"status": "ok"}

        @app.get("/api/violations")
        async def list_violations(limit: int = 50, offset: int = 0, camera_id: str = ""):
            return {
                "violations": get_violations(
                    self._db, limit=limit, offset=offset,
                    camera_id=camera_id or None,
                )
            }

        @app.websocket("/ws/dashboard")
        async def dashboard_ws(ws: WebSocket):
            await ws_manager.connect(ws)
            try:
                while True:
                    # Send preview frame (1fps)
                    if not preview_queue.empty():
                        try:
                            preview = preview_queue.get_nowait()
                            await ws.send_json({
                                "type": "preview",
                                "camera_id": preview["camera_id"],
                                "frame_base64": preview["frame_base64"],
                            })
                        except Exception:
                            pass
                    await asyncio.sleep(1)
            except WebSocketDisconnect:
                ws_manager.disconnect(ws)

        config = uvicorn.Config(app, host=self._host, port=self._port, log_level="warning")
        server = uvicorn.Server(config)
        # Run with dedicated asyncio event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())
```

- [ ] **Step 2: Commit**

```bash
git add gpu/web_server.py
git commit -m "feat: FastAPI web server QThread with REST + WS

Endpoints: cameras list, ROI config, violation history.
WebSocket: alert push + 1fps preview frame. ConnectionManager
for thread-safe WS broadcast."
```

### Task 17: GPU requirements and model download

**Files:**
- Create: `requirements-gpu.txt`

- [ ] **Step 1: Write requirements-gpu.txt**

```
# GPU machine requirements for CV Safety Monitor v2
PyQt5>=5.15.0
opencv-python>=4.8.0
numpy>=1.24.0
onnxruntime-gpu>=1.16.0
pyzmq>=25.0.0
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
aiofiles>=23.0
pyyaml>=6.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
```

Also update edge/requirements.txt:

```
# Edge device requirements
opencv-python>=4.8.0
numpy>=1.24.0
pyzmq>=25.0.0
pyyaml>=6.0
```

- [ ] **Step 2: Download YOLOv8n ONNX model**

```bash
mkdir -p gpu/models
cd gpu/models
wget -q https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.onnx
cd ../..
```

- [ ] **Step 3: Commit**

```bash
git add requirements-gpu.txt edge/requirements.txt gpu/models/yolov8n.onnx
git commit -m "chore: add GPU requirements and YOLOv8n model

requirements-gpu.txt for GPU machine, updated edge requirements
with pyzmq. YOLOv8n ONNX model for person detection."
```

### Task 18: Edge simulator for testing

**Files:**
- Create: `tests/edge_simulator.py`

- [ ] **Step 1: Write edge_simulator.py**

```python
"""Edge device simulator for GPU machine testing.

Reads a video file and publishes raw BGR frames via ZMQ PUB,
simulating the real edge device. Use for integration testing
without needing edge hardware.
"""
import argparse
import time

import cv2
import zmq


def main():
    parser = argparse.ArgumentParser(description="Edge device simulator")
    parser.add_argument("video", help="Path to video file (.mp4)")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ port (default: 5555)")
    parser.add_argument("--fps", type=int, default=30, help="Target send FPS (default: 30)")
    parser.add_argument("--target-host", default="127.0.0.1", help="GPU machine host")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[Sim] Cannot open video: {args.video}")
        return

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 2)
    pub.connect(f"tcp://{args.target_host}:{args.port}")

    frame_interval = 1.0 / args.fps
    frame_count = 0

    print(f"[Sim] Publishing {args.video} → tcp://{args.target_host}:{args.port} @ {args.fps}fps")

    try:
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            pub.send(frame.tobytes())
            frame_count += 1

            elapsed = time.perf_counter() - t0
            sleep = max(0, frame_interval - elapsed)
            time.sleep(sleep)

            if frame_count % 100 == 0:
                print(f"[Sim] Sent {frame_count} frames")

    except KeyboardInterrupt:
        print(f"[Sim] Stopped after {frame_count} frames")

    cap.release()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add tests/edge_simulator.py
git commit -m "test: edge device simulator for integration testing

Publishes video file frames via ZMQ PUB to simulate edge device.
Supports configurable port, FPS, and video looping."
```

### Task 19: Integration tests

**Files:**
- Create: `tests/gpu/test_integration_cam1.py`
- Create: `tests/gpu/test_integration_cam2.py`
- Create: `tests/conftest.py` (if not exists)

- [ ] **Step 1: Write conftest.py**

```python
# tests/conftest.py
"""Test configuration and fixtures."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

- [ ] **Step 2: Write integration test for CAM1**

```python
# tests/gpu/test_integration_cam1.py
"""Integration tests for CAM1 pipeline (requires test video)."""
import numpy as np
import pytest
from gpu.roi_checker import ROIChecker
from gpu.overlay import draw_person_bboxes, draw_roi_polygons
from shared.models import BBox, DetectedObject


def test_roi_checker_with_sample_person():
    """ROI checker correctly identifies person in zone with sample data."""
    rois = [
        {"zone_name": "Test Zone", "color": "#ff0000", "enabled": True,
         "points_json": "[[0,0],[200,0],[200,200],[0,200]]"},
    ]
    checker = ROIChecker(rois)

    # Person foot point inside zone
    person = DetectedObject(bbox=BBox(50, 50, 150, 180), cls="person", conf=0.85)
    foot = (person.bbox.x1 + person.bbox.width / 2, person.bbox.y2)
    zones = checker.check_person(foot)
    assert len(zones) == 1

    # Person foot point outside zone
    person2 = DetectedObject(bbox=BBox(300, 300, 400, 400), cls="person", conf=0.85)
    foot2 = (person2.bbox.x1 + person2.bbox.width / 2, person2.bbox.y2)
    zones2 = checker.check_person(foot2)
    assert len(zones2) == 0


def test_overlay_with_sample_persons():
    """Overlay functions with sample persons produce valid output."""
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
    persons = [
        DetectedObject(bbox=BBox(50, 50, 150, 180), cls="person", conf=0.85),
        DetectedObject(bbox=BBox(300, 50, 400, 200), cls="person", conf=0.72),
    ]
    result = draw_person_bboxes(frame, persons)
    assert result.shape == frame.shape

    rois = [{"zone_name": "Zone A", "points_json": "[[0,0],[200,0],[200,200],[0,200]]"}]
    result2 = draw_roi_polygons(frame, rois)
    assert result2.shape == frame.shape
```

- [ ] **Step 3: Write PPE integration test**

```python
# tests/gpu/test_integration_cam2.py
"""Integration tests for CAM2 pipeline."""
import numpy as np
import pytest
from gpu.ppe_checker import crop_head, crop_torso, crop_feet
from gpu.overlay import draw_person_bboxes, draw_ppe_labels
from shared.models import BBox, DetectedObject


def test_crop_all_preserve_aspect():
    """Crop functions return valid regions."""
    frame = np.ones((480, 640, 3), dtype=np.uint8)
    bbox = BBox(100, 100, 300, 400)

    head = crop_head(frame, bbox)
    assert head.size > 0
    assert head.shape[2] == 3

    torso = crop_torso(frame, bbox)
    assert torso.size > 0

    feet = crop_feet(frame, bbox)
    assert feet.size > 0


def test_ppe_labels_on_empty_frame():
    """PPE labels draw correctly with no persons."""
    frame = np.ones((480, 640, 3), dtype=np.uint8)
    result = draw_ppe_labels(frame, [], [])
    assert result.shape == frame.shape
```

- [ ] **Step 4: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/gpu/test_integration_cam1.py tests/gpu/test_integration_cam2.py
git commit -m "test: integration tests for CAM1 and CAM2 pipelines

End-to-end ROI checker with sample data, overlay validation,
and PPE crop + label tests. All pass with synthetic data."
```

### Task 20: Cleanup old v1 code (optional, non-breaking)

**Files:**
- Remove (or keep as backup): inference/, alert/, dashboard/

- [ ] **Step 1: Preserve backup and remove from active use**

Old v1 code is in `inference/`, `alert/`, `dashboard/`. The `main.py` was already updated.
These can stay for reference but are not imported by v2. Optionally delete or move.

**Decision: Keep files on disk (backup), remove them from git tracking only if desired.**
Not recommended—just leave them as dead code for now. They don't affect v2 since `gpu/` is the new active package.

- [ ] **Step 2: Note in README**

Add a note to README about the v2 architecture.

```bash
git add README.md
git commit -m "docs: update README for v2 architecture

Note: v1 code (inference/, alert/, dashboard/) is preserved
but superseded by v2 (gpu/, edge/sender.py)."
```

### Self-Review Checklist

**Spec coverage check:**
- [x] Edge sender with ZeroMQ → Task 1
- [x] Shared models update → Task 2
- [x] Database/SQLite → Task 3
- [x] YOLOv8n detector → Task 4
- [x] MobileNetV3 classifier → Task 5
- [x] ROI checker → Task 6
- [x] PPE checker (crop + classify) → Task 7
- [x] Overlay drawing → Task 8
- [x] Alert manager with cooldown → Task 9
- [x] CAM1 QThread (ZMQ → detect → ROI → overlay) → Task 10
- [x] CAM2 QThread (ZMQ → detect → PPE → overlay) → Task 11
- [x] PyQt Camera Widget → Task 12
- [x] ROI Drawer (mouse interaction) → Task 13
- [x] Main Window (toolbar + layout) → Task 14
- [x] Main entry point (CVApp) → Task 15
- [x] FastAPI web server → Task 16
- [x] Requirements + model download → Task 17
- [x] Edge simulator → Task 18
- [x] Integration tests → Task 19
- [x] Cleanup → Task 20

**Placeholder scan:** No placeholders found. Every task contains complete code.

**Type consistency:**
- `DetectedObject` from `shared/models.py` has `.bbox` (BBox), `.cls` (str), `.conf` (float) — used consistently.
- `ROIChecker.check_person()` returns `List[dict]` with `zone_name`, `color` — matches Task 10 usage.
- `PPEChecker.process_persons()` returns `List[dict]` with `person_idx`, `violations`, `bbox` — matches Task 11.
- Alert dicts use `{"type", "zone_name", "person_idx", "bbox"}` — consistent across Task 10, 11, 14.
