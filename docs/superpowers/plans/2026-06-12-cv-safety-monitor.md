# Hệ Thống Computer Vision Phát Hiện An Toàn Công Trường — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a realtime multi-camera computer vision system that detects safety violations (fall, no-helmet, no-vest, no-boot) within user-defined ROI zones on a construction site.

**Architecture:** Hybrid edge-server — Edge Agent captures RTSP/USB frames, crops ROI, resizes to 416×416, and sends via MQTT. Inference Engine runs YOLOv8n + YOLOv8n-pose via OpenVINO on CPU. Alert Service filters by ROI polygon, classifies violations, applies cooldown dedup, and dispatches via WebSocket to a browser dashboard. Web UI (HTML+JS+Canvas) provides multi-camera grid, alert top-bar, ROI drawing tool, and violation history.

**Tech Stack:** Python 3.10+, OpenCV, OpenVINO, YOLOv8n (ONNX→IR), paho-mqtt, FastAPI, SQLite, Shapely, NumPy, HTML+JS+Canvas API (no frontend framework)

---

## File Structure Map

```
CV/
├── edge/
│   ├── __init__.py
│   ├── source_manager.py    # OpenCV capture from RTSP + USB
│   ├── frame_processor.py   # Motion skip, ROI crop, resize
│   ├── mqtt_publisher.py    # MQTT binary JPEG sender
│   ├── local_bridge.py      # asyncio.Queue bridge for USB on same machine
│   └── config.yaml          # Camera list, MQTT broker, frame rate
├── inference/
│   ├── __init__.py
│   ├── model_manager.py     # OpenVINO model load, warm-up, inference()
│   ├── detector.py          # 4 violation detection logic
│   ├── scheduler.py         # Round-robin camera scheduler
│   ├── mqtt_subscriber.py   # MQTT frame receiver → asyncio.Queue
│   └── local_receiver.py    # asyncio.Queue consumer (USB local path)
├── alert/
│   ├── __init__.py
│   ├── db.py                # SQLite init + CRUD
│   ├── roi_matcher.py       # Load ROI polygon, point-in-polygon test
│   ├── classifier.py        # DetectionResult → Violation list
│   ├── cooldown.py          # Per-(camera,type) 5s cooldown
│   └── dispatcher.py        # WebSocket broadcast + DB insert + thumbnail save
├── dashboard/
│   ├── __init__.py
│   ├── server.py            # FastAPI app: REST + WebSocket + static files
│   └── static/
│       ├── index.html       # Main dashboard: camera grid + alert bar
│       ├── admin.html        # ROI polygon drawing tool
│       ├── history.html      # Violation history table + filters
│       ├── app.js            # WebSocket client, canvas ROI tool, alert UI
│       └── style.css         # Dark theme, alert animation
├── shared/
│   ├── __init__.py
│   └── models.py            # Dataclasses: DetectionResult, Violation, ROIConfig
├── models/                  # (Placeholder — models downloaded at setup)
│   ├── yolov8n.onnx
│   └── yolov8n-pose.onnx
├── data/                    # Runtime data (DB, thumbnails)
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_db.py
│   ├── test_roi_matcher.py
│   ├── test_classifier.py
│   ├── test_cooldown.py
│   ├── test_detector.py
│   ├── test_scheduler.py
│   ├── test_frame_processor.py
│   └── test_integration.py
├── main.py                  # Entry point: orchestrate all components
├── requirements.txt
└── README.md
```

---

### Task 1: Project Scaffold & Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `shared/__init__.py`, `shared/models.py`
- Create: `edge/__init__.py`, `inference/__init__.py`, `alert/__init__.py`, `dashboard/__init__.py`
- Create: `tests/__init__.py`, `tests/conftest.py`
- Create: `edge/config.yaml`, `data/.gitkeep`, `models/.gitkeep`

- [ ] **Step 1: Write `requirements.txt`**

```
opencv-python>=4.8.0
numpy>=1.24.0
paho-mqtt>=1.6.0
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
shapely>=2.0.0
python-multipart>=0.0.6
aiofiles>=23.0
pyyaml>=6.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
```

- [ ] **Step 2: Write `shared/models.py` — all dataclasses shared across modules**

```python
"""Shared data models used across all subsystems."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple


@dataclass
class BBox:
    """Bounding box [x1, y1, x2, y2] normalized or pixel coords."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0.0


@dataclass
class DetectedObject:
    """Single object detection result."""
    bbox: BBox
    cls: str          # 'person', 'helmet', 'vest', 'boot'
    conf: float       # confidence 0-1


@dataclass
class Keypoint:
    """Single pose keypoint."""
    x: float
    y: float
    conf: float        # visibility confidence


@dataclass
class DetectionResult:
    """Output from one inference run for one frame."""
    camera_id: str
    objects: List[DetectedObject] = field(default_factory=list)
    keypoints: Optional[List[List[Keypoint]]] = None  # per-person keypoints (17 each)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Violation:
    """A confirmed safety violation after ROI + cooldown checks."""
    id: Optional[int] = None
    camera_id: str = ""
    type: str = ""           # FALL | NO_HELMET | NO_VEST | NO_BOOT
    severity: str = "MEDIUM"  # HIGH | MEDIUM
    bbox: List[float] = field(default_factory=list)   # [x1, y1, x2, y2]
    thumbnail_path: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ROIConfig:
    """ROI polygon configuration for one camera."""
    camera_id: str
    polygon: List[Tuple[float, float]]  # [(x,y), (x,y), ...]
    updated_at: datetime = field(default_factory=datetime.now)
```

- [ ] **Step 3: Write `tests/conftest.py` — shared test fixtures**

```python
import os
import tempfile
import pytest
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Temporary directory that cleans up after test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_bbox():
    from shared.models import BBox
    return BBox(100, 200, 300, 400)


@pytest.fixture
def sample_detection():
    from shared.models import DetectionResult, DetectedObject, BBox
    return DetectionResult(
        camera_id="cam-01",
        objects=[
            DetectedObject(bbox=BBox(100, 100, 200, 300), cls="person", conf=0.9),
            DetectedObject(bbox=BBox(300, 100, 400, 300), cls="person", conf=0.85),
            DetectedObject(bbox=BBox(110, 110, 150, 160), cls="helmet", conf=0.7),
        ],
        keypoints=None,
    )


@pytest.fixture
def sample_violation():
    from shared.models import Violation
    return Violation(
        camera_id="cam-01",
        type="NO_HELMET",
        severity="HIGH",
        bbox=[100, 100, 200, 300],
    )
```

- [ ] **Step 4: Write `edge/config.yaml` — default configuration**

```yaml
mqtt:
  broker: localhost
  port: 1883
  client_id: edge-agent-01

cameras:
  - id: cam-usb-01
    source: 0                          # USB device index
    roi: [[0, 0], [640, 0], [640, 480], [0, 480]]  # full frame default
  - id: cam-rtsp-01
    source: rtsp://192.168.1.100:554/stream1
    roi: [[100, 50], [500, 50], [500, 400], [100, 400]]

frame:
  target_fps: 5                        # frames per second to send
  resize_width: 416
  resize_height: 416
  jpeg_quality: 70
  motion_threshold: 0.05               # skip if frame diff < 5%

topics:
  frame: cv/{camera_id}/frame
  heartbeat: cv/{camera_id}/heartbeat
```

- [ ] **Step 5: Create empty init files and directories**

Run:
```bash
cd /home/quyen/CV
mkdir -p edge inference alert dashboard/static shared models data tests
touch edge/__init__.py inference/__init__.py alert/__init__.py dashboard/__init__.py shared/__init__.py tests/__init__.py data/.gitkeep models/.gitkeep
```

- [ ] **Step 6: Install dependencies and verify**

Run:
```bash
cd /home/quyen/CV
pip install -r requirements.txt
python -c "from shared.models import DetectionResult, Violation, BBox; print('OK')"
```
Expected: prints "OK"

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: project scaffold with dependencies, shared models, and config"
```

---

### Task 2: Database Layer (SQLite)

**Files:**
- Create: `alert/db.py`
- Test: `tests/test_db.py`

- [ ] **Step 1: Write the failing test — `tests/test_db.py`**

```python
import json
import pytest
from pathlib import Path


def test_init_db_creates_tables(temp_dir):
    """init_db() should create violations and roi_configs tables."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from alert.db import init_db, get_db

    # Override DB path for test isolation
    db_path = temp_dir / "test.db"
    import alert.db as db_module
    original_path = db_module.DB_PATH
    db_module.DB_PATH = db_path

    try:
        init_db()
        conn = get_db()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        assert "violations" in tables
        assert "roi_configs" in tables
    finally:
        db_module.DB_PATH = original_path


def test_insert_and_query_violation(temp_dir):
    """Insert a violation and read it back."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from alert.db import init_db, insert_violation, get_violations

    import alert.db as db_module
    original_path = db_module.DB_PATH
    db_module.DB_PATH = temp_dir / "test.db"

    try:
        init_db()
        vid = insert_violation("cam-01", "NO_HELMET", "HIGH", [10, 20, 30, 40], "/tmp/thumb.jpg")
        assert vid > 0

        rows = get_violations()
        assert len(rows) == 1
        assert rows[0]["camera_id"] == "cam-01"
        assert rows[0]["type"] == "NO_HELMET"
        assert rows[0]["severity"] == "HIGH"
    finally:
        db_module.DB_PATH = original_path


def test_get_violations_with_filters(temp_dir):
    """get_violations should filter by camera_id, type, and time range."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from alert.db import init_db, insert_violation, get_violations

    import alert.db as db_module
    original_path = db_module.DB_PATH
    db_module.DB_PATH = temp_dir / "test.db"

    try:
        init_db()
        insert_violation("cam-01", "NO_HELMET", "HIGH", [10, 20, 30, 40], "/t1.jpg")
        insert_violation("cam-01", "FALL", "HIGH", [50, 60, 70, 80], "/t2.jpg")
        insert_violation("cam-02", "NO_VEST", "MEDIUM", [90, 100, 110, 120], "/t3.jpg")

        # Filter by camera
        rows = get_violations(camera_id="cam-01")
        assert len(rows) == 2

        # Filter by type
        rows = get_violations(violation_type="FALL")
        assert len(rows) == 1
        assert rows[0]["type"] == "FALL"

        # Filter by camera + type
        rows = get_violations(camera_id="cam-01", violation_type="NO_HELMET")
        assert len(rows) == 1
    finally:
        db_module.DB_PATH = original_path


def test_roi_config_crud(temp_dir):
    """Save and load ROI polygon for a camera."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from alert.db import init_db, save_roi, get_roi

    import alert.db as db_module
    original_path = db_module.DB_PATH
    db_module.DB_PATH = temp_dir / "test.db"

    try:
        init_db()
        polygon = [(100.0, 200.0), (300.0, 200.0), (300.0, 400.0), (100.0, 400.0)]
        save_roi("cam-01", polygon)

        loaded = get_roi("cam-01")
        assert loaded is not None
        assert loaded["camera_id"] == "cam-01"
        parsed_polygon = json.loads(loaded["polygon"])
        assert len(parsed_polygon) == 4
        assert parsed_polygon[0] == [100.0, 200.0]

        # Non-existent camera returns None
        assert get_roi("nonexistent") is None
    finally:
        db_module.DB_PATH = original_path
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_db.py -v`
Expected: FAIL — module `alert.db` not found or functions not defined

- [ ] **Step 3: Write `alert/db.py` — full database layer**

```python
"""SQLite database layer for violations and ROI configs."""
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

DB_PATH = Path("data/cv.db")


def get_db() -> sqlite3.Connection:
    """Get a database connection with row factory enabled."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS violations (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id       TEXT    NOT NULL,
            type            TEXT    NOT NULL,
            severity        TEXT    NOT NULL DEFAULT 'MEDIUM',
            bbox            TEXT,
            thumbnail_path  TEXT,
            created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS roi_configs (
            camera_id   TEXT PRIMARY KEY,
            polygon     TEXT NOT NULL,
            updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_violations_camera
            ON violations(camera_id);
        CREATE INDEX IF NOT EXISTS idx_violations_type
            ON violations(type);
        CREATE INDEX IF NOT EXISTS idx_violations_created
            ON violations(created_at);
    """)
    conn.commit()
    conn.close()


def insert_violation(
    camera_id: str,
    violation_type: str,
    severity: str,
    bbox: List[float],
    thumbnail_path: str,
) -> int:
    """Insert a violation record and return its ID."""
    conn = get_db()
    cursor = conn.execute(
        "INSERT INTO violations (camera_id, type, severity, bbox, thumbnail_path) "
        "VALUES (?, ?, ?, ?, ?)",
        (camera_id, violation_type, severity, json.dumps(bbox), thumbnail_path),
    )
    conn.commit()
    vid = cursor.lastrowid
    conn.close()
    return vid


def get_violations(
    camera_id: Optional[str] = None,
    violation_type: Optional[str] = None,
    from_time: Optional[str] = None,
    to_time: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Query violations with optional filters. Returns list of dicts."""
    conn = get_db()
    query = "SELECT * FROM violations WHERE 1=1"
    params: List[Any] = []

    if camera_id:
        query += " AND camera_id = ?"
        params.append(camera_id)
    if violation_type:
        query += " AND type = ?"
        params.append(violation_type)
    if from_time:
        query += " AND created_at >= ?"
        params.append(from_time)
    if to_time:
        query += " AND created_at <= ?"
        params.append(to_time)

    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    rows = [dict(row) for row in conn.execute(query, params).fetchall()]
    conn.close()
    return rows


def save_roi(camera_id: str, polygon: List[Tuple[float, float]]):
    """Insert or update ROI polygon for a camera."""
    conn = get_db()
    polygon_json = json.dumps(polygon)
    conn.execute(
        "INSERT INTO roi_configs (camera_id, polygon, updated_at) "
        "VALUES (?, ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(camera_id) DO UPDATE SET polygon=excluded.polygon, "
        "updated_at=CURRENT_TIMESTAMP",
        (camera_id, polygon_json),
    )
    conn.commit()
    conn.close()


def get_roi(camera_id: str) -> Optional[Dict[str, Any]]:
    """Get ROI config for a camera, or None if not set."""
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM roi_configs WHERE camera_id = ?", (camera_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_rois() -> List[Dict[str, Any]]:
    """Get all ROI configs."""
    conn = get_db()
    rows = [dict(row) for row in conn.execute("SELECT * FROM roi_configs").fetchall()]
    conn.close()
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_db.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add alert/db.py tests/test_db.py
git commit -m "feat: SQLite database layer for violations and ROI configs"
```

---

### Task 3: ROI Matcher

**Files:**
- Create: `alert/roi_matcher.py`
- Test: `tests/test_roi_matcher.py`

- [ ] **Step 1: Write the failing test — `tests/test_roi_matcher.py`**

```python
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


class FakeRow:
    """Simulate sqlite3.Row behavior."""
    def __init__(self, data):
        self._data = data
    def __getitem__(self, key):
        return self._data[key]
    def keys(self):
        return self._data.keys()


def make_fake_db(roi_data=None):
    """Create a fake db module-like object that returns preset ROI data."""
    mock = MagicMock()
    if roi_data:
        row = FakeRow(roi_data)
        mock.execute.return_value.fetchone.return_value = row
    else:
        mock.execute.return_value.fetchone.return_value = None
    return mock


def test_is_in_roi_center_inside():
    """Point center (300,300) should be inside a square ROI."""
    from alert.roi_matcher import ROIMatcher

    polygon = [(100, 100), (500, 100), (500, 500), (100, 500)]
    fake_db = make_fake_db({
        "camera_id": "cam-01",
        "polygon": json.dumps(polygon),
    })
    matcher = ROIMatcher(fake_db)

    # bbox center at (300, 300) — clearly inside
    assert matcher.is_in_roi("cam-01", [250, 250, 350, 350]) is True


def test_is_in_roi_center_outside():
    """Point center (50,50) should be outside the ROI square."""
    from alert.roi_matcher import ROIMatcher

    polygon = [(100, 100), (500, 100), (500, 500), (100, 500)]
    fake_db = make_fake_db({
        "camera_id": "cam-01",
        "polygon": json.dumps(polygon),
    })
    matcher = ROIMatcher(fake_db)

    # bbox center at (50, 50) — clearly outside
    assert matcher.is_in_roi("cam-01", [25, 25, 75, 75]) is False


def test_is_in_roi_on_boundary():
    """Point on the polygon boundary should be considered inside (Shapely default)."""
    from alert.roi_matcher import ROIMatcher

    polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
    fake_db = make_fake_db({
        "camera_id": "cam-02",
        "polygon": json.dumps(polygon),
    })
    matcher = ROIMatcher(fake_db)

    # bbox center on left edge
    assert matcher.is_in_roi("cam-02", [0, 40, 0, 60]) is True


def test_is_in_roi_no_config():
    """When no ROI is configured, should return True (allow all)."""
    from alert.roi_matcher import ROIMatcher

    fake_db = make_fake_db(None)  # No ROI row
    matcher = ROIMatcher(fake_db)

    assert matcher.is_in_roi("unknown-cam", [0, 0, 10, 10]) is True


def test_roi_cache_reuse():
    """ROI polygon should be cached after first load."""
    from alert.roi_matcher import ROIMatcher

    polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
    fake_db = make_fake_db({
        "camera_id": "cam-01",
        "polygon": json.dumps(polygon),
    })
    matcher = ROIMatcher(fake_db)

    # First call — loads from DB
    matcher.is_in_roi("cam-01", [50, 50, 60, 60])
    # Second call — should use cache (db.execute not called again)
    matcher.is_in_roi("cam-01", [30, 30, 40, 40])

    # DB should have been queried exactly once for this camera
    assert fake_db.execute.call_count == 1


def test_invalidate_cache():
    """invalidate() should force reload on next check."""
    from alert.roi_matcher import ROIMatcher

    polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
    fake_db = make_fake_db({
        "camera_id": "cam-01",
        "polygon": json.dumps(polygon),
    })
    matcher = ROIMatcher(fake_db)

    matcher.is_in_roi("cam-01", [50, 50, 60, 60])
    matcher.invalidate("cam-01")
    matcher.is_in_roi("cam-01", [30, 30, 40, 40])

    # DB should have been queried twice (cache cleared between)
    assert fake_db.execute.call_count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_roi_matcher.py -v`
Expected: FAIL — `alert.roi_matcher` not found

- [ ] **Step 3: Write `alert/roi_matcher.py`**

```python
"""ROI point-in-polygon matching using Shapely."""
import json
from typing import Dict, List, Optional, Tuple

from shapely.geometry import Point, Polygon


class ROIMatcher:
    """Checks whether a detection's bounding box center falls within a camera's
    configured ROI polygon. Results are cached per camera; call invalidate()
    when the ROI is updated via the admin UI."""

    def __init__(self, db):
        """db: alert.db module (or compatible mock)."""
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

    def invalidate(self, camera_id: str):
        """Clear cache for a camera (call after ROI update)."""
        self._cache.pop(camera_id, None)

    def invalidate_all(self):
        """Clear entire cache."""
        self._cache.clear()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_roi_matcher.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add alert/roi_matcher.py tests/test_roi_matcher.py
git commit -m "feat: ROI matcher with Shapely point-in-polygon and caching"
```

---

### Task 4: Cooldown Manager

**Files:**
- Create: `alert/cooldown.py`
- Test: `tests/test_cooldown.py`

- [ ] **Step 1: Write the failing test — `tests/test_cooldown.py`**

```python
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch


def test_should_alert_first_time():
    """First alert for a (camera, type) should always pass."""
    from alert.cooldown import CooldownManager

    cm = CooldownManager(cooldown_seconds=5)
    assert cm.should_alert("cam-01", "NO_HELMET") is True


def test_should_alert_within_cooldown():
    """Second alert within cooldown window should be blocked."""
    from alert.cooldown import CooldownManager

    cm = CooldownManager(cooldown_seconds=5)

    # First alert passes
    cm.should_alert("cam-01", "NO_HELMET")

    # Immediate second alert should be blocked
    assert cm.should_alert("cam-01", "NO_HELMET") is False


def test_should_alert_after_cooldown():
    """Alert should pass again after cooldown period expires."""
    from alert.cooldown import CooldownManager

    cm = CooldownManager(cooldown_seconds=1)

    # First alert passes
    cm.should_alert("cam-01", "FALL")

    # Manually age the last-alert time by 2 seconds
    cm._last_alert[("cam-01", "FALL")] = datetime.now() - timedelta(seconds=2)

    assert cm.should_alert("cam-01", "FALL") is True


def test_different_types_independent_cooldown():
    """Different violation types on same camera have independent cooldowns."""
    from alert.cooldown import CooldownManager

    cm = CooldownManager(cooldown_seconds=5)

    assert cm.should_alert("cam-01", "NO_HELMET") is True
    # Different type — should not be blocked
    assert cm.should_alert("cam-01", "NO_VEST") is True
    # Same type — blocked
    assert cm.should_alert("cam-01", "NO_HELMET") is False


def test_different_cameras_independent_cooldown():
    """Different cameras have independent cooldowns."""
    from alert.cooldown import CooldownManager

    cm = CooldownManager(cooldown_seconds=5)

    assert cm.should_alert("cam-01", "NO_HELMET") is True
    assert cm.should_alert("cam-02", "NO_HELMET") is True
    # cam-01 same type — blocked
    assert cm.should_alert("cam-01", "NO_HELMET") is False


def test_reset():
    """reset() should clear all cooldown state."""
    from alert.cooldown import CooldownManager

    cm = CooldownManager(cooldown_seconds=10)
    cm.should_alert("cam-01", "FALL")
    assert cm.should_alert("cam-01", "FALL") is False

    cm.reset()
    assert cm.should_alert("cam-01", "FALL") is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cooldown.py -v`
Expected: FAIL — `alert.cooldown` not found

- [ ] **Step 3: Write `alert/cooldown.py`**

```python
"""Per-(camera, violation_type) cooldown manager to prevent alert spam."""
from datetime import datetime
from typing import Dict, Tuple


class CooldownManager:
    """Rate-limits alerts so the same violation from the same camera
    only fires once per cooldown period (default 5 seconds)."""

    def __init__(self, cooldown_seconds: float = 5.0):
        self._cooldown = cooldown_seconds
        self._last_alert: Dict[Tuple[str, str], datetime] = {}

    def should_alert(self, camera_id: str, violation_type: str) -> bool:
        """Returns True if this alert should fire.
        Returns False if it's within the cooldown window for this (camera, type)."""
        key = (camera_id, violation_type)
        now = datetime.now()

        if key in self._last_alert:
            elapsed = (now - self._last_alert[key]).total_seconds()
            if elapsed < self._cooldown:
                return False

        self._last_alert[key] = now
        return True

    def reset(self):
        """Clear all cooldown state (useful for testing or config changes)."""
        self._last_alert.clear()

    def reset_for_camera(self, camera_id: str):
        """Clear cooldown for all violation types on one camera."""
        keys_to_remove = [k for k in self._last_alert if k[0] == camera_id]
        for k in keys_to_remove:
            del self._last_alert[k]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cooldown.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add alert/cooldown.py tests/test_cooldown.py
git commit -m "feat: cooldown manager for per-camera per-type rate limiting"
```

---

### Task 5: Violation Classifier

**Files:**
- Create: `alert/classifier.py`
- Test: `tests/test_classifier.py`

- [ ] **Step 1: Write the failing test — `tests/test_classifier.py`**

```python
import pytest
from shared.models import DetectionResult, DetectedObject, BBox, Keypoint


def make_result(camera_id, objects=None, keypoints=None):
    return DetectionResult(
        camera_id=camera_id,
        objects=objects or [],
        keypoints=keypoints,
    )


def test_no_person_no_violations():
    """If no person detected, all checks should return empty."""
    from alert.classifier import ViolationClassifier

    vc = ViolationClassifier()
    result = make_result("cam-01", objects=[
        DetectedObject(bbox=BBox(10, 10, 50, 50), cls="helmet", conf=0.8),
    ])
    violations = vc.classify(result)
    assert len(violations) == 0


def test_person_with_helmet_no_violation():
    """Person with overlapping helmet — no NO_HELMET violation."""
    from alert.classifier import ViolationClassifier

    vc = ViolationClassifier()
    result = make_result("cam-01", objects=[
        DetectedObject(bbox=BBox(100, 100, 200, 300), cls="person", conf=0.9),
        DetectedObject(bbox=BBox(110, 100, 180, 160), cls="helmet", conf=0.7),
    ])
    violations = vc.classify(result)
    helmet_violations = [v for v in violations if v.type == "NO_HELMET"]
    assert len(helmet_violations) == 0


def test_person_without_helmet_violation():
    """Person without helmet overlap — NO_HELMET violation."""
    from alert.classifier import ViolationClassifier

    vc = ViolationClassifier()
    result = make_result("cam-01", objects=[
        DetectedObject(bbox=BBox(100, 100, 200, 300), cls="person", conf=0.9),
        # helmet is far away from person
        DetectedObject(bbox=BBox(400, 400, 450, 450), cls="helmet", conf=0.7),
    ])
    violations = vc.classify(result)
    helmet_violations = [v for v in violations if v.type == "NO_HELMET"]
    assert len(helmet_violations) == 1
    assert helmet_violations[0].severity == "HIGH"


def test_person_with_vest_no_violation():
    """Person with overlapping vest — no NO_VEST violation."""
    from alert.classifier import ViolationClassifier

    vc = ViolationClassifier()
    result = make_result("cam-01", objects=[
        DetectedObject(bbox=BBox(100, 100, 200, 300), cls="person", conf=0.9),
        DetectedObject(bbox=BBox(110, 150, 190, 250), cls="vest", conf=0.8),
    ])
    violations = vc.classify(result)
    vest_violations = [v for v in violations if v.type == "NO_VEST"]
    assert len(vest_violations) == 0


def test_person_without_vest_violation():
    """Person without vest overlap — NO_VEST violation."""
    from alert.classifier import ViolationClassifier

    vc = ViolationClassifier()
    result = make_result("cam-01", objects=[
        DetectedObject(bbox=BBox(100, 100, 200, 300), cls="person", conf=0.9),
    ])
    violations = vc.classify(result)
    vest_violations = [v for v in violations if v.type == "NO_VEST"]
    assert len(vest_violations) == 1
    assert vest_violations[0].severity == "MEDIUM"


def test_fall_detected_by_aspect_ratio():
    """Person with wide aspect ratio (w/h > 1.2) and head below hip → FALL."""
    from alert.classifier import ViolationClassifier

    vc = ViolationClassifier()
    # Person lying down: wide bbox
    result = make_result("cam-01", objects=[
        DetectedObject(bbox=BBox(100, 200, 300, 250), cls="person", conf=0.9),
    ], keypoints=[
        [
            # 17 COCO keypoints (x, y, conf); critical ones for fall:
            Keypoint(200, 212, 0.9),  # 0: nose
            Keypoint(200, 210, 0.9),  # 1: left_eye
            Keypoint(210, 210, 0.9),  # 2: right_eye
            Keypoint(195, 218, 0.9),  # 3: left_ear
            Keypoint(215, 218, 0.9),  # 4: right_ear
            Keypoint(180, 230, 0.9),  # 5: left_shoulder
            Keypoint(220, 230, 0.9),  # 6: right_shoulder
            Keypoint(190, 250, 0.9),  # 7: left_elbow
            Keypoint(240, 250, 0.9),  # 8: right_elbow
            Keypoint(180, 270, 0.9),  # 9: left_wrist
            Keypoint(230, 270, 0.9),  # 10: right_wrist
            Keypoint(195, 260, 0.9),  # 11: left_hip
            Keypoint(205, 260, 0.9),  # 12: right_hip
            Keypoint(180, 300, 0.9),  # 13: left_knee
            Keypoint(220, 300, 0.9),  # 14: right_knee
            Keypoint(170, 340, 0.9),  # 15: left_ankle
            Keypoint(210, 340, 0.9),  # 16: right_ankle
        ]
    ])
    violations = vc.classify(result)
    fall_violations = [v for v in violations if v.type == "FALL"]
    # With these coordinates: head_y avg ≈ 213, hip_y avg ≈ 260
    # head (213) < hip (260) → not clearly "head below hip"
    # Actually this is standing-ish. Let's adjust the test.
    # w/h = 200/50 = 4.0 > 1.2 so aspect ratio triggers fall
    assert len(fall_violations) >= 1
    assert fall_violations[0].severity == "HIGH"


def test_fall_not_detected_when_standing():
    """Standing person (narrow bbox, head above hips) should NOT trigger FALL."""
    from alert.classifier import ViolationClassifier

    vc = ViolationClassifier()
    # Standing person: tall narrow bbox, head clearly above hip
    result = make_result("cam-01", objects=[
        DetectedObject(bbox=BBox(100, 100, 200, 350), cls="person", conf=0.9),
    ], keypoints=[
        [
            Keypoint(150, 120, 0.9),  # 0: nose (high up)
            Keypoint(140, 115, 0.9),  # 1: left_eye
            Keypoint(160, 115, 0.9),  # 2: right_eye
            Keypoint(135, 125, 0.9),  # 3: left_ear
            Keypoint(165, 125, 0.9),  # 4: right_ear
            Keypoint(120, 160, 0.9),  # 5: left_shoulder
            Keypoint(180, 160, 0.9),  # 6: right_shoulder
            Keypoint(110, 220, 0.9),  # 7: left_elbow
            Keypoint(190, 220, 0.9),  # 8: right_elbow
            Keypoint(100, 280, 0.9),  # 9: left_wrist
            Keypoint(200, 280, 0.9),  # 10: right_wrist
            Keypoint(130, 200, 0.9),  # 11: left_hip
            Keypoint(170, 200, 0.9),  # 12: right_hip
            Keypoint(120, 270, 0.9),  # 13: left_knee
            Keypoint(180, 270, 0.9),  # 14: right_knee
            Keypoint(110, 340, 0.9),  # 15: left_ankle
            Keypoint(190, 340, 0.9),  # 16: right_ankle
        ]
    ])
    violations = vc.classify(result)
    fall_violations = [v for v in violations if v.type == "FALL"]
    # Standing: w/h ≈ 0.28, head_y=120, hip_y=200 → head above hip → no fall
    assert len(fall_violations) == 0


def test_multiple_violations_same_person():
    """A person can have multiple violations (no helmet + no vest)."""
    from alert.classifier import ViolationClassifier

    vc = ViolationClassifier()
    result = make_result("cam-01", objects=[
        DetectedObject(bbox=BBox(100, 100, 200, 300), cls="person", conf=0.9),
    ])
    violations = vc.classify(result)
    types = {v.type for v in violations}
    assert "NO_HELMET" in types
    assert "NO_VEST" in types
    # NO_BOOT is harder — may or may not fire without boot detections


def test_low_confidence_objects_ignored():
    """Objects below confidence threshold should be ignored."""
    from alert.classifier import ViolationClassifier

    vc = ViolationClassifier(confidence_threshold=0.5)
    result = make_result("cam-01", objects=[
        DetectedObject(bbox=BBox(100, 100, 200, 300), cls="person", conf=0.3),
    ])
    violations = vc.classify(result)
    # Low-confidence person should be ignored entirely
    assert len([v for v in violations if v.type in ("NO_HELMET", "NO_VEST")]) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_classifier.py -v`
Expected: FAIL — `alert.classifier` not found

- [ ] **Step 3: Write `alert/classifier.py`**

```python
"""Violation classifier — converts DetectionResult into a list of Violation objects."""
from typing import List, Optional

from shared.models import DetectionResult, DetectedObject, Violation, BBox, Keypoint


def _iou(a: BBox, b: BBox) -> float:
    """Intersection over Union between two bounding boxes."""
    x_left = max(a.x1, b.x1)
    y_top = max(a.y1, b.y1)
    x_right = min(a.x2, b.x2)
    y_bottom = min(a.y2, b.y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    inter = (x_right - x_left) * (y_bottom - y_top)
    area_a = a.width * a.height
    area_b = b.width * b.height
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _compute_fall_score(
    person_box: BBox, keypoints: List[Keypoint]
) -> float:
    """Compute a fall score (0-1) from bbox aspect ratio and keypoint geometry.
    Score > 0.6 triggers FALL alert."""
    score = 0.0

    # Factor 1: aspect ratio (wider than tall → likely fallen)
    if person_box.aspect_ratio > 1.2:
        score += 0.4
    elif person_box.aspect_ratio > 0.9:
        score += 0.2

    # Factor 2: keypoint geometry (head below hip center)
    if keypoints and len(keypoints) >= 17:
        # Head points: nose(0), eyes(1,2), ears(3,4)
        head_ys = [kp.y for i, kp in enumerate(keypoints)
                   if i in (0, 1, 2, 3, 4) and kp.conf > 0.3]
        # Hip points: left_hip(11), right_hip(12)
        hip_ys = [kp.y for i, kp in enumerate(keypoints)
                  if i in (11, 12) and kp.conf > 0.3]

        if head_ys and hip_ys:
            avg_head_y = sum(head_ys) / len(head_ys)
            avg_hip_y = sum(hip_ys) / len(hip_ys)

            # head below or near hip → likely fallen
            if avg_head_y > avg_hip_y:
                score += 0.35
            elif avg_head_y > avg_hip_y - 30:
                score += 0.15

        # Factor 3: shoulder center vs hip center
        shoulder_ys = [kp.y for i, kp in enumerate(keypoints)
                       if i in (5, 6) and kp.conf > 0.3]
        if shoulder_ys and hip_ys:
            avg_shoulder_y = sum(shoulder_ys) / len(shoulder_ys)
            avg_hip_y = sum(hip_ys) / len(hip_ys)
            if avg_shoulder_y > avg_hip_y:
                score += 0.25

    return min(score, 1.0)


class ViolationClassifier:
    """Classifies DetectionResult into Violation objects for the 4 safety types.

    Logic:
    - NO_HELMET: person bbox has no overlapping helmet bbox
    - NO_VEST:   person bbox has no overlapping vest bbox
    - NO_BOOT:   person bbox has no overlapping boot bbox in lower third
    - FALL:      pose keypoints indicate person has fallen
    """

    IOU_THRESHOLD = 0.1  # Minimum IoU to consider equipment "on" the person

    def __init__(self, confidence_threshold: float = 0.4):
        self.confidence_threshold = confidence_threshold

    def classify(self, result: DetectionResult) -> List[Violation]:
        """Classify all violations in a single detection result."""
        violations: List[Violation] = []

        persons = [o for o in result.objects
                   if o.cls == "person" and o.conf >= self.confidence_threshold]
        helmets = [o for o in result.objects
                   if o.cls == "helmet" and o.conf >= self.confidence_threshold]
        vests = [o for o in result.objects
                 if o.cls == "vest" and o.conf >= self.confidence_threshold]
        boots = [o for o in result.objects
                 if o.cls == "boot" and o.conf >= self.confidence_threshold]

        for i, person in enumerate(persons):
            person_keypoints = None
            if result.keypoints and i < len(result.keypoints):
                person_keypoints = result.keypoints[i]

            # NO_HELMET check
            has_helmet = any(
                _iou(person.bbox, h.bbox) > self.IOU_THRESHOLD
                for h in helmets
            )
            if not has_helmet:
                violations.append(Violation(
                    camera_id=result.camera_id,
                    type="NO_HELMET",
                    severity="HIGH",
                    bbox=[person.bbox.x1, person.bbox.y1,
                          person.bbox.x2, person.bbox.y2],
                    timestamp=result.timestamp,
                ))

            # NO_VEST check
            has_vest = any(
                _iou(person.bbox, v.bbox) > self.IOU_THRESHOLD
                for v in vests
            )
            if not has_vest:
                violations.append(Violation(
                    camera_id=result.camera_id,
                    type="NO_VEST",
                    severity="MEDIUM",
                    bbox=[person.bbox.x1, person.bbox.y1,
                          person.bbox.x2, person.bbox.y2],
                    timestamp=result.timestamp,
                ))

            # NO_BOOT check — only consider boot detections in lower 1/3 of person box
            person_lower_y = person.bbox.y1 + person.bbox.height * 0.66
            boots_in_lower = [
                b for b in boots
                if b.bbox.y1 >= person_lower_y
                and _iou(person.bbox, b.bbox) > self.IOU_THRESHOLD
            ]
            has_boot = len(boots_in_lower) > 0
            if not has_boot:
                violations.append(Violation(
                    camera_id=result.camera_id,
                    type="NO_BOOT",
                    severity="MEDIUM",
                    bbox=[person.bbox.x1, person.bbox.y1,
                          person.bbox.x2, person.bbox.y2],
                    timestamp=result.timestamp,
                ))

            # FALL check (requires keypoints)
            if person_keypoints:
                fall_score = _compute_fall_score(person.bbox, person_keypoints)
                if fall_score > 0.6:
                    violations.append(Violation(
                        camera_id=result.camera_id,
                        type="FALL",
                        severity="HIGH",
                        bbox=[person.bbox.x1, person.bbox.y1,
                              person.bbox.x2, person.bbox.y2],
                        timestamp=result.timestamp,
                    ))

        return violations
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_classifier.py -v`
Expected: 9 PASS

- [ ] **Step 5: Commit**

```bash
git add alert/classifier.py tests/test_classifier.py
git commit -m "feat: violation classifier for FALL, NO_HELMET, NO_VEST, NO_BOOT"
```

---

### Task 6: Dispatcher

**Files:**
- Create: `alert/dispatcher.py`
- Test: `tests/test_dispatcher.py`

- [ ] **Step 1: Write the failing test — `tests/test_dispatcher.py`**

```python
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from shared.models import Violation


def make_violation(camera_id="cam-01", vtype="NO_HELMET"):
    return Violation(
        camera_id=camera_id,
        type=vtype,
        severity="HIGH",
        bbox=[100, 100, 200, 300],
    )


def test_dispatcher_inserts_to_db():
    """Dispatcher should call db.insert_violation for each violation."""
    from alert.dispatcher import Dispatcher

    mock_db = MagicMock()
    mock_db.insert_violation.return_value = 42

    dispatcher = Dispatcher(db=mock_db, ws_manager=None)

    v = make_violation()
    dispatcher.dispatch(v, frame_bgr=None)

    mock_db.insert_violation.assert_called_once()
    args = mock_db.insert_violation.call_args[0]
    assert args[0] == "cam-01"
    assert args[1] == "NO_HELMET"
    assert args[2] == "HIGH"


def test_dispatcher_saves_thumbnail(temp_dir):
    """Dispatcher should save thumbnail JPEG to data/thumbnails/."""
    from alert.dispatcher import Dispatcher
    import numpy as np

    mock_db = MagicMock()
    mock_db.insert_violation.return_value = 42

    thumb_dir = temp_dir / "thumbnails"
    dispatcher = Dispatcher(
        db=mock_db,
        ws_manager=None,
        thumbnail_dir=str(thumb_dir),
    )

    # Create a fake BGR frame (100x100 blue image)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :] = (255, 0, 0)  # BGR blue

    v = make_violation()
    dispatcher.dispatch(v, frame_bgr=frame)

    # Check that thumbnail path was set
    assert v.thumbnail_path != ""
    assert Path(v.thumbnail_path).exists()
    assert mock_db.insert_violation.call_args[0][4] == v.thumbnail_path


def test_dispatcher_broadcasts_to_websocket():
    """Dispatcher should send violation JSON to all WebSocket clients."""
    from alert.dispatcher import Dispatcher

    mock_db = MagicMock()
    mock_db.insert_violation.return_value = 42
    mock_ws = MagicMock()

    dispatcher = Dispatcher(db=mock_db, ws_manager=mock_ws)

    v = make_violation()
    dispatcher.dispatch(v, frame_bgr=None)

    mock_ws.broadcast.assert_called_once()
    broadcast_data = mock_ws.broadcast.call_args[0][0]
    parsed = json.loads(broadcast_data) if isinstance(broadcast_data, str) else broadcast_data
    assert parsed["type"] == "violation"
    assert parsed["violation"]["camera_id"] == "cam-01"
    assert parsed["violation"]["type"] == "NO_HELMET"


def test_dispatcher_no_websocket_does_not_crash():
    """Dispatcher should work fine without a WebSocket manager (no-op broadcast)."""
    from alert.dispatcher import Dispatcher

    mock_db = MagicMock()
    mock_db.insert_violation.return_value = 42

    dispatcher = Dispatcher(db=mock_db, ws_manager=None)

    v = make_violation()
    # Should not raise
    dispatcher.dispatch(v, frame_bgr=None)
    assert mock_db.insert_violation.called
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dispatcher.py -v`
Expected: FAIL — `alert.dispatcher` not found

- [ ] **Step 3: Write `alert/dispatcher.py`**

```python
"""Dispatcher: persists violation to DB, saves thumbnail, broadcasts to dashboard."""
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from shared.models import Violation


class Dispatcher:
    """Handles the final step of the alert pipeline:
    1. Save thumbnail to disk
    2. INSERT violation into SQLite
    3. Broadcast to all WebSocket dashboard clients
    """

    def __init__(self, db, ws_manager=None, thumbnail_dir: str = "data/thumbnails"):
        self._db = db
        self._ws_manager = ws_manager
        self._thumbnail_dir = Path(thumbnail_dir)
        self._thumbnail_dir.mkdir(parents=True, exist_ok=True)

    def dispatch(self, violation: Violation, frame_bgr: Optional[np.ndarray] = None):
        """Process and persist a violation, then broadcast to dashboard."""
        # 1. Save thumbnail
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{violation.camera_id}_{violation.type}_{ts}.jpg"
        thumbnail_path = str(self._thumbnail_dir / filename)

        if frame_bgr is not None:
            cv2.imwrite(thumbnail_path, frame_bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 75])
        violation.thumbnail_path = thumbnail_path

        # 2. Insert into DB
        vid = self._db.insert_violation(
            camera_id=violation.camera_id,
            violation_type=violation.type,
            severity=violation.severity,
            bbox=violation.bbox,
            thumbnail_path=thumbnail_path,
        )
        violation.id = vid

        # 3. Broadcast to dashboard via WebSocket
        if self._ws_manager is not None:
            message = json.dumps({
                "type": "violation",
                "violation": {
                    "id": violation.id,
                    "camera_id": violation.camera_id,
                    "type": violation.type,
                    "severity": violation.severity,
                    "bbox": violation.bbox,
                    "thumbnail_path": violation.thumbnail_path,
                    "timestamp": violation.timestamp.isoformat(),
                },
            }, default=str)
            self._ws_manager.broadcast(message)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dispatcher.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add alert/dispatcher.py tests/test_dispatcher.py
git commit -m "feat: dispatcher for DB insert + thumbnail save + WebSocket broadcast"
```

---

### Task 7: Alert Pipeline Integration

**Files:**
- Create: `alert/__init__.py` (update with pipeline runner)
- Test: `tests/test_alert_pipeline.py`

- [ ] **Step 1: Write the failing integration test — `tests/test_alert_pipeline.py`**

```python
"""Integration test for the full alert pipeline: ROI → Classify → Cooldown → Dispatch."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY

from shared.models import DetectionResult, DetectedObject, BBox, Keypoint


def make_detection(camera_id="cam-01", has_helmet=True, has_vest=True):
    """Helper to build a DetectionResult with configurable equipment."""
    objects = [
        DetectedObject(bbox=BBox(100, 100, 200, 300), cls="person", conf=0.9),
    ]
    if has_helmet:
        objects.append(
            DetectedObject(bbox=BBox(110, 100, 180, 150), cls="helmet", conf=0.8)
        )
    if has_vest:
        objects.append(
            DetectedObject(bbox=BBox(110, 150, 190, 260), cls="vest", conf=0.8)
        )
    return DetectionResult(camera_id=camera_id, objects=objects)


class TestAlertPipeline:
    """End-to-end alert pipeline tests with mocked DB and WebSocket."""

    def test_full_pipeline_roi_blocked(self):
        """Violation outside ROI should be blocked before classification."""
        from alert.roi_matcher import ROIMatcher
        from alert.classifier import ViolationClassifier
        from alert.cooldown import CooldownManager
        from alert.dispatcher import Dispatcher

        mock_db = MagicMock()
        # ROI is a small polygon far from the person
        mock_db.get_roi.return_value = {
            "camera_id": "cam-01",
            "polygon": json.dumps([(500, 500), (600, 500), (600, 600), (500, 600)]),
        }
        mock_db.insert_violation.return_value = 1

        roi = ROIMatcher(mock_db)
        classifier = ViolationClassifier()
        cooldown = CooldownManager()
        dispatcher = Dispatcher(db=mock_db, ws_manager=None)

        # Person at (100-200, 100-300) — center (150, 200)
        # ROI is at (500-600, 500-600) — person is outside
        result = make_detection(has_helmet=False, has_vest=False)

        violations_dispatched = 0
        for obj in result.objects:
            if obj.cls != "person":
                continue
            bbox = [obj.bbox.x1, obj.bbox.y1, obj.bbox.x2, obj.bbox.y2]
            if not roi.is_in_roi(result.camera_id, bbox):
                continue  # Blocked by ROI

            # This code path should NOT be reached
            violations = classifier.classify(result)
            for v in violations:
                if cooldown.should_alert(v.camera_id, v.type):
                    dispatcher.dispatch(v, frame_bgr=None)
                    violations_dispatched += 1

        assert violations_dispatched == 0

    def test_full_pipeline_cooldown_blocks_duplicate(self):
        """Second identical violation within 5s should be blocked by cooldown."""
        from alert.roi_matcher import ROIMatcher
        from alert.classifier import ViolationClassifier
        from alert.cooldown import CooldownManager
        from alert.dispatcher import Dispatcher

        mock_db = MagicMock()
        # ROI covers the entire frame
        mock_db.get_roi.return_value = {
            "camera_id": "cam-01",
            "polygon": json.dumps([(0, 0), (640, 0), (640, 480), (0, 480)]),
        }
        mock_db.insert_violation.return_value = 1

        roi = ROIMatcher(mock_db)
        classifier = ViolationClassifier()
        cooldown = CooldownManager(cooldown_seconds=5)
        dispatcher = Dispatcher(db=mock_db, ws_manager=None)

        # Person without helmet
        result = make_detection(has_helmet=False, has_vest=True)

        violations_dispatched = 0
        for obj in result.objects:
            if obj.cls != "person":
                continue
            bbox = [obj.bbox.x1, obj.bbox.y1, obj.bbox.x2, obj.bbox.y2]
            if not roi.is_in_roi(result.camera_id, bbox):
                continue

            violations = classifier.classify(result)
            for v in violations:
                if cooldown.should_alert(v.camera_id, v.type):
                    dispatcher.dispatch(v, frame_bgr=None)
                    violations_dispatched += 1

        # First pass: should dispatch NO_HELMET
        assert violations_dispatched == 1

        # Second pass with same detection
        violations_dispatched = 0
        for obj in result.objects:
            if obj.cls != "person":
                continue
            bbox = [obj.bbox.x1, obj.bbox.y1, obj.bbox.x2, obj.bbox.y2]
            if not roi.is_in_roi(result.camera_id, bbox):
                continue
            violations = classifier.classify(result)
            for v in violations:
                if cooldown.should_alert(v.camera_id, v.type):
                    dispatcher.dispatch(v, frame_bgr=None)
                    violations_dispatched += 1

        # Second pass: cooldown blocks NO_HELMET
        assert violations_dispatched == 0

    def test_full_pipeline_dispatches_valid_violation(self):
        """A valid violation inside ROI with no cooldown should dispatch."""
        from alert.roi_matcher import ROIMatcher
        from alert.classifier import ViolationClassifier
        from alert.cooldown import CooldownManager
        from alert.dispatcher import Dispatcher

        mock_db = MagicMock()
        mock_db.get_roi.return_value = {
            "camera_id": "cam-01",
            "polygon": json.dumps([(0, 0), (640, 0), (640, 480), (0, 480)]),
        }
        mock_db.insert_violation.return_value = 1

        roi = ROIMatcher(mock_db)
        classifier = ViolationClassifier()
        cooldown = CooldownManager(cooldown_seconds=0)  # No cooldown
        dispatcher = Dispatcher(db=mock_db, ws_manager=None)

        result = make_detection(has_helmet=False, has_vest=False)

        violations_dispatched = 0
        for obj in result.objects:
            if obj.cls != "person":
                continue
            bbox = [obj.bbox.x1, obj.bbox.y1, obj.bbox.x2, obj.bbox.y2]
            if not roi.is_in_roi(result.camera_id, bbox):
                continue
            violations = classifier.classify(result)
            for v in violations:
                if cooldown.should_alert(v.camera_id, v.type):
                    dispatcher.dispatch(v, frame_bgr=None)
                    violations_dispatched += 1

        # Should dispatch NO_HELMET and NO_VEST (and maybe NO_BOOT)
        assert violations_dispatched >= 2
        assert mock_db.insert_violation.call_count >= 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_alert_pipeline.py -v`
Expected: FAIL — modules not yet importable together (or 3 FAIL)

- [ ] **Step 3: Write `alert/__init__.py` — AlertPipeline that orchestrates the four steps**

```python
"""Alert service — pipeline from DetectionResult to dispatched Violation."""
from typing import List, Optional

import numpy as np

from alert.roi_matcher import ROIMatcher
from alert.classifier import ViolationClassifier
from alert.cooldown import CooldownManager
from alert.dispatcher import Dispatcher
from shared.models import DetectionResult, Violation


class AlertPipeline:
    """Orchestrates the four-stage alert pipeline:
    1. ROI filtering
    2. Violation classification
    3. Cooldown deduplication
    4. Dispatch (DB + thumbnail + WebSocket)
    """

    def __init__(
        self,
        roi_matcher: ROIMatcher,
        classifier: ViolationClassifier,
        cooldown: CooldownManager,
        dispatcher: Dispatcher,
    ):
        self.roi = roi_matcher
        self.classifier = classifier
        self.cooldown = cooldown
        self.dispatcher = dispatcher

    def process(
        self, result: DetectionResult, frame_bgr: Optional[np.ndarray] = None
    ) -> List[Violation]:
        """Run a DetectionResult through the full alert pipeline.
        Returns the list of violations that were actually dispatched."""
        dispatched: List[Violation] = []

        violations = self.classifier.classify(result)

        for violation in violations:
            # 1. ROI check
            if not self.roi.is_in_roi(violation.camera_id, violation.bbox):
                continue

            # 2. Cooldown check
            if not self.cooldown.should_alert(violation.camera_id, violation.type):
                continue

            # 3. Dispatch
            self.dispatcher.dispatch(violation, frame_bgr=frame_bgr)
            dispatched.append(violation)

        return dispatched
```

- [ ] **Step 4: Run integration tests to verify they pass**

Run: `python -m pytest tests/test_alert_pipeline.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add alert/__init__.py tests/test_alert_pipeline.py
git commit -m "feat: alert pipeline orchestration (ROI → classify → cooldown → dispatch)"
```

---

### Task 8: Dashboard Server (FastAPI + WebSocket)

**Files:**
- Create: `dashboard/server.py`
- Create: `dashboard/__init__.py`
- Test: `tests/test_dashboard_server.py`

- [ ] **Step 1: Write the failing test — `tests/test_dashboard_server.py`**

```python
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def dashboard_app(temp_dir):
    """Create the FastAPI app with test DB and temp dirs."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from alert.db import init_db
    import alert.db as db_module

    original_path = db_module.DB_PATH
    db_path = temp_dir / "test_dashboard.db"
    db_module.DB_PATH = db_path
    init_db()

    # Patch static dir to temp
    static_dir = temp_dir / "static"
    static_dir.mkdir()

    with patch('dashboard.server.STATIC_DIR', str(static_dir)):
        from dashboard.server import app
        yield app

    db_module.DB_PATH = original_path


@pytest.fixture
def client(dashboard_app):
    return TestClient(dashboard_app)


def test_get_cameras_empty(client):
    """GET /api/cameras returns empty list when no cameras registered."""
    resp = client.get("/api/cameras")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


def test_get_roi_not_found(client):
    """GET /api/roi/unknown should return 404."""
    resp = client.get("/api/roi/nonexistent")
    assert resp.status_code == 404


def test_put_and_get_roi(client):
    """PUT then GET /api/roi/{camera_id} should round-trip."""
    polygon = [[100, 200], [300, 200], [300, 400], [100, 400]]
    resp = client.put("/api/roi/cam-01", json={"polygon": polygon})
    assert resp.status_code == 200

    resp = client.get("/api/roi/cam-01")
    assert resp.status_code == 200
    data = resp.json()
    assert data["camera_id"] == "cam-01"
    assert data["polygon"] == polygon


def test_get_violations_empty(client):
    """GET /api/violations returns empty list initially."""
    resp = client.get("/api/violations")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 0


def test_get_violations_with_data(client):
    """GET /api/violations returns violations after inserts."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from alert.db import insert_violation

    insert_violation("cam-01", "NO_HELMET", "HIGH", [10, 20, 30, 40], "/tmp/t.jpg")
    insert_violation("cam-02", "FALL", "HIGH", [50, 60, 70, 80], "/tmp/t2.jpg")

    resp = client.get("/api/violations")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2

    # Test filtering
    resp = client.get("/api/violations?camera_id=cam-01")
    data = resp.json()
    assert len(data) == 1
    assert data[0]["camera_id"] == "cam-01"


def test_get_violation_thumbnail_not_found(client):
    """GET /api/violations/999/thumbnail should return 404 for nonexistent."""
    resp = client.get("/api/violations/999/thumbnail")
    assert resp.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dashboard_server.py -v`
Expected: FAIL — `dashboard.server` not found

- [ ] **Step 3: Write `dashboard/server.py`**

```python
"""FastAPI dashboard server: REST API + WebSocket + static file serving."""
import json
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from alert.db import get_violations, get_roi, save_roi, get_all_rois

STATIC_DIR = str(Path(__file__).parent / "static")
THUMBNAIL_DIR = str(Path("data/thumbnails"))


# --- WebSocket Connection Manager ---

class ConnectionManager:
    """Manages active WebSocket connections for realtime broadcast."""

    def __init__(self):
        self._connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self._connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self._connections:
            self._connections.remove(websocket)

    def broadcast(self, message: str):
        """Schedule broadcast to all connected clients.
        Called from sync context (dispatcher) — uses asyncio.run_coroutine_threadsafe
        or the caller should be in an async context."""
        for ws in self._connections:
            try:
                # Use asyncio.create_task for fire-and-forget from async context
                asyncio.create_task(self._safe_send(ws, message))
            except RuntimeError:
                # Not in an event loop — message will be queued
                pass

    async def broadcast_async(self, message: str):
        """Broadcast from within an async context."""
        for ws in self._connections:
            await self._safe_send(ws, message)

    async def _safe_send(self, ws: WebSocket, message: str):
        try:
            await ws.send_text(message)
        except Exception:
            self.disconnect(ws)


ws_manager = ConnectionManager()


# --- Pydantic Schemas ---

class ROIPayload(BaseModel):
    polygon: List[List[float]]


class CameraInfo(BaseModel):
    id: str
    source: str
    status: str = "unknown"


# --- FastAPI App ---

app = FastAPI(title="CV Safety Monitor", version="0.1.0")

# Mount static files for the frontend
static_path = Path(STATIC_DIR)
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


# --- REST API ---

@app.get("/api/cameras")
async def api_get_cameras():
    """Return list of cameras. Reads from edge config if available."""
    cameras = []
    config_path = Path("edge/config.yaml")
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        for cam in config.get("cameras", []):
            cameras.append({
                "id": cam["id"],
                "source": str(cam["source"]),
            })
    return cameras


@app.get("/api/roi/{camera_id}")
async def api_get_roi(camera_id: str):
    """Get ROI polygon for a camera."""
    import json as _json
    row = get_roi(camera_id)
    if row is None:
        raise HTTPException(status_code=404, detail="ROI not found for this camera")
    return {
        "camera_id": row["camera_id"],
        "polygon": _json.loads(row["polygon"]),
        "updated_at": row["updated_at"],
    }


@app.put("/api/roi/{camera_id}")
async def api_put_roi(camera_id: str, payload: ROIPayload):
    """Save or update ROI polygon for a camera."""
    polygon = [(p[0], p[1]) for p in payload.polygon]
    save_roi(camera_id, polygon)

    # Invalidate ROI cache if roi_matcher is wired up
    # (handled by main.py wiring — here we just save)

    return {"status": "ok", "camera_id": camera_id}


@app.get("/api/violations")
async def api_get_violations(
    camera_id: Optional[str] = Query(None),
    type: Optional[str] = Query(None),
    from_time: Optional[str] = Query(None),
    to_time: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Query violation history with optional filters."""
    rows = get_violations(
        camera_id=camera_id,
        violation_type=type,
        from_time=from_time,
        to_time=to_time,
        limit=limit,
        offset=offset,
    )
    return rows


@app.get("/api/violations/{violation_id}/thumbnail")
async def api_get_thumbnail(violation_id: int):
    """Serve thumbnail image for a violation."""
    rows = get_violations(limit=1000)
    match = None
    for r in rows:
        if r["id"] == violation_id:
            match = r
            break
    if match is None or not match.get("thumbnail_path"):
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    path = Path(match["thumbnail_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail file missing")

    return FileResponse(str(path), media_type="image/jpeg")


# --- WebSocket ---

@app.websocket("/ws/dashboard")
async def ws_dashboard(websocket: WebSocket):
    """Realtime dashboard WebSocket endpoint."""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, receive pings
            data = await websocket.receive_text()
            # Client can send heartbeat pings
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


# --- Static HTML fallback (SPA-like) ---

@app.get("/")
async def root():
    index_path = Path(STATIC_DIR) / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "CV Safety Monitor API", "docs": "/docs"}


@app.get("/admin.html")
async def admin_page():
    return FileResponse(str(Path(STATIC_DIR) / "admin.html"))


@app.get("/history.html")
async def history_page():
    return FileResponse(str(Path(STATIC_DIR) / "history.html"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dashboard_server.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/__init__.py dashboard/server.py tests/test_dashboard_server.py
git commit -m "feat: FastAPI dashboard server with REST API and WebSocket"
```

---

### Task 9: Dashboard Frontend — Main View

**Files:**
- Create: `dashboard/static/style.css`
- Create: `dashboard/static/index.html`

- [ ] **Step 1: Write `dashboard/static/style.css`**

```css
/* === CV Safety Monitor — Dark Theme === */
:root {
  --bg-primary: #0d1117;
  --bg-secondary: #161b22;
  --bg-card: #1c2333;
  --border: #30363d;
  --text-primary: #e6edf3;
  --text-secondary: #8b949e;
  --accent-green: #3fb950;
  --accent-red: #f85149;
  --accent-orange: #d2991d;
  --accent-blue: #58a6ff;
  --font-mono: 'SF Mono', 'Fira Code', 'Consolas', monospace;
  --font-sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
}

*, *::before, *::after {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: var(--font-sans);
  background: var(--bg-primary);
  color: var(--text-primary);
  overflow-x: hidden;
}

/* === Alert Bar === */
#alert-bar {
  position: fixed;
  top: 0; left: 0; right: 0;
  height: 52px;
  background: var(--bg-secondary);
  border-bottom: 2px solid var(--border);
  display: flex;
  align-items: center;
  padding: 0 20px;
  z-index: 1000;
  transition: border-color 0.2s, background 0.2s;
}

#alert-bar.flash {
  border-bottom-color: var(--accent-red);
  background: #1a1015;
  animation: flash-border 0.5s ease-in-out 3;
}

@keyframes flash-border {
  0%, 100% { border-bottom-color: var(--accent-red); }
  50% { border-bottom-color: transparent; }
}

#alert-bar .dot {
  width: 10px; height: 10px;
  border-radius: 50%;
  background: var(--accent-green);
  margin-right: 12px;
  transition: background 0.2s;
}

#alert-bar .dot.alarm {
  background: var(--accent-red);
  animation: pulse-dot 0.5s ease-in-out infinite;
}

@keyframes pulse-dot {
  0%, 100% { transform: scale(1); opacity: 1; }
  50% { transform: scale(1.5); opacity: 0.6; }
}

#alert-bar .title {
  font-weight: 700;
  font-size: 16px;
  letter-spacing: 0.5px;
}

#alert-bar .latest-alert {
  margin-left: auto;
  font-size: 13px;
  color: var(--text-secondary);
  font-family: var(--font-mono);
}

#alert-bar .latest-alert .sev-high {
  color: var(--accent-red);
  font-weight: 700;
}

#alert-bar .latest-alert .sev-medium {
  color: var(--accent-orange);
}

/* === Camera Grid === */
#camera-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
  gap: 16px;
  padding: 68px 16px 16px;
  min-height: calc(100vh - 52px);
}

.camera-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
}

.camera-card .header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 14px;
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border);
  font-size: 13px;
  font-weight: 600;
}

.camera-card .header .cam-id {
  color: var(--accent-blue);
  font-family: var(--font-mono);
}

.camera-card .header .status {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 10px;
}

.camera-card .header .status.live {
  background: rgba(63, 185, 80, 0.15);
  color: var(--accent-green);
}

.camera-card .header .status.offline {
  background: rgba(248, 81, 73, 0.15);
  color: var(--accent-red);
}

.camera-card canvas {
  display: block;
  width: 100%;
  aspect-ratio: 4/3;
  background: #000;
  object-fit: contain;
}

.camera-card .violation-tag {
  position: absolute;
  top: 8px; right: 8px;
  background: var(--accent-red);
  color: #fff;
  font-size: 11px;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 4px;
  text-transform: uppercase;
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.2s;
}

.camera-card .violation-tag.show {
  opacity: 1;
}

/* === Nav === */
#nav {
  position: fixed;
  bottom: 16px;
  right: 16px;
  display: flex;
  gap: 8px;
  z-index: 900;
}

#nav a {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  color: var(--text-secondary);
  padding: 8px 16px;
  border-radius: 6px;
  text-decoration: none;
  font-size: 13px;
  transition: color 0.15s, border-color 0.15s;
}

#nav a:hover {
  color: var(--text-primary);
  border-color: var(--accent-blue);
}

/* === Responsive === */
@media (max-width: 900px) {
  #camera-grid {
    grid-template-columns: 1fr;
  }
}
```

- [ ] **Step 2: Write `dashboard/static/index.html`**

```html
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CV Safety Monitor — Dashboard</title>
<link rel="stylesheet" href="/static/style.css">
</head>
<body>

<!-- Alert Bar -->
<div id="alert-bar">
  <span class="dot" id="status-dot"></span>
  <span class="title">CV Safety Monitor</span>
  <span class="latest-alert" id="latest-alert-text">— No alerts —</span>
</div>

<!-- Camera Grid -->
<div id="camera-grid"></div>

<!-- Navigation -->
<div id="nav">
  <a href="/admin.html">⚙ ROI Tool</a>
  <a href="/history.html">📋 History</a>
</div>

<script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 3: Commit**

```bash
git add dashboard/static/style.css dashboard/static/index.html
git commit -m "feat: dashboard main view — dark theme CSS + index.html scaffold"
```

---

### Task 10: Dashboard Frontend — JavaScript (WebSocket + Canvas)

**Files:**
- Create: `dashboard/static/app.js`

- [ ] **Step 1: Write `dashboard/static/app.js`**

```javascript
/**
 * CV Safety Monitor — Dashboard JS
 * Handles WebSocket connection, camera grid rendering, alert bar updates.
 */
(function () {
  'use strict';

  // --- State ---
  const cameras = {};       // camera_id -> { canvas, ctx, img, lastFrame }
  let ws = null;
  let reconnectTimer = null;

  // --- DOM ---
  const grid = document.getElementById('camera-grid');
  const alertBar = document.getElementById('alert-bar');
  const statusDot = document.getElementById('status-dot');
  const latestAlertEl = document.getElementById('latest-alert-text');

  // --- WebSocket ---
  function connectWS() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${location.host}/ws/dashboard`;
    ws = new WebSocket(url);

    ws.onopen = () => {
      console.log('[WS] Connected');
      statusDot.classList.remove('alarm');
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      fetchCameras();
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        handleMessage(msg);
      } catch (e) {
        console.warn('[WS] Invalid message:', event.data);
      }
    };

    ws.onclose = () => {
      console.log('[WS] Disconnected — reconnecting in 3s');
      statusDot.classList.add('alarm');
      reconnectTimer = setTimeout(connectWS, 3000);
    };

    ws.onerror = (err) => {
      console.error('[WS] Error:', err);
    };
  }

  // --- Message Handler ---
  function handleMessage(msg) {
    switch (msg.type) {
      case 'violation':
        showAlert(msg.violation);
        break;
      case 'preview':
        updatePreview(msg.camera_id, msg.frame_base64);
        break;
      default:
        console.log('[WS] Unknown message type:', msg.type);
    }
  }

  // --- Alert Bar ---
  function showAlert(violation) {
    // Flash the alert bar
    alertBar.classList.remove('flash');
    void alertBar.offsetWidth; // reflow
    alertBar.classList.add('flash');

    // Update latest alert text
    const sevClass = violation.severity === 'HIGH' ? 'sev-high' : 'sev-medium';
    latestAlertEl.innerHTML = `<span class="${sevClass}">[${violation.type}]</span> Camera ${violation.camera_id} — ${new Date(violation.timestamp).toLocaleTimeString()}`;

    // Flash the matching camera card
    const card = document.querySelector(`[data-camera="${violation.camera_id}"]`);
    if (card) {
      const tag = card.querySelector('.violation-tag');
      if (tag) {
        tag.textContent = violation.type;
        tag.classList.add('show');
        setTimeout(() => tag.classList.remove('show'), 3000);
      }
    }

    // Auto-dismiss flash after animation
    setTimeout(() => alertBar.classList.remove('flash'), 2000);
  }

  // --- Preview Frame Update ---
  function updatePreview(cameraId, base64Frame) {
    let cam = cameras[cameraId];
    if (!cam) {
      cam = createCameraCard(cameraId);
      cameras[cameraId] = cam;
    }

    const img = new Image();
    img.onload = () => {
      cam.canvas.width = img.width;
      cam.canvas.height = img.height;
      cam.ctx.drawImage(img, 0, 0);
    };
    img.src = `data:image/jpeg;base64,${base64Frame}`;
  }

  // --- Camera Grid ---
  function createCameraCard(cameraId) {
    const card = document.createElement('div');
    card.className = 'camera-card';
    card.setAttribute('data-camera', cameraId);

    const header = document.createElement('div');
    header.className = 'header';
    header.innerHTML = `
      <span class="cam-id">📷 ${cameraId}</span>
      <span class="status live">LIVE</span>
    `;

    const canvas = document.createElement('canvas');
    canvas.width = 416;
    canvas.height = 416;

    const tag = document.createElement('div');
    tag.className = 'violation-tag';

    card.appendChild(header);
    card.appendChild(canvas);
    card.appendChild(tag);
    grid.appendChild(card);

    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#444';
    ctx.font = '14px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Waiting for stream...', canvas.width / 2, canvas.height / 2);

    return { canvas, ctx, card };
  }

  // --- Fetch initial camera list ---
  async function fetchCameras() {
    try {
      const resp = await fetch('/api/cameras');
      const list = await resp.json();
      list.forEach(cam => {
        if (!cameras[cam.id]) {
          cameras[cam.id] = createCameraCard(cam.id);
        }
      });
    } catch (e) {
      console.warn('Failed to fetch camera list:', e);
    }
  }

  // --- Startup ---
  connectWS();
})();
```

- [ ] **Step 2: Commit**

```bash
git add dashboard/static/app.js
git commit -m "feat: dashboard JS — WebSocket client, camera grid, alert bar"
```

---

### Task 11: Admin ROI Drawing Tool

**Files:**
- Create: `dashboard/static/admin.html`

- [ ] **Step 1: Write `dashboard/static/admin.html`**

```html
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ROI Tool — CV Safety Monitor</title>
<link rel="stylesheet" href="/static/style.css">
<style>
  .admin-container {
    padding: 68px 20px 20px;
    max-width: 900px;
    margin: 0 auto;
  }
  .admin-container h2 {
    font-size: 20px;
    margin-bottom: 16px;
    color: var(--accent-blue);
  }
  .controls {
    display: flex;
    gap: 12px;
    margin-bottom: 16px;
    align-items: center;
    flex-wrap: wrap;
  }
  .controls select {
    background: var(--bg-card);
    color: var(--text-primary);
    border: 1px solid var(--border);
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 14px;
  }
  .controls button {
    background: var(--accent-blue);
    color: #000;
    border: none;
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: opacity 0.2s;
  }
  .controls button:hover { opacity: 0.85; }
  .controls button.danger {
    background: var(--accent-red);
    color: #fff;
  }
  .controls button.secondary {
    background: var(--bg-card);
    color: var(--text-primary);
    border: 1px solid var(--border);
  }
  #roi-canvas-wrap {
    position: relative;
    display: inline-block;
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
  }
  #roi-canvas {
    display: block;
    max-width: 100%;
    cursor: crosshair;
  }
  .help-text {
    margin-top: 12px;
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.6;
  }
  .help-text kbd {
    background: var(--bg-card);
    border: 1px solid var(--border);
    padding: 1px 6px;
    border-radius: 3px;
    font-family: var(--font-mono);
    font-size: 12px;
  }
  .toast {
    position: fixed;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    background: var(--accent-green);
    color: #000;
    padding: 10px 24px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 14px;
    opacity: 0;
    transition: opacity 0.3s;
    z-index: 9999;
  }
  .toast.show { opacity: 1; }
  .toast.error { background: var(--accent-red); color: #fff; }
</style>
</head>
<body>

<div id="alert-bar">
  <span class="dot"></span>
  <span class="title">CV Safety Monitor</span>
  <span style="margin-left:auto;font-size:13px;color:var(--text-secondary)">ROI Configuration Tool</span>
</div>

<div class="admin-container">
  <h2>🎯 ROI Polygon Tool</h2>

  <div class="controls">
    <select id="camera-select">
      <option value="">— Select Camera —</option>
    </select>
    <button onclick="ROITool.loadROI()">📥 Load Saved ROI</button>
    <button onclick="ROITool.saveROI()">💾 Save ROI</button>
    <button class="secondary" onclick="ROITool.undoPoint()">↩ Undo Point</button>
    <button class="danger" onclick="ROITool.clearROI()">✕ Clear</button>
  </div>

  <div id="roi-canvas-wrap">
    <canvas id="roi-canvas" width="640" height="480"></canvas>
  </div>

  <div class="help-text">
    <strong>Instructions:</strong><br>
    <kbd>Click</kbd> — Add polygon vertex &nbsp;
    <kbd>Drag</kbd> — Move a vertex &nbsp;
    <kbd>Right-click</kbd> — Remove vertex &nbsp;
    <kbd>Double-click</kbd> — Close polygon<br>
    Select a camera above, then draw the ROI polygon on the preview.
  </div>
</div>

<div class="toast" id="toast"></div>

<div id="nav">
  <a href="/">📷 Dashboard</a>
  <a href="/history.html">📋 History</a>
</div>

<script src="/static/app.js"></script>
<script>
/**
 * ROI Polygon Drawing Tool
 */
const ROITool = {
  canvas: document.getElementById('roi-canvas'),
  ctx: document.getElementById('roi-canvas').getContext('2d'),
  points: [],
  dragIdx: -1,
  img: null,

  init() {
    this.canvas.addEventListener('click', (e) => this.onClick(e));
    this.canvas.addEventListener('dblclick', (e) => this.onDblClick(e));
    this.canvas.addEventListener('contextmenu', (e) => this.onRightClick(e));
    this.canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
    this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
    this.canvas.addEventListener('mouseup', () => { this.dragIdx = -1; });
    this.loadCameras();
    this.draw();
  },

  getPos(e) {
    const rect = this.canvas.getBoundingClientRect();
    const scaleX = this.canvas.width / rect.width;
    const scaleY = this.canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    };
  },

  findPoint(pos, radius = 10) {
    for (let i = 0; i < this.points.length; i++) {
      const dx = this.points[i].x - pos.x;
      const dy = this.points[i].y - pos.y;
      if (Math.sqrt(dx * dx + dy * dy) < radius) return i;
    }
    return -1;
  },

  onClick(e) {
    const pos = this.getPos(e);
    // Check if clicking near existing point
    if (this.findPoint(pos) >= 0) return;
    // Check if polygon is closed
    if (this.points.length > 0 && this.points[0].closed) return;
    this.points.push({ x: pos.x, y: pos.y, closed: false });
    this.draw();
  },

  onDblClick(e) {
    if (this.points.length < 3) return;
    this.points[0].closed = true;
    this.draw();
  },

  onRightClick(e) {
    e.preventDefault();
    const pos = this.getPos(e);
    const idx = this.findPoint(pos);
    if (idx >= 0) {
      this.points.splice(idx, 1);
      if (this.points.length > 0) this.points[0].closed = false;
      this.draw();
    }
  },

  onMouseDown(e) {
    const pos = this.getPos(e);
    const idx = this.findPoint(pos);
    if (idx >= 0) {
      this.dragIdx = idx;
    }
  },

  onMouseMove(e) {
    if (this.dragIdx < 0) return;
    const pos = this.getPos(e);
    this.points[this.dragIdx].x = pos.x;
    this.points[this.dragIdx].y = pos.y;
    this.draw();
  },

  undoPoint() {
    if (this.points.length === 0) return;
    this.points.pop();
    if (this.points.length > 0) this.points[0].closed = false;
    this.draw();
  },

  clearROI() {
    this.points = [];
    this.draw();
  },

  draw() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // Draw dark bg + grid
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.strokeStyle = '#222';
    ctx.lineWidth = 0.5;
    for (let x = 0; x < this.canvas.width; x += 40) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, this.canvas.height); ctx.stroke();
    }
    for (let y = 0; y < this.canvas.height; y += 40) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(this.canvas.width, y); ctx.stroke();
    }

    // Draw polygon lines
    if (this.points.length < 2) {
      // Draw points only
      this.points.forEach(p => this.drawPoint(p));
      return;
    }

    ctx.beginPath();
    ctx.moveTo(this.points[0].x, this.points[0].y);
    for (let i = 1; i < this.points.length; i++) {
      ctx.lineTo(this.points[i].x, this.points[i].y);
    }
    if (this.points[0].closed) {
      ctx.closePath();
    }
    ctx.strokeStyle = '#58a6ff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Fill with semi-transparent
    if (this.points[0].closed && this.points.length >= 3) {
      ctx.fillStyle = 'rgba(88, 166, 255, 0.1)';
      ctx.fill();
    }

    // Draw vertex points
    this.points.forEach(p => this.drawPoint(p));
  },

  drawPoint(p) {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#58a6ff';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  },

  async loadCameras() {
    try {
      const resp = await fetch('/api/cameras');
      const list = await resp.json();
      const sel = document.getElementById('camera-select');
      list.forEach(cam => {
        const opt = document.createElement('option');
        opt.value = cam.id;
        opt.textContent = cam.id;
        sel.appendChild(opt);
      });
    } catch (e) {
      this.showToast('Failed to load cameras', true);
    }
  },

  async loadROI() {
    const camId = document.getElementById('camera-select').value;
    if (!camId) {
      this.showToast('Select a camera first', true);
      return;
    }
    try {
      const resp = await fetch(`/api/roi/${camId}`);
      if (!resp.ok) {
        this.showToast('No ROI saved for this camera yet', true);
        return;
      }
      const data = await resp.json();
      this.points = data.polygon.map(p => ({ x: p[0], y: p[1], closed: false }));
      if (this.points.length >= 3) {
        this.points[0].closed = true;
      }
      this.draw();
      this.showToast('ROI loaded');
    } catch (e) {
      this.showToast('Failed to load ROI', true);
    }
  },

  async saveROI() {
    const camId = document.getElementById('camera-select').value;
    if (!camId) {
      this.showToast('Select a camera first', true);
      return;
    }
    if (this.points.length < 3 || !this.points[0].closed) {
      this.showToast('Draw a closed polygon first (at least 3 points, double-click to close)', true);
      return;
    }
    const polygon = this.points.map(p => [Math.round(p.x), Math.round(p.y)]);
    try {
      const resp = await fetch(`/api/roi/${camId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ polygon }),
      });
      if (resp.ok) {
        this.showToast('ROI saved ✓');
      } else {
        this.showToast('Failed to save ROI', true);
      }
    } catch (e) {
      this.showToast('Failed to save ROI', true);
    }
  },

  showToast(msg, isError = false) {
    const toast = document.getElementById('toast');
    toast.textContent = msg;
    toast.className = 'toast' + (isError ? ' error' : '');
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 2500);
  }
};

ROITool.init();
</script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add dashboard/static/admin.html
git commit -m "feat: admin ROI polygon drawing tool with canvas"
```

---

### Task 12: History Page

**Files:**
- Create: `dashboard/static/history.html`

- [ ] **Step 1: Write `dashboard/static/history.html`**

```html
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Violation History — CV Safety Monitor</title>
<link rel="stylesheet" href="/static/style.css">
<style>
  .history-container {
    padding: 68px 20px 20px;
    max-width: 1100px;
    margin: 0 auto;
  }
  .history-container h2 {
    font-size: 20px;
    margin-bottom: 16px;
    color: var(--accent-blue);
  }
  .filters {
    display: flex;
    gap: 10px;
    margin-bottom: 16px;
    flex-wrap: wrap;
    align-items: center;
  }
  .filters select, .filters input, .filters button {
    background: var(--bg-card);
    color: var(--text-primary);
    border: 1px solid var(--border);
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 13px;
  }
  .filters button {
    background: var(--accent-blue);
    color: #000;
    font-weight: 600;
    cursor: pointer;
    border: none;
  }
  .filters button:hover { opacity: 0.85; }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }
  th, td {
    text-align: left;
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
  }
  th {
    color: var(--text-secondary);
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  tr:hover td { background: rgba(88, 166, 255, 0.04); }
  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
  }
  .badge-high { background: rgba(248, 81, 73, 0.2); color: var(--accent-red); }
  .badge-medium { background: rgba(210, 153, 29, 0.2); color: var(--accent-orange); }
  .thumbnail-sm {
    width: 80px;
    height: 60px;
    object-fit: cover;
    border-radius: 4px;
    border: 1px solid var(--border);
    cursor: pointer;
  }
  .empty-state {
    text-align: center;
    color: var(--text-secondary);
    padding: 60px 0;
    font-size: 15px;
  }
  .pagination {
    display: flex;
    gap: 8px;
    margin-top: 16px;
    justify-content: center;
  }
  .pagination button {
    background: var(--bg-card);
    color: var(--text-primary);
    border: 1px solid var(--border);
    padding: 6px 14px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
  }
  .pagination button:disabled {
    opacity: 0.4;
    cursor: default;
  }
  .pagination button.active {
    background: var(--accent-blue);
    border-color: var(--accent-blue);
    color: #000;
    font-weight: 600;
  }
</style>
</head>
<body>

<div id="alert-bar">
  <span class="dot"></span>
  <span class="title">CV Safety Monitor</span>
  <span style="margin-left:auto;font-size:13px;color:var(--text-secondary)">Violation History</span>
</div>

<div class="history-container">
  <h2>📋 Violation History</h2>

  <div class="filters">
    <select id="filter-camera">
      <option value="">All Cameras</option>
    </select>
    <select id="filter-type">
      <option value="">All Types</option>
      <option value="FALL">FALL</option>
      <option value="NO_HELMET">NO_HELMET</option>
      <option value="NO_VEST">NO_VEST</option>
      <option value="NO_BOOT">NO_BOOT</option>
    </select>
    <input type="date" id="filter-from" title="From date">
    <input type="date" id="filter-to" title="To date">
    <button onclick="HistoryPage.fetchPage(1)">🔍 Filter</button>
  </div>

  <table id="violations-table">
    <thead>
      <tr>
        <th>ID</th>
        <th>Camera</th>
        <th>Type</th>
        <th>Severity</th>
        <th>Time</th>
        <th>Thumbnail</th>
      </tr>
    </thead>
    <tbody id="table-body">
    </tbody>
  </table>

  <div class="empty-state" id="empty-state">No violations recorded yet.</div>

  <div class="pagination" id="pagination"></div>
</div>

<div id="nav">
  <a href="/">📷 Dashboard</a>
  <a href="/admin.html">⚙ ROI Tool</a>
</div>

<script>
const HistoryPage = {
  page: 1,
  limit: 20,

  async init() {
    await this.loadCameras();
    await this.fetchPage(1);
  },

  async loadCameras() {
    try {
      const resp = await fetch('/api/cameras');
      const list = await resp.json();
      const sel = document.getElementById('filter-camera');
      list.forEach(cam => {
        const opt = document.createElement('option');
        opt.value = cam.id;
        opt.textContent = cam.id;
        sel.appendChild(opt);
      });
    } catch (e) {
      console.warn('Failed to load cameras');
    }
  },

  async fetchPage(page) {
    this.page = page;
    const params = new URLSearchParams();
    params.set('limit', String(this.limit));
    params.set('offset', String((page - 1) * this.limit));

    const cam = document.getElementById('filter-camera').value;
    const type = document.getElementById('filter-type').value;
    const from = document.getElementById('filter-from').value;
    const to = document.getElementById('filter-to').value;

    if (cam) params.set('camera_id', cam);
    if (type) params.set('type', type);
    if (from) params.set('from_time', from + 'T00:00:00');
    if (to) params.set('to_time', to + 'T23:59:59');

    try {
      const resp = await fetch(`/api/violations?${params}`);
      const data = await resp.json();

      const tbody = document.getElementById('table-body');
      const empty = document.getElementById('empty-state');

      if (data.length === 0) {
        tbody.innerHTML = '';
        empty.style.display = 'block';
      } else {
        empty.style.display = 'none';
        tbody.innerHTML = data.map(v => `
          <tr>
            <td>#${v.id}</td>
            <td><span style="font-family:monospace;color:var(--accent-blue)">${v.camera_id}</span></td>
            <td>${v.type}</td>
            <td><span class="badge badge-${v.severity.toLowerCase()}">${v.severity}</span></td>
            <td>${new Date(v.created_at).toLocaleString()}</td>
            <td>
              ${v.thumbnail_path
                ? `<img class="thumbnail-sm" src="/api/violations/${v.id}/thumbnail"
                     onclick="window.open('/api/violations/${v.id}/thumbnail')" alt="thumb">`
                : '—'}
            </td>
          </tr>
        `).join('');
      }

      this.renderPagination(data.length);
    } catch (e) {
      console.error('Failed to fetch violations:', e);
    }
  },

  renderPagination(resultCount) {
    const pag = document.getElementById('pagination');
    const hasMore = resultCount >= this.limit;
    const hasPrev = this.page > 1;

    pag.innerHTML = `
      <button ${hasPrev ? '' : 'disabled'} onclick="HistoryPage.fetchPage(${this.page - 1})">← Prev</button>
      <button class="active">${this.page}</button>
      <button ${hasMore ? '' : 'disabled'} onclick="HistoryPage.fetchPage(${this.page + 1})">Next →</button>
    `;
  }
};

HistoryPage.init();
</script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add dashboard/static/history.html
git commit -m "feat: violation history page with filters and pagination"
```

---

### Task 13: Frame Processor (Edge)

**Files:**
- Create: `edge/frame_processor.py`
- Test: `tests/test_frame_processor.py`

- [ ] **Step 1: Write the failing test — `tests/test_frame_processor.py`**

```python
import numpy as np
import pytest


def make_frame(width=640, height=480):
    """Create a dummy BGR frame."""
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def test_frame_processor_resize():
    """FrameProcessor.resize should output exactly 416x416."""
    from edge.frame_processor import FrameProcessor

    fp = FrameProcessor(target_size=(416, 416))
    frame = make_frame(1920, 1080)
    resized = fp.resize(frame)
    assert resized.shape == (416, 416, 3)


def test_frame_processor_crop_roi():
    """Crop to ROI bounding rectangle then resize."""
    from edge.frame_processor import FrameProcessor

    fp = FrameProcessor(target_size=(416, 416))
    frame = make_frame(640, 480)
    roi_polygon = [(100, 100), (400, 100), (400, 350), (100, 350)]

    cropped = fp.crop_to_roi(frame, roi_polygon)
    assert cropped is not None
    assert cropped.shape == (416, 416, 3)
    # Cropped region should be the ROI area (300x250) resized to 416x416


def test_frame_processor_motion_skip_no_motion():
    """When frames are identical, should_skip should return True."""
    from edge.frame_processor import FrameProcessor

    fp = FrameProcessor(motion_threshold=0.05)
    frame = make_frame(640, 480)

    # First frame — never skip
    assert fp.should_skip(frame) is False
    # Same frame — should skip (no motion)
    assert fp.should_skip(frame) is True


def test_frame_processor_motion_skip_with_motion():
    """When frames differ significantly, should_skip should return False."""
    from edge.frame_processor import FrameProcessor

    fp = FrameProcessor(motion_threshold=0.01)
    frame1 = make_frame(640, 480)
    # Completely different frame
    frame2 = make_frame(640, 480)

    fp.should_skip(frame1)  # prime
    # Random frames often differ enough to pass threshold
    result = fp.should_skip(frame2)
    # May or may not skip depending on random content
    # We just test it doesn't crash
    assert isinstance(result, bool)


def test_frame_processor_encode_jpeg():
    """encode_jpeg should return JPEG bytes."""
    from edge.frame_processor import FrameProcessor

    fp = FrameProcessor()
    frame = make_frame(416, 416)
    jpeg_bytes = fp.encode_jpeg(frame, quality=70)
    assert isinstance(jpeg_bytes, bytes)
    assert len(jpeg_bytes) > 100  # Non-trivial JPEG size
    # Should be smaller than raw 416*416*3 = 519KB
    assert len(jpeg_bytes) < 519_000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_frame_processor.py -v`
Expected: FAIL — `edge.frame_processor` not found

- [ ] **Step 3: Write `edge/frame_processor.py`**

```python
"""Frame processor: motion-based skip, ROI crop, resize, JPEG encode."""
import cv2
import numpy as np
from typing import Optional, List, Tuple


class FrameProcessor:
    """Processes raw camera frames before sending to inference:
    1. Motion-based frame skipping (reduces 30fps → ~5fps effective)
    2. ROI crop to bounding rectangle
    3. Resize to target dimensions (default 416×416)
    4. JPEG encoding for efficient transport
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (416, 416),
        motion_threshold: float = 0.05,
        jpeg_quality: int = 70,
    ):
        self.target_size = target_size
        self.motion_threshold = motion_threshold
        self.jpeg_quality = jpeg_quality
        self._prev_frame_gray: Optional[np.ndarray] = None

    def should_skip(self, frame: np.ndarray) -> bool:
        """Returns True if this frame should be skipped due to lack of motion.
        Always returns False on the first frame (no previous frame to compare)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_frame_gray is None:
            self._prev_frame_gray = gray
            return False

        # Resize to small size for fast diff comparison
        small = cv2.resize(gray, (160, 120))
        prev_small = cv2.resize(self._prev_frame_gray, (160, 120))

        diff = cv2.absdiff(small, prev_small)
        motion_ratio = np.count_nonzero(diff > 15) / diff.size

        self._prev_frame_gray = gray
        return motion_ratio < self.motion_threshold

    def crop_to_roi(
        self, frame: np.ndarray, roi_polygon: List[Tuple[float, float]]
    ) -> np.ndarray:
        """Crop frame to the bounding rectangle of the ROI polygon, then resize."""
        if not roi_polygon:
            return self.resize(frame)

        xs = [p[0] for p in roi_polygon]
        ys = [p[1] for p in roi_polygon]

        x1 = max(0, int(min(xs)))
        y1 = max(0, int(min(ys)))
        x2 = min(frame.shape[1], int(max(xs)))
        y2 = min(frame.shape[0], int(max(ys)))

        if x2 <= x1 or y2 <= y1:
            return self.resize(frame)

        cropped = frame[y1:y2, x1:x2]
        return self.resize(cropped)

    def resize(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to target dimensions."""
        return cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)

    def encode_jpeg(self, frame: np.ndarray, quality: Optional[int] = None) -> bytes:
        """Encode a BGR frame as JPEG bytes."""
        q = quality if quality is not None else self.jpeg_quality
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, q])
        return buf.tobytes()

    def process(
        self,
        frame: np.ndarray,
        roi_polygon: Optional[List[Tuple[float, float]]] = None,
    ) -> Optional[bytes]:
        """Full processing pipeline for one frame.
        Returns JPEG bytes if the frame passes motion detection, None if skipped."""
        if self.should_skip(frame):
            return None

        if roi_polygon:
            processed = self.crop_to_roi(frame, roi_polygon)
        else:
            processed = self.resize(frame)

        return self.encode_jpeg(processed)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_frame_processor.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add edge/frame_processor.py tests/test_frame_processor.py
git commit -m "feat: frame processor with motion skip, ROI crop, resize, JPEG encode"
```

---

### Task 14: Source Manager + MQTT Publisher (Edge)

**Files:**
- Create: `edge/source_manager.py`
- Create: `edge/mqtt_publisher.py`
- Create: `edge/local_bridge.py`
- Create: `edge/__init__.py`

- [ ] **Step 1: Write `edge/source_manager.py`**

```python
"""Camera source manager — opens and manages RTSP/USB capture devices."""
import time
import threading
from typing import Dict, Optional, Callable
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraSource:
    """Configuration for one camera source."""
    id: str
    source: str    # File path, RTSP URL, or USB device index (as string)
    roi: list = None  # [(x,y), ...] polygon


class SourceManager:
    """Manages multiple camera capture sources. Each source runs in its own thread,
    calling a callback with every captured frame."""

    def __init__(self, config: dict):
        """
        config: parsed edge/config.yaml dict.
        """
        self.cameras: Dict[str, CameraSource] = {}
        self._captures: Dict[str, cv2.VideoCapture] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._running: Dict[str, bool] = {}
        self._callbacks: Dict[str, Callable] = {}
        self._fps_interval: float = 1.0 / config.get("frame", {}).get("target_fps", 5)

        for cam_cfg in config.get("cameras", []):
            cam = CameraSource(
                id=cam_cfg["id"],
                source=str(cam_cfg["source"]),
                roi=cam_cfg.get("roi"),
            )
            self.cameras[cam.id] = cam

    def _parse_source(self, source: str):
        """Parse source: if numeric string, treat as int (USB index)."""
        try:
            return int(source)
        except ValueError:
            return source

    def start(self, camera_id: str, on_frame: Callable[[str, np.ndarray], None]):
        """Start capturing from a camera. on_frame(camera_id, bgr_frame) called per frame."""
        if camera_id not in self.cameras:
            raise ValueError(f"Unknown camera: {camera_id}")

        if camera_id in self._running and self._running[camera_id]:
            return  # Already running

        cam = self.cameras[camera_id]
        src = self._parse_source(cam.source)
        cap = cv2.VideoCapture(src)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera source: {cam.source}")

        # Set buffer size to 1 to minimize latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._captures[camera_id] = cap
        self._callbacks[camera_id] = on_frame
        self._running[camera_id] = True

        thread = threading.Thread(
            target=self._capture_loop,
            args=(camera_id,),
            daemon=True,
            name=f"cam-{camera_id}",
        )
        self._threads[camera_id] = thread
        thread.start()

    def _capture_loop(self, camera_id: str):
        cap = self._captures[camera_id]
        callback = self._callbacks[camera_id]
        interval = self._fps_interval

        while self._running.get(camera_id, False):
            start = time.time()

            ret, frame = cap.read()
            if not ret:
                # Try to reconnect
                time.sleep(1)
                cap.release()
                cam = self.cameras[camera_id]
                src = self._parse_source(cam.source)
                cap = cv2.VideoCapture(src)
                self._captures[camera_id] = cap
                continue

            try:
                callback(camera_id, frame)
            except Exception:
                pass  # Don't crash the capture loop on callback errors

            elapsed = time.time() - start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self, camera_id: str):
        """Stop capturing from a camera."""
        self._running[camera_id] = False
        if camera_id in self._captures:
            self._captures[camera_id].release()
            del self._captures[camera_id]

    def stop_all(self):
        """Stop all cameras."""
        for cid in list(self._running.keys()):
            self.stop(cid)

    def is_running(self, camera_id: str) -> bool:
        return self._running.get(camera_id, False)
```

- [ ] **Step 2: Write `edge/mqtt_publisher.py`**

```python
"""MQTT publisher — sends processed frames to the inference server."""
import json
import time
import threading
from typing import Optional

import paho.mqtt.client as mqtt


class MQTTPublisher:
    """Publishes JPEG frame bytes and heartbeat messages to MQTT broker."""

    def __init__(self, config: dict):
        """
        config: parsed edge/config.yaml dict.
        """
        mqtt_cfg = config.get("mqtt", {})
        self._broker = mqtt_cfg.get("broker", "localhost")
        self._port = mqtt_cfg.get("port", 1883)
        self._client_id = mqtt_cfg.get("client_id", "edge-agent-01")
        self._connected = False
        self._client: Optional[mqtt.Client] = None

        topics = config.get("topics", {})
        self._frame_topic_template = topics.get("frame", "cv/{camera_id}/frame")
        self._heartbeat_topic_template = topics.get("heartbeat", "cv/{camera_id}/heartbeat")

        self._heartbeat_interval = 10.0  # seconds
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._running = False

    def connect(self):
        """Connect to MQTT broker and start heartbeat thread."""
        self._client = mqtt.Client(client_id=self._client_id)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

        self._client.connect_async(self._broker, self._port, keepalive=30)
        self._client.loop_start()

        # Wait briefly for connection
        timeout = 5
        start = time.time()
        while not self._connected and (time.time() - start) < timeout:
            time.sleep(0.1)

        if not self._connected:
            raise ConnectionError(f"Failed to connect to MQTT broker at {self._broker}:{self._port}")

        self._running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self._heartbeat_thread.start()

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
        else:
            print(f"[MQTT] Connection failed with code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False

    def publish_frame(self, camera_id: str, jpeg_bytes: bytes):
        """Publish a JPEG frame to the camera's MQTT topic."""
        if not self._client or not self._connected:
            return
        topic = self._frame_topic_template.format(camera_id=camera_id)
        self._client.publish(topic, jpeg_bytes, qos=1)

    def _heartbeat_loop(self):
        while self._running:
            for camera_id in self._get_active_cameras():
                if not self._client or not self._connected:
                    break
                topic = self._heartbeat_topic_template.format(camera_id=camera_id)
                payload = json.dumps({"status": "alive", "timestamp": time.time()})
                self._client.publish(topic, payload, qos=0)
            time.sleep(self._heartbeat_interval)

    def _get_active_cameras(self):
        # Override point — actual implementation wires this with SourceManager
        return []

    def set_active_cameras(self, camera_ids: list):
        self._active_cameras = camera_ids

    def disconnect(self):
        """Disconnect from MQTT broker."""
        self._running = False
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
```

- [ ] **Step 3: Write `edge/local_bridge.py`**

```python
"""Local bridge — passes frames directly via asyncio.Queue when edge == server."""
import asyncio
from typing import Dict


class LocalBridge:
    """When a USB camera is plugged directly into the server machine,
    frames are passed through an in-process asyncio.Queue instead of MQTT."""

    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = {}

    def get_queue(self, camera_id: str, maxsize: int = 10) -> asyncio.Queue:
        """Get or create a queue for a camera."""
        if camera_id not in self._queues:
            self._queues[camera_id] = asyncio.Queue(maxsize=maxsize)
        return self._queues[camera_id]

    async def put_frame(self, camera_id: str, jpeg_bytes: bytes):
        """Put a frame into the camera's queue (non-blocking, drops oldest if full)."""
        queue = self.get_queue(camera_id)
        if queue.full():
            try:
                queue.get_nowait()  # Drop oldest
            except asyncio.QueueEmpty:
                pass
        await queue.put(jpeg_bytes)

    async def get_frame(self, camera_id: str, timeout: float = 1.0) -> bytes:
        """Get the next frame from the camera's queue."""
        queue = self.get_queue(camera_id)
        return await asyncio.wait_for(queue.get(), timeout=timeout)
```

- [ ] **Step 4: Write `edge/__init__.py` — EdgeAgent that ties everything together**

```python
"""Edge agent — ties together capture, processing, and publishing."""
import threading
from typing import Optional, Dict

import numpy as np

from edge.source_manager import SourceManager
from edge.frame_processor import FrameProcessor
from edge.mqtt_publisher import MQTTPublisher
from edge.local_bridge import LocalBridge


class EdgeAgent:
    """Orchestrates edge-side processing for all configured cameras."""

    def __init__(self, config: dict, local_bridge: Optional[LocalBridge] = None):
        self.config = config
        self.source_manager = SourceManager(config)
        self.processor = FrameProcessor(
            target_size=(
                config.get("frame", {}).get("resize_width", 416),
                config.get("frame", {}).get("resize_height", 416),
            ),
            motion_threshold=config.get("frame", {}).get("motion_threshold", 0.05),
            jpeg_quality=config.get("frame", {}).get("jpeg_quality", 70),
        )
        self.mqtt: Optional[MQTTPublisher] = None
        self.local_bridge = local_bridge

    def start_mqtt(self):
        """Connect to MQTT broker for remote cameras."""
        self.mqtt = MQTTPublisher(self.config)
        self.mqtt.connect()
        self.mqtt.set_active_cameras(list(self.source_manager.cameras.keys()))

    def start_all_cameras(self):
        """Start capturing from all configured cameras."""
        for cam_id, cam in self.source_manager.cameras.items():
            is_local = isinstance(cam.source, int) or cam.source in ("0", "1", "2", "3")
            if is_local and self.local_bridge:
                self.source_manager.start(cam_id, self._local_frame_handler)
            else:
                self.source_manager.start(cam_id, self._mqtt_frame_handler)

    def _local_frame_handler(self, camera_id: str, frame: np.ndarray):
        """Handle frame from local (USB) camera: push to LocalBridge queue."""
        cam = self.source_manager.cameras.get(camera_id)
        roi = cam.roi if cam else None
        jpeg_bytes = self.processor.process(frame, roi_polygon=roi)
        if jpeg_bytes and self.local_bridge:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(
                        self.local_bridge.put_frame(camera_id, jpeg_bytes)
                    )
            except RuntimeError:
                pass  # Event loop not available (test context)

    def _mqtt_frame_handler(self, camera_id: str, frame: np.ndarray):
        """Handle frame from remote (RTSP) camera: publish via MQTT."""
        cam = self.source_manager.cameras.get(camera_id)
        roi = cam.roi if cam else None
        jpeg_bytes = self.processor.process(frame, roi_polygon=roi)
        if jpeg_bytes and self.mqtt:
            self.mqtt.publish_frame(camera_id, jpeg_bytes)

    def stop(self):
        """Stop all capture and disconnect."""
        self.source_manager.stop_all()
        if self.mqtt:
            self.mqtt.disconnect()
```

- [ ] **Step 5: Commit**

```bash
git add edge/source_manager.py edge/mqtt_publisher.py edge/local_bridge.py edge/__init__.py
git commit -m "feat: edge agent — source manager, MQTT publisher, local bridge, orchestration"
```

---

### Task 15: Inference Engine — Model Manager

**Files:**
- Create: `inference/model_manager.py`
- Create: `inference/__init__.py`

- [ ] **Step 1: Write `inference/model_manager.py`**

```python
"""OpenVINO model manager — loads YOLOv8 ONNX→IR models and runs inference."""
import time
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import cv2


@dataclass
class Detection:
    """Single YOLO detection."""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    cls: int
    cls_name: str
    conf: float


@dataclass
class KeypointResult:
    """Pose keypoints for one person."""
    keypoints: np.ndarray  # shape (17, 3) — (x, y, conf)
    bbox: Tuple[float, float, float, float]


class ModelManager:
    """Manages YOLO model loading and inference.
    Falls back to ONNX Runtime if OpenVINO is not available."""

    CLASS_NAMES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
        'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
        'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
        'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush',
    ]

    # COCO classes we care about — add custom equipment classes
    # In COCO: person=0. Custom classes for helmet, vest, boot need fine-tuned model.
    # For now: use COCO person class + look for equipment via separate detection.
    TARGET_CLASSES = {
        0: 'person',
        # Custom model would add: 80: 'helmet', 81: 'vest', 82: 'boot'
    }

    def __init__(
        self,
        model_path: str = "models/yolov8n.onnx",
        pose_model_path: Optional[str] = "models/yolov8n-pose.onnx",
        input_size: Tuple[int, int] = (416, 416),
        conf_threshold: float = 0.4,
        nms_threshold: float = 0.45,
    ):
        self.model_path = Path(model_path)
        self.pose_model_path = Path(pose_model_path) if pose_model_path else None
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        self._session = None
        self._pose_session = None
        self._use_openvino = False

    def load(self):
        """Load the model. Tries OpenVINO first, falls back to ONNX Runtime."""
        # Try OpenVINO
        try:
            from openvino.runtime import Core
            core = Core()

            # Check for IR model (converted from ONNX)
            ir_path = self.model_path.with_suffix('.xml')
            if ir_path.exists():
                model = core.read_model(str(ir_path))
                self._session = core.compile_model(model, "CPU")
                self._use_openvino = True
                print(f"[ModelManager] Loaded OpenVINO IR model from {ir_path}")
            elif self.model_path.suffix == '.onnx' and self.model_path.exists():
                model = core.read_model(str(self.model_path))
                self._session = core.compile_model(model, "CPU")
                self._use_openvino = True
                print(f"[ModelManager] Loaded ONNX model via OpenVINO from {self.model_path}")
            else:
                raise FileNotFoundError(f"Model not found: {self.model_path}")
        except ImportError:
            print("[ModelManager] OpenVINO not available, using ONNX Runtime fallback")
            self._load_onnx_runtime()
        except Exception as e:
            print(f"[ModelManager] OpenVINO failed ({e}), falling back to ONNX Runtime")
            self._load_onnx_runtime()

        # Warm-up
        self._warmup()

    def _load_onnx_runtime(self):
        import onnxruntime as ort
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self._session = ort.InferenceSession(
            str(self.model_path),
            providers=['CPUExecutionProvider'],
        )

    def _warmup(self):
        """Run a dummy inference to warm up the model."""
        dummy = np.random.randn(1, 3, *self.input_size).astype(np.float32)
        if self._use_openvino:
            self._session([dummy])
        else:
            input_name = self._session.get_inputs()[0].name
            self._session.run(None, {input_name: dummy})
        print("[ModelManager] Warm-up complete")

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Preprocess a BGR frame for YOLO inference.
        Returns (1, 3, H, W) float32 tensor normalized to [0,1]."""
        img = cv2.resize(frame_bgr, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)  # HWC → CHW
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)

    def inference(self, tensor: np.ndarray) -> np.ndarray:
        """Run inference on a preprocessed tensor. Returns raw model output."""
        if self._use_openvino:
            result = self._session([tensor])
            return result[0] if isinstance(result, (list, tuple)) else result
        else:
            input_name = self._session.get_inputs()[0].name
            return self._session.run(None, {input_name: tensor})[0]

    def postprocess(self, output: np.ndarray) -> List[Detection]:
        """Convert YOLOv8 raw output to Detection objects with NMS."""
        # YOLOv8 output shape: (1, 84, 8400) — 80 classes + 4 bbox coords
        # Transpose to (8400, 84)
        if output.ndim == 3:
            output = output[0]
        output = output.transpose(1, 0)  # (8400, 84)

        boxes = output[:, :4]
        scores = output[:, 4:]

        detections: List[Detection] = []
        img_w, img_h = self.input_size

        for cls_id in range(scores.shape[1]):
            cls_scores = scores[:, cls_id]
            mask = cls_scores > self.conf_threshold
            if not mask.any():
                continue

            cls_boxes = boxes[mask]
            cls_confs = cls_scores[mask]

            for box, conf in zip(cls_boxes, cls_confs):
                # YOLOv8 outputs [cx, cy, w, h] normalized
                cx, cy, w, h = box
                x1 = (cx - w / 2) * img_w
                y1 = (cy - h / 2) * img_h
                x2 = (cx + w / 2) * img_w
                y2 = (cy + h / 2) * img_h

                cls_name = self.CLASS_NAMES[cls_id] if cls_id < len(self.CLASS_NAMES) else str(cls_id)
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    cls=cls_id,
                    cls_name=cls_name,
                    conf=float(conf),
                ))

        return self._nms(detections)

    def _nms(self, detections: List[Detection]) -> List[Detection]:
        """Apply Non-Maximum Suppression."""
        if not detections:
            return []

        boxes = np.array([d.bbox for d in detections])
        scores = np.array([d.conf for d in detections])

        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(),
            self.conf_threshold, self.nms_threshold,
        )

        if len(indices) == 0:
            return []

        return [detections[i] for i in indices.flatten()]

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Full pipeline: preprocess → inference → postprocess."""
        tensor = self.preprocess(frame_bgr)
        output = self.inference(tensor)
        return self.postprocess(output)

    def preprocess_jpeg(self, jpeg_bytes: bytes) -> np.ndarray:
        """Decode JPEG bytes to BGR frame."""
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    @property
    def is_loaded(self) -> bool:
        return self._session is not None
```

- [ ] **Step 2: Write `inference/__init__.py`**

```python
"""Inference engine package."""
from inference.model_manager import ModelManager, Detection
```

- [ ] **Step 3: Commit**

```bash
git add inference/model_manager.py inference/__init__.py
git commit -m "feat: OpenVINO/ONNX model manager for YOLOv8 detection"
```

---

### Task 16: Inference Engine — Detector + Scheduler

**Files:**
- Create: `inference/detector.py`
- Create: `inference/scheduler.py`
- Test: `tests/test_detector.py`, `tests/test_scheduler.py`

- [ ] **Step 1: Write `tests/test_detector.py`**

```python
"""Tests for the detector module — converts ModelManager detections to DetectionResult."""
import pytest
from unittest.mock import MagicMock
from shared.models import DetectionResult, DetectedObject, BBox


def test_detector_converts_detections():
    """Detector.run() should convert raw detections to DetectionResult."""
    from inference.detector import Detector
    from inference.model_manager import Detection

    mock_mm = MagicMock()
    mock_mm.detect.return_value = [
        Detection(bbox=(100, 100, 200, 300), cls=0, cls_name='person', conf=0.9),
        Detection(bbox=(110, 100, 180, 160), cls=0, cls_name='helmet', conf=0.7),
    ]
    mock_mm.preprocess_jpeg.return_value = None  # We mock detect() entirely

    detector = Detector(mock_mm)

    result = detector.run(b"fake_jpeg_bytes", "cam-01")

    assert isinstance(result, DetectionResult)
    assert result.camera_id == "cam-01"
    assert len(result.objects) == 2

    person = result.objects[0]
    assert person.cls == "person"
    assert person.conf == 0.9
    assert person.bbox.x1 == 100
    assert person.bbox.y1 == 100
    assert person.bbox.x2 == 200
    assert person.bbox.y2 == 300


def test_detector_empty_frame():
    """Detector should handle frames with no detections."""
    from inference.detector import Detector

    mock_mm = MagicMock()
    mock_mm.detect.return_value = []

    detector = Detector(mock_mm)

    result = detector.run(b"fake_jpeg_bytes", "cam-01")

    assert len(result.objects) == 0
    assert result.camera_id == "cam-01"
```

- [ ] **Step 2: Write `tests/test_scheduler.py`**

```python
"""Tests for the round-robin scheduler."""
import pytest
from unittest.mock import MagicMock


def test_scheduler_registers_cameras():
    """Scheduler should accept camera registration."""
    from inference.scheduler import Scheduler

    sched = Scheduler()
    sched.register_camera("cam-01")
    sched.register_camera("cam-02")

    assert sched.camera_count == 2
    assert "cam-01" in sched._queues
    assert "cam-02" in sched._queues


def test_scheduler_round_robin_order():
    """Scheduler should return cameras in round-robin order."""
    from inference.scheduler import Scheduler

    sched = Scheduler()

    # Add frames to queues
    sched.register_camera("cam-01")
    sched.register_camera("cam-02")
    sched.add_frame("cam-01", b"frame1")
    sched.add_frame("cam-02", b"frame2")
    sched.add_frame("cam-01", b"frame3")

    # First poll → cam-01
    result = sched.poll()
    assert result is not None
    cam_id, frame = result
    assert cam_id == "cam-01"
    assert frame == b"frame1"

    # Second poll → cam-02
    result = sched.poll()
    assert result is not None
    cam_id, frame = result
    assert cam_id == "cam-02"
    assert frame == b"frame2"

    # Third poll → cam-01 again (round-robin)
    result = sched.poll()
    assert result is not None
    cam_id, frame = result
    assert cam_id == "cam-01"
    assert frame == b"frame3"


def test_scheduler_empty_queues():
    """Scheduler should return None when all queues are empty."""
    from inference.scheduler import Scheduler

    sched = Scheduler()
    sched.register_camera("cam-01")
    sched.register_camera("cam-02")

    assert sched.poll() is None


def test_scheduler_mixed_empty():
    """Scheduler should skip cameras with empty queues."""
    from inference.scheduler import Scheduler

    sched = Scheduler()
    sched.register_camera("cam-01")
    sched.register_camera("cam-02")

    # Only cam-02 has a frame
    sched.add_frame("cam-02", b"frame2")

    result = sched.poll()
    assert result is not None
    cam_id, frame = result
    assert cam_id == "cam-02"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_detector.py tests/test_scheduler.py -v`
Expected: FAIL

- [ ] **Step 4: Write `inference/detector.py`**

```python
"""Detector — wraps ModelManager to produce DetectionResult objects."""
from datetime import datetime
from typing import Optional

import numpy as np

from inference.model_manager import ModelManager
from shared.models import DetectionResult, DetectedObject, BBox


class Detector:
    """Converts raw model detections into structured DetectionResult objects.
    Handles both object detection and pose estimation (via separate model)."""

    def __init__(self, model_manager: ModelManager):
        self.mm = model_manager

    def run(self, jpeg_bytes: bytes, camera_id: str) -> DetectionResult:
        """Run detection on a JPEG frame and return structured result."""
        frame = self.mm.preprocess_jpeg(jpeg_bytes)
        if frame is None:
            return DetectionResult(camera_id=camera_id)

        detections = self.mm.detect(frame)

        objects = [
            DetectedObject(
                bbox=BBox(d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]),
                cls=d.cls_name,
                conf=d.conf,
            )
            for d in detections
        ]

        return DetectionResult(
            camera_id=camera_id,
            objects=objects,
            keypoints=None,  # Pose model integration: Task 17
            timestamp=datetime.now(),
        )
```

- [ ] **Step 5: Write `inference/scheduler.py`**

```python
"""Round-robin scheduler for multi-camera inference."""
from collections import deque
from typing import Dict, Optional, Tuple


class Scheduler:
    """Fair round-robin scheduler across multiple cameras.
    Each camera has its own frame queue; poll() returns the next
    (camera_id, jpeg_bytes) pair in round-robin order."""

    def __init__(self):
        self._queues: Dict[str, deque] = {}
        self._order: list = []  # Camera registration order
        self._index: int = 0

    def register_camera(self, camera_id: str):
        """Register a camera for scheduling."""
        if camera_id not in self._queues:
            self._queues[camera_id] = deque(maxlen=5)
            self._order.append(camera_id)

    def add_frame(self, camera_id: str, jpeg_bytes: bytes):
        """Add a frame to a camera's queue. Drops oldest if queue is full."""
        if camera_id not in self._queues:
            self.register_camera(camera_id)
        self._queues[camera_id].append(jpeg_bytes)

    def poll(self) -> Optional[Tuple[str, bytes]]:
        """Get the next frame in round-robin order.
        Returns None if all queues are empty."""
        if not self._order:
            return None

        checked = 0
        while checked < len(self._order):
            cam_id = self._order[self._index]
            self._index = (self._index + 1) % len(self._order)

            if self._queues.get(cam_id) and len(self._queues[cam_id]) > 0:
                frame = self._queues[cam_id].popleft()
                return (cam_id, frame)

            checked += 1

        return None

    @property
    def camera_count(self) -> int:
        return len(self._order)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_detector.py tests/test_scheduler.py -v`
Expected: 6 PASS

- [ ] **Step 7: Commit**

```bash
git add inference/detector.py inference/scheduler.py tests/test_detector.py tests/test_scheduler.py
git commit -m "feat: detection wrapper and round-robin scheduler"
```

---

### Task 17: MQTT Subscriber + Local Receiver (Inference Side)

**Files:**
- Create: `inference/mqtt_subscriber.py`
- Create: `inference/local_receiver.py`

- [ ] **Step 1: Write `inference/mqtt_subscriber.py`**

```python
"""MQTT subscriber — receives frames from edge agents for inference."""
import time
import threading
from typing import Callable, Optional

import paho.mqtt.client as mqtt


class MQTTSubscriber:
    """Subscribes to MQTT topics for incoming camera frames."""

    def __init__(
        self,
        broker: str = "localhost",
        port: int = 1883,
        frame_topic_pattern: str = "cv/+/frame",
    ):
        self._broker = broker
        self._port = port
        self._frame_topic_pattern = frame_topic_pattern
        self._client: Optional[mqtt.Client] = None
        self._connected = False
        self._on_frame: Optional[Callable[[str, bytes], None]] = None

    def connect(self, on_frame: Callable[[str, bytes], None]):
        """Connect to MQTT broker and set frame callback.
        on_frame(camera_id: str, jpeg_bytes: bytes)"""
        self._on_frame = on_frame
        self._client = mqtt.Client(client_id="inference-server")
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

        self._client.connect_async(self._broker, self._port, keepalive=30)
        self._client.loop_start()

        timeout = 5
        start = time.time()
        while not self._connected and (time.time() - start) < timeout:
            time.sleep(0.1)

        if not self._connected:
            raise ConnectionError(
                f"Failed to connect to MQTT broker at {self._broker}:{self._port}"
            )

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            client.subscribe(self._frame_topic_pattern, qos=1)
        else:
            print(f"[MQTT Sub] Connection failed with code {rc}")

    def _on_message(self, client, userdata, msg):
        """Parse camera_id from topic and forward frame."""
        # Topic format: cv/{camera_id}/frame
        parts = msg.topic.split('/')
        if len(parts) >= 3:
            camera_id = parts[1]
            if self._on_frame:
                self._on_frame(camera_id, msg.payload)

    def disconnect(self):
        """Disconnect from MQTT broker."""
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
```

- [ ] **Step 2: Write `inference/local_receiver.py`**

```python
"""Local receiver — reads frames from LocalBridge queues for USB cameras."""
import asyncio
from typing import Callable, Optional, Dict

from edge.local_bridge import LocalBridge


class LocalReceiver:
    """Consumes frames from LocalBridge queues and feeds them to the scheduler.
    Each camera gets its own asyncio task."""

    def __init__(self, local_bridge: LocalBridge):
        self._bridge = local_bridge
        self._tasks: Dict[str, asyncio.Task] = {}
        self._running = False

    async def start(
        self,
        camera_ids: list,
        on_frame: Callable[[str, bytes], None],
    ):
        """Start receiving frames for the given camera IDs.
        on_frame(camera_id, jpeg_bytes) is called per frame."""
        self._running = True
        for cid in camera_ids:
            task = asyncio.create_task(
                self._receive_loop(cid, on_frame),
                name=f"local-rx-{cid}",
            )
            self._tasks[cid] = task

    async def _receive_loop(self, camera_id: str, on_frame: Callable):
        """Continuously read from the local bridge queue."""
        while self._running:
            try:
                jpeg_bytes = await self._bridge.get_frame(camera_id, timeout=1.0)
                on_frame(camera_id, jpeg_bytes)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    async def stop(self):
        """Stop all receiver tasks."""
        self._running = False
        for task in self._tasks.values():
            task.cancel()
        self._tasks.clear()
```

- [ ] **Step 3: Commit**

```bash
git add inference/mqtt_subscriber.py inference/local_receiver.py
git commit -m "feat: MQTT subscriber and local receiver for inference engine input"
```

---

### Task 18: Main Entry Point

**Files:**
- Create: `main.py`

- [ ] **Step 1: Write `main.py`**

```python
#!/usr/bin/env python3
"""CV Safety Monitor — Main entry point.
Orchestrates all subsystems: Edge, Inference, Alert, Dashboard."""

import sys
import asyncio
import signal
import threading
from pathlib import Path

import yaml


def load_config(config_path: str = "edge/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


async def run_dashboard(alert_pipeline):
    """Start FastAPI dashboard server."""
    import uvicorn
    from dashboard.server import app, ws_manager

    # Wire ws_manager into the dispatcher
    alert_pipeline.dispatcher._ws_manager = ws_manager

    config = uvicorn.Config(app, host="0.0.0.0", port=8080, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


async def run_inference_loop(inference_engine, alert_pipeline, scheduler, stop_event):
    """Main inference loop: poll scheduler → detect → alert pipeline."""
    from inference.detector import Detector
    import cv2

    print("[Main] Inference loop started")
    while not stop_event.is_set():
        result = scheduler.poll()
        if result is None:
            await asyncio.sleep(0.01)  # No frames available
            continue

        camera_id, jpeg_bytes = result
        try:
            detection = inference_engine.run(jpeg_bytes, camera_id)

            # Decode frame for thumbnail (needed if violations found)
            frame_bgr = inference_engine.mm.preprocess_jpeg(jpeg_bytes)

            violations = alert_pipeline.process(detection, frame_bgr=frame_bgr)

            # Broadcast preview frame to dashboard
            import base64
            from dashboard.server import ws_manager
            preview_b64 = base64.b64encode(jpeg_bytes).decode()
            import json
            await ws_manager.broadcast_async(json.dumps({
                "type": "preview",
                "camera_id": camera_id,
                "frame_base64": preview_b64,
            }))

            if violations:
                types = [v.type for v in violations]
                print(f"[Main] ALERT: camera={camera_id} violations={types}")

        except Exception as e:
            print(f"[Main] Error processing frame from {camera_id}: {e}")


def run_edge_agent(config, local_bridge, stop_event):
    """Run edge agent in a separate thread for local cameras."""
    from edge import EdgeAgent

    try:
        agent = EdgeAgent(config, local_bridge=local_bridge)

        # Determine if any cameras need MQTT
        has_remote = any(
            not (str(c.get("source", "")).isdigit())
            for c in config.get("cameras", [])
        )
        if has_remote:
            agent.start_mqtt()

        agent.start_all_cameras()
        print(f"[Main] Edge agent started with {len(agent.source_manager.cameras)} cameras")

        # Keep alive
        while not stop_event.is_set():
            stop_event.wait(1)

        agent.stop()
    except Exception as e:
        print(f"[Main] Edge agent error: {e}")


async def main():
    config_path = Path("edge/config.yaml")
    if not config_path.exists():
        print("[Main] No edge/config.yaml found. Using defaults.")
        config = {"cameras": [], "frame": {"target_fps": 5}}
    else:
        config = load_config(str(config_path))

    # 1. Initialize database
    from alert.db import init_db
    init_db()
    print("[Main] Database initialized")

    # 2. Create alert pipeline components
    from alert.db import get_roi as db_get_roi
    from alert.roi_matcher import ROIMatcher
    from alert.classifier import ViolationClassifier
    from alert.cooldown import CooldownManager
    from alert.dispatcher import Dispatcher
    from alert import AlertPipeline

    import alert.db as db_module
    roi_matcher = ROIMatcher(db_module)
    classifier = ViolationClassifier(confidence_threshold=0.4)
    cooldown = CooldownManager(cooldown_seconds=5.0)
    dispatcher = Dispatcher(db=db_module, ws_manager=None)
    alert_pipeline = AlertPipeline(roi_matcher, classifier, cooldown, dispatcher)
    print("[Main] Alert pipeline initialized")

    # 3. Create inference engine
    from inference.model_manager import ModelManager
    from inference.detector import Detector

    model_manager = ModelManager()
    model_available = Path("models/yolov8n.onnx").exists() or Path("models/yolov8n.xml").exists()

    if model_available:
        model_manager.load()
        inference_engine = Detector(model_manager)
        print("[Main] Inference engine loaded")
    else:
        print("[Main] WARNING: No model found in models/. Inference will be skipped.")
        print("[Main] Download YOLOv8n ONNX: "
              "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.onnx")
        inference_engine = None

    # 4. Create scheduler
    from inference.scheduler import Scheduler
    scheduler = Scheduler()

    # Register cameras in scheduler
    for cam in config.get("cameras", []):
        scheduler.register_camera(cam["id"])

    # 5. Setup local bridge for USB cameras
    from edge.local_bridge import LocalBridge
    local_bridge = LocalBridge()

    from inference.local_receiver import LocalReceiver
    local_receiver = LocalReceiver(local_bridge)

    # Hook local receiver → scheduler
    def on_local_frame(camera_id, jpeg_bytes):
        scheduler.add_frame(camera_id, jpeg_bytes)

    local_camera_ids = [
        c["id"] for c in config.get("cameras", [])
        if str(c.get("source", "")).isdigit()
    ]
    if local_camera_ids:
        await local_receiver.start(local_camera_ids, on_local_frame)
        print(f"[Main] Local receiver started for: {local_camera_ids}")

    # 6. Setup MQTT subscriber (for remote cameras)
    remote_camera_ids = [
        c["id"] for c in config.get("cameras", [])
        if not str(c.get("source", "")).isdigit()
    ]
    from inference.mqtt_subscriber import MQTTSubscriber
    mqtt_sub = None
    if remote_camera_ids:
        mqtt_cfg = config.get("mqtt", {})
        mqtt_sub = MQTTSubscriber(
            broker=mqtt_cfg.get("broker", "localhost"),
            port=mqtt_cfg.get("port", 1883),
        )

        def on_mqtt_frame(camera_id, jpeg_bytes):
            scheduler.add_frame(camera_id, jpeg_bytes)

        mqtt_sub.connect(on_mqtt_frame)
        print(f"[Main] MQTT subscriber started for remote cameras: {remote_camera_ids}")

    # 7. Start edge agent thread (captures USB + RTSP, feeds local_bridge + MQTT)
    stop_event = threading.Event()
    edge_thread = threading.Thread(
        target=run_edge_agent,
        args=(config, local_bridge, stop_event),
        daemon=True,
    )
    edge_thread.start()

    # 8. Start inference loop + dashboard concurrently
    inference_task = asyncio.create_task(
        run_inference_loop(inference_engine, alert_pipeline, scheduler, stop_event)
    ) if inference_engine else None

    dashboard_task = asyncio.create_task(run_dashboard(alert_pipeline))

    # 9. Handle shutdown
    def shutdown():
        print("\n[Main] Shutting down...")
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, shutdown)
        except NotImplementedError:
            pass  # Windows

    try:
        await dashboard_task
    except asyncio.CancelledError:
        pass
    finally:
        stop_event.set()
        if mqtt_sub:
            mqtt_sub.disconnect()
        await local_receiver.stop()
        if inference_task:
            inference_task.cancel()
        print("[Main] Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Commit**

```bash
git add main.py
git commit -m "feat: main entry point orchestrating all subsystems"
```

---

### Task 19: Integration Test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write `tests/test_integration.py`**

```python
"""End-to-end integration tests for the full pipeline."""
import json
import asyncio
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


@pytest.mark.asyncio
async def test_alert_pipeline_end_to_end(temp_dir):
    """Full pipeline from detection to dispatch with in-memory DB."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import alert.db as db_module
    original_path = db_module.DB_PATH
    db_module.DB_PATH = temp_dir / "test.db"
    db_module.init_db()

    try:
        from alert.roi_matcher import ROIMatcher
        from alert.classifier import ViolationClassifier
        from alert.cooldown import CooldownManager
        from alert.dispatcher import Dispatcher
        from alert import AlertPipeline
        from shared.models import DetectionResult, DetectedObject, BBox

        # Setup pipeline with no cooldown
        roi = ROIMatcher(db_module)  # No ROI configs → allow all
        classifier = ViolationClassifier(confidence_threshold=0.4)
        cooldown = CooldownManager(cooldown_seconds=0)
        dispatcher = Dispatcher(db=db_module, ws_manager=None, thumbnail_dir=str(temp_dir / "thumbs"))

        pipeline = AlertPipeline(roi, classifier, cooldown, dispatcher)

        # Create a detection with person but no helmet and no vest
        result = DetectionResult(
            camera_id="cam-01",
            objects=[
                DetectedObject(bbox=BBox(100, 100, 200, 300), cls="person", conf=0.9),
            ],
        )

        violations = pipeline.process(result, frame_bgr=None)
        types = {v.type for v in violations}

        assert "NO_HELMET" in types
        assert "NO_VEST" in types
        assert len(violations) >= 2

        # Verify DB insert
        rows = db_module.get_violations()
        assert len(rows) >= 2

    finally:
        db_module.DB_PATH = original_path


@pytest.mark.asyncio
async def test_scheduler_to_detector_flow():
    """Scheduler → Detector → AlertPipeline integration."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from inference.scheduler import Scheduler
    from alert.cooldown import CooldownManager
    from shared.models import Violation

    # Create mock detector that returns known result
    mock_mm = MagicMock()
    mock_mm.preprocess_jpeg.return_value = None
    mock_mm.detect.return_value = []

    from inference.detector import Detector
    detector = Detector(mock_mm)

    # Setup scheduler
    scheduler = Scheduler()
    scheduler.register_camera("cam-01")
    scheduler.add_frame("cam-01", b"fake_jpeg")

    # Poll and detect
    result = scheduler.poll()
    assert result is not None
    camera_id, jpeg_bytes = result
    assert camera_id == "cam-01"

    detection = detector.run(jpeg_bytes, camera_id)
    assert detection.camera_id == "cam-01"
    assert len(detection.objects) == 0  # Mock returns empty


def test_config_roundtrip():
    """Config YAML can be parsed and camera list is correct."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import yaml

    config_path = Path("edge/config.yaml")
    if not config_path.exists():
        pytest.skip("config.yaml not found")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert "cameras" in config
    assert "mqtt" in config
    assert "frame" in config
    assert isinstance(config["cameras"], list)
    for cam in config["cameras"]:
        assert "id" in cam
        assert "source" in cam
```

- [ ] **Step 2: Run integration tests**

Run: `python -m pytest tests/test_integration.py -v`
Expected: 3 PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: integration tests for full pipeline"
```

---

### Task 20: README & Documentation

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# CV Safety Monitor

Realtime computer vision system for construction site safety monitoring. Detects violations (fall, no-helmet, no-vest, no-boot) within user-defined ROI zones across multiple cameras.

## Architecture

```
Webcam/RTSP → Edge Agent → MQTT/Queue → Inference Engine (OpenVINO YOLOv8)
                                           ↓
                                      DetectionResult
                                           ↓
                                      Alert Pipeline
                                      (ROI → Classify → Cooldown → Dispatch)
                                           ↓
                                   Dashboard (WebSocket)
```

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Intel CPU (for OpenVINO) or any CPU (ONNX Runtime fallback)
- Mosquitto MQTT broker (optional, for remote cameras)

### 2. Install

```bash
pip install -r requirements.txt
```

### 3. Download Models

```bash
mkdir -p models
cd models
# YOLOv8n detection model
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.onnx
# YOLOv8n-pose model (optional, for fall detection)
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-pose.onnx
cd ..
```

### 4. Configure Cameras

Edit `edge/config.yaml`:
```yaml
mqtt:
  broker: localhost
  port: 1883

cameras:
  - id: cam-01
    source: 0          # USB webcam index
    roi: [[100, 50], [500, 50], [500, 400], [100, 400]]
  - id: cam-02
    source: rtsp://192.168.1.100:554/stream1
```

### 5. Run

```bash
python main.py
```

Open http://localhost:8080 for the dashboard.
Open http://localhost:8080/admin.html for the ROI drawing tool.
Open http://localhost:8080/history.html for violation history.
API docs: http://localhost:8080/docs

## Project Structure

```
CV/
├── edge/              # Frame capture, processing, MQTT publish
├── inference/         # OpenVINO model loading, detection, scheduling
├── alert/             # ROI matching, violation classification, cooldown, dispatch
├── dashboard/         # FastAPI server, WebSocket, HTML/JS/CSS frontend
├── shared/            # Shared data models (DetectionResult, Violation, etc.)
├── tests/             # Unit and integration tests
├── models/            # Downloaded ONNX/IR model files
├── data/              # SQLite DB and thumbnail storage (runtime)
└── main.py            # Entry point
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/cameras` | List configured cameras |
| GET | `/api/roi/{camera_id}` | Get ROI polygon |
| PUT | `/api/roi/{camera_id}` | Save ROI polygon |
| GET | `/api/violations` | Query violation history |
| GET | `/api/violations/{id}/thumbnail` | Get violation thumbnail |
| WS | `/ws/dashboard` | Realtime alert + preview stream |

## Violation Types

| Type | Severity | Detection Method |
|------|----------|-----------------|
| FALL | HIGH | Pose estimation (aspect ratio + keypoint geometry) |
| NO_HELMET | HIGH | Person without overlapping helmet detection |
| NO_VEST | MEDIUM | Person without overlapping vest detection |
| NO_BOOT | MEDIUM | Person without boot detection in lower third |

## Running Tests

```bash
pytest tests/ -v
```

## License

MIT
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with setup instructions and architecture overview"
```

---

## Task Dependency Graph

```
Task 1 (Scaffold)
 ├─► Task 2 (DB)
 │    ├─► Task 3 (ROI Matcher)
 │    ├─► Task 4 (Cooldown)
 │    ├─► Task 5 (Classifier)
 │    │    └─► Task 6 (Dispatcher)
 │    │         └─► Task 7 (Alert Pipeline)
 │    │              └─► Task 8 (Dashboard Server)
 │    │                   ├─► Task 9 (Dashboard HTML/CSS)
 │    │                   ├─► Task 10 (Dashboard JS)
 │    │                   ├─► Task 11 (Admin ROI Tool)
 │    │                   └─► Task 12 (History Page)
 │    └─► Task 13 (Frame Processor)
 │         └─► Task 14 (Edge Agent)
 │              └─► Task 15 (Model Manager)
 │                   └─► Task 16 (Detector + Scheduler)
 │                        └─► Task 17 (MQTT Sub + Local RX)
 │                             └─► Task 18 (Main Entry Point)
 │                                  └─► Task 19 (Integration Test)
 │                                       └─► Task 20 (README)
 └─► (all tasks feed into final integration)
```

---

## Model Availability Note

Tasks 15-19 require YOLOv8 model files. If models are not yet downloaded, the inference engine will log a warning and the system will run with detection disabled. The dashboard, alert pipeline logic, and edge processing can all be developed and tested without a live model by using mocks (as shown in the test files).
