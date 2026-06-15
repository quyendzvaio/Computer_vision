# CV Safety Monitor v2 — Design Specification

**Date:** 2026-06-16  
**Status:** Draft  
**Author:** Brainstorming process  

## 1. Tổng quan

### 1.1 Mục tiêu

Thiết kế lại hệ thống CV Safety Monitor từ kiến trúc multi-process + Web Dashboard sang kiến trúc PyQt5 + QThread + ZeroMQ, tối ưu latency làm ưu tiên hàng đầu. Accuracy mục tiêu ~65% overall, riêng boots ~60%.

### 1.2 Bài toán

Hệ thống gồm 2 camera USB gắn trên edge device, mỗi camera giải quyết một bài toán riêng biệt:

| Camera | Bài toán | Phương pháp |
|--------|----------|-------------|
| **CAM1** | Phát hiện người xuất hiện trong vùng ROI | YOLOv8n detect person → foot point → point_in_polygon |
| **CAM2** | Phát hiện thiếu trang bị bảo hộ (PPE) | YOLOv8n detect person → crop vùng cơ thể → MobileNetV3 classifier |

### 1.3 Công nghệ

| Thành phần | Công nghệ | Lý do |
|------------|-----------|-------|
| Edge capture | Python + OpenCV | Đơn giản, không cần GUI trên edge |
| Transport | ZeroMQ PUB/SUB | Nhẹ, auto-reconnect, không framing overhead |
| Detection | YOLOv8n ONNX (GPU) | Cân bằng speed/accuracy |
| PPE Classification | MobileNetV3-small ONNX | Nhẹ (~2-5ms), đủ accuracy |
| Desktop UI | PyQt5 + QPainter | Render native, ROI drawing mượt |
| Web secondary | FastAPI + WebSocket | History, remote viewing nhẹ |
| Database | SQLite | Đơn giản, đủ cho single machine |
| Edge process manager | systemd | Auto-restart, logging |

## 2. Kiến trúc tổng quan

### 2.1 Sơ đồ kiến trúc

```
┌───────────────────────────┐         ┌──────────────────────────────────────────┐
│   EDGE DEVICE             │         │   GPU MACHINE                            │
│                           │         │                                          │
│  USB Cam1 ──┐             │         │   PyQt5 Main Process                      │
│  USB Cam2 ──┤ capture     │         │   ├─ QThread CAM1:                       │
│              │ (OpenCV)   │         │   │   ZMQ SUB → YOLOv8n → ROI → overlay  │
│              │            │ ZMQ TCP │   ├─ QThread CAM2:                       │
│              │ raw BGR    │◄────────│   │   ZMQ SUB → YOLOv8n → crop → classify│
│              │ port 5555  │         │   ├─ Main Thread:                        │
│              │ port 5556  │         │   │   PyQt QWidget render + ROI draw     │
│              └────────────┘         │   │   + Alert popup + Status bar         │
│                                     │   └─ QThread FastAPI:                    │
│                                     │       REST API + WS push (frame 1fps)    │
└───────────────────────────┘         └──────────────────────────────────────────┘
```

### 2.2 Luồng dữ liệu tổng quát

Edge device capture USB camera → gửi raw BGR qua ZMQ PUB → GPU machine ZMQ SUB nhận frame → YOLOv8n infer → hậu xử lý (ROI/PPE) → overlay → PyQt render.

### 2.3 Nguyên tắc thiết kế

- **Zero IPC:** Mọi xử lý trên GPU machine trong cùng 1 process, không shared memory, không multiprocessing
- **Latest frame:** Không queue dồn frame, chỉ giữ frame mới nhất. ZMQ HWM=2.
- **Frame budget:** Mỗi QThread tự điều tiết, skip frame nếu quá budget.
- **Graceful degradation:** Inference lỗi → vẫn hiển thị frame raw. Edge mất kết nối → "Disconnected" overlay.

## 3. Edge Device

### 3.1 Chức năng

Script Python đơn giản, không GUI, không PyQt. Capture 2 USB camera → gửi raw BGR frame qua ZMQ PUB.

### 3.2 Cấu trúc file

```
/home/quyen/CV/edge/
├── sender.py           # Capture + send loop
├── config.yaml         # Camera config, ZMQ ports
└── requirements.txt    # opencv-python, pyzmq, pyyaml
```

### 3.3 Xử lý chi tiết

```python
# Pseudo-code — edge/sender.py

main():
    cfg = load_config("config.yaml")
    ctx = zmq.Context()
    pubs = {}  # camera_id -> ZMQ PUB socket

    for cam in cfg["cameras"]:
        cap = cv2.VideoCapture(cam["device_path"])  # /dev/v4l/by-id/...
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        pub = ctx.socket(zmq.PUB)
        pub.setsockopt(zmq.SNDHWM, 2)      # Chỉ queue 2 frame
        pub.connect(f"tcp://{cfg['gpu_host']}:{cam['zmq_port']}")
        pubs[cam["id"]] = (cap, pub)

    while True:
        for cam_id, (cap, pub) in pubs.items():
            ret, frame = cap.read()
            if ret:
                pub.send(frame.tobytes())   # raw BGR, không encode
            else:
                log_error(f"Camera {cam_id} disconnected")
                # Retry logic: reconnect sau 1s
```

### 3.4 Camera config

```yaml
# edge/config.yaml
gpu_host: 192.168.1.100

cameras:
  - id: cam1
    device_path: /dev/v4l/by-id/usb-046d_0825_ABC123-video-index0
    zmq_port: 5555
    fps: 30
    resolution: [640, 480]

  - id: cam2
    device_path: /dev/v4l/by-id/usb-046d_0825_DEF456-video-index0
    zmq_port: 5556
    fps: 30
    resolution: [640, 480]
```

### 3.5 Error handling

| Scenario | Behavior |
|----------|----------|
| Camera disconnect | Retry mỗi 1s, tối đa 5 lần → log error → chờ reconnect |
| GPU machine offline | ZMQ PUB queue tối đa 2 frame (HWM=2), frame cũ bị drop |
| Edge crash | systemd Restart=always, restart <1s |

## 4. GPU Machine — CAM1 (ROI Inside Detection)

### 4.1 Pipeline

```
ZMQ SUB cam1 port 5555
  → raw BGR bytes → numpy array
  → YOLOv8n detect person (640×480)
  → Với mỗi person:
      foot_point = (x_center, y2)       # đáy bbox
      rows = point_in_poly(foot_point, roi_polygons)
      Nếu inside bất kỳ ROI:
        alert = PERSON_IN_ZONE
  → Overlay: vẽ person bbox + ROI polygons + highlight zone
  → emit_signal(overlay_frame, alerts)
```

### 4.2 ROI Checker

- Input: foot point (x_center, y2) của bbox person
- Algorithm: point_in_polygon (ray casting)
- Lưu ý: dùng foot point thay vì center point → tránh trường hợp thân trên trong zone nhưng chân ngoài
- Configuration: ROI polygons load từ SQLite (3-4 zones per camera)
- Mỗi zone có label, màu sắc, enabled/disabled

### 4.3 Alert

- Alert type: `PERSON_IN_ZONE`
- Severity: `HIGH`
- Cooldown: 30s mỗi zone, không spam nếu person đứng trong zone liên tục
- Person tracking: không cần ID tracking. Chỉ cần person inside zone tại frame hiện tại là đủ.

## 5. GPU Machine — CAM2 (PPE Detection)

### 5.1 Pipeline

```
ZMQ SUB cam2 port 5556
  → raw BGR bytes → numpy array
  → YOLOv8n detect person (640×480)
  → Với mỗi person bbox (x1,y1,x2,y2):
      h = y2 - y1

      # Head crop
      head = frame[y1:y1+int(0.2*h), x1:x2]
      head_resized = resize(head, 224, 224)
      helmet = helmet_classifier(head_resized)   # MobileNetV3-small

      # Torso crop
      torso = frame[y1+int(0.2*h):y2-int(0.3*h), x1:x2]
      torso_resized = resize(torso, 224, 224)
      vest = vest_classifier(torso_resized)       # MobileNetV3-small

      # Feet crop
      feet = frame[y2-int(0.15*h):y2, x1:x2]
      feet_resized = resize(feet, 224, 224)
      boot = boot_classifier(feet_resized)        # MobileNetV3-small

      alerts = []
      if helmet == "NO_HELMET": alerts.append("NO_HELMET")
      if vest == "NO_VEST":   alerts.append("NO_VEST")
      if boot == "NO_BOOT":   alerts.append("NO_BOOT")

  → Overlay: person bbox + PPE status label per person
  → emit_signal(overlay_frame, alerts)
```

### 5.2 Skip frame strategy cho CAM2

Do PPE pipeline (person detect + 3×crop + 3×classify) nặng hơn CAM1, áp dụng skip theo 2 cấp:

**Cấp 1 — Frame-level skip (trước YOLOv8):**
- Motion detection bằng frame difference > threshold → nếu không có motion, skip toàn bộ pipeline cho frame đó
- Frame budget: nếu frame trước xử lý > 50ms → skip frame kế tiếp

**Cấp 2 — Classify-level skip (sau YOLOv8, trước classifier):**
- Classify skip: chỉ chạy 3 classifier mỗi 2-3 frame. Frame chỉ có YOLOv8 detect + overlay (không PPE check) vẫn hữu ích cho preview.
- Person count: nếu > 3 persons trong frame → giảm classify frequency xuống 1/4 frames

Ví dụ: CAM2 nhận 30fps, có motion 20fps → YOLOv8 chạy 20fps. Trong 20fps đó, classifier chạy 7fps. Effective ~7fps cho PPE check, nhưng preview luôn 20fps.

### 5.3 MobileNetV3 Classifier

| Model | Input size | Latency (GPU) | Target accuracy | Training data |
|-------|-----------|---------------|-----------------|---------------|
| helmet_classifier | 224×224 | ~3ms | 60-70% | Head crops + helmet/no_helmet |
| vest_classifier | 224×224 | ~3ms | 65-75% | Torso crops + vest/no_vest |
| boot_classifier | 224×224 | ~3ms | 55-65% | Feet crops + boot/no_boot |

Classifier training không nằm trong scope của spec này. Cần dataset riêng.

## 6. GPU Machine — PyQt Main Window

### 6.1 Layout

```
┌──────────────────────────────────────────────────────────────────────┐
│ [CV Safety Monitor v2]                                   — □ ✗     │
├──────────────────────────────────────────────────────────────────────┤
│ [Cam1 ▼] [Cam2]  [✎ ROI draw]  [⏺ Record]  [⚙ Settings]          │
├────────────────────────────┬─────────────────────────────────────────┤
│                            │                                         │
│   ┌──────────────────┐   │   ┌──────────────────┐                  │
│   │  CAM1 Preview    │   │   │  CAM2 Preview    │                  │
│   │  ROI polygons    │   │   │  Bboxes + PPE    │                  │
│   │  Person bboxes   │   │   │  labels          │                  │
│   │  Zone A: 1 alert │   │   │  Status per zone │                  │
│   └──────────────────┘   │   └──────────────────┘                  │
│                            │                                         │
├────────────────────────────┴─────────────────────────────────────────┤
│ [Alert] 13:45:02  PERSON_IN_ZONE - Cam1 - Zone A                    │
│ [Alert] 13:45:05  NO_VEST - Cam2 - Person #2                       │
│ [Alert] 13:45:12  NO_BOOT - Cam2 - Person #1                       │
└──────────────────────────────────────────────────────────────────────┘
```

### 6.2 Camera widget

- QWidget với QPainter vẽ frame + overlay
- Sử dụng QImage từ numpy array (zero-copy)
- Mouse events cho ROI drawing (click → add point, double-click → close polygon)

### 6.3 ROI drawing mode

| Action | Behavior |
|--------|----------|
| Click on canvas | Add vertex point |
| Double-click | Close polygon |
| Drag vertex | Move vertex |
| Drag edge | Move entire polygon |
| Right-click vertex | Delete vertex |
| Right-click polygon | Delete polygon |
| ESC | Exit draw mode |
| Tab cycle zones | Chọn zone đang vẽ (A/B/C/D) |

### 6.4 Alert notification

- Visual: Flash đỏ border trên camera widget + Alert row in status bar
- System tray: QSystemTrayIcon.showMessage() for HIGH severity
- Sound: optional
- Thumbnail: snapshot frame + violation bbox lưu vào disk

## 7. GPU Machine — FastAPI Web (Secondary)

### 7.1 Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/cameras` | List camera configs |
| GET | `/api/roi/{camera_id}` | Get ROI polygons |
| PUT | `/api/roi/{camera_id}` | Save ROI polygons |
| GET | `/api/violations` | Query history (page, filter) |
| GET | `/api/violations/{id}/thumbnail` | Get violation image |
| WS | `/ws/dashboard` | Alert text + frame 1fps |

### 7.2 WebSocket protocol

Frame push 1fps (không 30fps) để tiết kiệm CPU:

```json
{
  "type": "preview",
  "camera_id": "cam1",
  "frame_base64": "...",    // JPEG quality 50, 1fps
  "alerts_count": 3
}
```

Alert push real-time:

```json
{
  "type": "alert",
  "violation": {
    "id": "uuid",
    "camera_id": "cam1",
    "type": "PERSON_IN_ZONE",
    "zone": "Zone A",
    "timestamp": "2026-06-16T13:45:02"
  }
}
```

### 7.3 Lifecycle

FastAPI chạy trong QThread riêng với `uvicorn.Server`, có shutdown event để cleanup khi app đóng.

## 8. SQLite Schema

### Tables

```sql
CREATE TABLE cameras (
    id TEXT PRIMARY KEY,
    zmq_port INTEGER NOT NULL,
    device_path TEXT,
    enabled BOOLEAN DEFAULT 1
);

CREATE TABLE roi_config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL REFERENCES cameras(id),
    zone_name TEXT NOT NULL,
    points_json TEXT NOT NULL,       -- [[x1,y1],[x2,y2],...]
    color TEXT NOT NULL DEFAULT '#ff0000',
    enabled BOOLEAN DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE violations (
    id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL,
    type TEXT NOT NULL,             -- PERSON_IN_ZONE, NO_HELMET, NO_VEST, NO_BOOT
    severity TEXT NOT NULL,         -- HIGH, MEDIUM, LOW
    zone_name TEXT,
    person_idx INTEGER,
    bbox_json TEXT,
    thumbnail_path TEXT,
    acknowledged BOOLEAN DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

## 9. Error Handling

### 9.1 Edge device

| Lỗi | Xử lý |
|-----|-------|
| Camera disconnect | Retry 5 lần × 1s → log → chờ |
| ZMQ send fail | Auto queue → drop khi HWM full |
| Edge crash | systemd auto-restart |
| Frame read fail | Skip frame, không retry |

### 9.2 GPU machine

| Lỗi | Xử lý |
|-----|-------|
| ZMQ SUB timeout >3s | "Disconnected" overlay |
| Inference crash (OOM) | Catch → fallback CPU → log |
| Model load fail | "Detection offline" badge |
| QThread unhandled exception | Catch → log → restart ≤5 lần/phút |
| FastAPI port conflict | Increment port, log warning |

## 10. Testing Strategy

### 10.1 Unit tests (pytest)

| Module | Test cases |
|--------|------------|
| ROIChecker | point_in_poly với polygon lồi, lõm, cạnh, điểm trong/ngoài |
| PPEChecker | Crop theo bbox ratio, verify crop region |
| FrameBudget | Skip logic, queued latest frame |
| AlertManager | Cooldown dedup, duplicate trong N giây |
| ZMQProtocol | Send/recv raw bytes, HWM behavior |

### 10.2 Integration tests (cần GPU)

| Test | Mô tả |
|------|-------|
| YOLOv8n inference | Frame → objects → format check |
| Full CAM1 pipeline | ZMQ → detect → ROI → alert |
| Full CAM2 pipeline | ZMQ → detect → crop → classify → alert |
| PyQt render | Signal receive → QImage → show |

### 10.3 Edge simulator

Script phát video file (.mp4) qua ZMQ PUB giả lập edge device. Cho phép test GPU machine trên cùng máy, không cần edge HW.

```
tests/
├── edge_simulator.py    # Play .mp4 via ZMQ PUB
├── test_roi_checker.py
├── test_ppe_checker.py
├── test_alert_manager.py
├── test_frame_budget.py
├── test_integration_cam1.py
├── test_integration_cam2.py
└── conftest.py           # Fixtures: model, mock ZMQ, test data
```

## 11. Latency Budget

### 11.1 CAM1 (ROI Inside)

| Stage | Time |
|-------|------|
| ZMQ network | ~1ms |
| YOLOv8n 640×480 GPU | ~12ms |
| ROI check | ~1ms |
| Overlay draw | ~1ms |
| **Total** | **~15ms** |

### 11.2 CAM2 (PPE)

| Stage | Time | Note |
|-------|------|------|
| ZMQ network | ~1ms | |
| YOLOv8n 640×480 GPU | ~12ms | |
| 3× crop + resize | ~3ms | |
| 3× MobileNetV3 GPU | ~9ms | Skip 2/3 frames |
| Overlay draw | ~1ms | |
| **Total (no skip)** | **~26ms** | ~38fps |
| **Total (with skip)** | **~15ms** | classifier ~10fps |

### 11.3 End-to-end latency

- CAM1: 15ms (67fps throughput)
- CAM2: 15-26ms (38-67fps throughput)
- PyQt render: <1ms (QImage from numpy, zero-copy)

## 12. Trade-offs & Known Limitations

### 12.1 Accuracy
- **Helmet/vet:** 60-75%. Ok với target 65%.
- **Boots:** 55-65%. Giới hạn của crop từ bbox person (không có keypoint).
- **Person in ROI:** ~70% (chỉ phụ thuộc YOLOv8n detect person + point_in_poly).

### 12.2 Latency vs accuracy
- Boots 60% là mức cao nhất có thể với crop + classifier approach.
- Nếu muốn boots >70%, cần YOLO-Pose keypoint → crop chính xác → +10ms latency.

### 12.3 Accessibility
- PyQt desktop: cần màn hình/mouse trên GPU machine.
- FastAPI web: lịch sử, alert, preview 1fps — đủ cho supervisor xem từ xa.
- MediaMTX: đã loại bỏ khỏi spec. Nếu cần RTSP sau này, có thể thêm.

### 12.4 Single point of failure
- GPU machine die → toàn bộ system die.
- Edge device die → mất cả 2 camera.
- Cần backup plan (không trong scope spec này).

## 13. Project Structure

```
/home/quyen/CV/
├── edge/
│   ├── sender.py            # Capture + ZMQ PUB
│   ├── config.yaml          # Camera, GPU host config
│   └── requirements.txt
│
├── gpu/
│   ├── main.py              # Entry point: start PyQt app
│   ├── main_window.py       # PyQt MainWindow + toolbar
│   ├── camera_widget.py     # QWidget render frame + overlay
│   ├── roi_drawer.py        # Mouse event → ROI polygon
│   ├── cam1_thread.py       # QThread: ZMQ SUB + YOLOv8 + ROI
│   ├── cam2_thread.py       # QThread: ZMQ SUB + YOLOv8 + PPE
│   ├── detector.py          # YOLOv8n ONNX wrapper
│   ├── classifier.py        # MobileNetV3 ONNX wrapper
│   ├── roi_checker.py       # point_in_polygon
│   ├── ppe_checker.py       # crop → classify pipeline
│   ├── alert_manager.py     # Cooldown, SQLite, WS broadcast
│   ├── overlay.py           # Draw bbox + ROI + labels
│   ├── web_server.py        # FastAPI QThread
│   ├── database.py          # SQLite helper
│   └── models/
│       ├── yolov8n.onnx
│       ├── helmet.onnx      # MobileNetV3-small
│       ├── vest.onnx        # MobileNetV3-small
│       └── boot.onnx        # MobileNetV3-small
│
├── tests/
│   ├── edge_simulator.py
│   ├── test_roi_checker.py
│   ├── test_ppe_checker.py
│   ├── test_alert_manager.py
│   ├── test_frame_budget.py
│   ├── test_integration_cam1.py
│   ├── test_integration_cam2.py
│   └── conftest.py
│
├── data/
│   ├── cv.db               # SQLite runtime
│   └── thumbnails/
│
├── docs/superpowers/specs/
│   └── 2026-06-16-cv-safety-monitor-v2-design.md
│
└── requirements-gpu.txt
```

## 14. Implementation Order

1. **Edge sender** — capture + ZMQ PUB (đơn giản, test độc lập)
2. **GPU detector** — YOLOv8n wrapper (test với edge simulator)
3. **CAM1 pipeline** — ROI checker + overlay
4. **CAM2 pipeline** — Crop + classifier + overlay
5. **PyQt Main Window** — 2 camera widget, toolbar
6. **ROI drawer** — Mouse interaction
7. **Alert manager** — SQLite + cooldown
8. **FastAPI web** — REST + WS
9. **Integration test** — Full system
10. **Deploy** — systemd service, config
