# Spec: Hệ Thống Computer Vision Phát Hiện An Toàn Công Trường

> **Ngày:** 2026-06-12
> **Trạng thái:** Draft — chờ review

---

## 1. Tổng Quan

Hệ thống computer vision realtime phát hiện vi phạm an toàn trong vùng ROI được khoanh sẵn trên công trường xây dựng, qua nhiều camera (webcam USB + IP). Khi phát hiện vi phạm trong ROI, cảnh báo lớn hiển thị tức thì trên dashboard giám sát và lưu lại để xem sau.

### Các tình huống cần phát hiện

| ID | Tình huống | Severity |
|----|-----------|----------|
| FALL | Người ngã | HIGH |
| NO_HELMET | Không đội mũ bảo hộ | HIGH |
| NO_VEST | Không mặc áo bảo hộ | MEDIUM |
| NO_BOOT | Không đeo giày bảo hộ | MEDIUM |

### Ràng buộc

- Chỉ cảnh báo khi vi phạm nằm trong vùng ROI (polygon) được khoanh sẵn
- Recall > 70% (ưu tiên không bỏ sót vi phạm)
- Độ trễ mục tiêu: 500ms–1s từ frame capture đến alert
- Realtime multi-camera

---

## 2. Kiến Trúc Tổng Thể

Phương án: **Hybrid cân bằng** — Edge mỏng + Server tập trung inference.

```
┌─────────────────────────────────────────────────────────────┐
│                        CÔNG TRƯỜNG                           │
│                                                              │
│  [Webcam₁]──┐                                                │
│  [Webcam₂]──┤   RTSP                                         │
│  [Webcam₃]──┼────────► [Edge Device]                         │
│  [Webcam₄]──┤          (Intel NUC / Mini PC)                │
│             ...          - Capture RTSP streams               │
│                          - Skip frame thông minh              │
│                          - Crop ROI từng camera               │
│                          - Resize xuống (416x416)             │
│                          - Gửi frame đã xử lý lên Server      │
│                                                              │
└──────────────────────────┬──────────────────────────────────┘
                           │ MQTT / WebSocket
                           │ (frame đã crop + metadata)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     MAIN MACHINE (Edge + Server)             │
│                                                              │
│  ┌────────────────┐  ┌─────────────────────────────────┐   │
│  │ USB Webcam     │  │ Inference Engine (OpenVINO)      │   │
│  │ (direct capture│  │ - YOLOv8n → ONNX → OpenVINO IR  │   │
│  │  qua local      │  │ - YOLOv8n-pose → ONNX → IR     │   │
│  │  bridge)       │  └──────────────┬──────────────────┘   │
│  └────────────────┘                ▼                        │
│                           ┌──────────────────────────┐      │
│                           │ Alert Service             │      │
│                           │ - ROI Matcher             │      │
│                           │ - Violation Classifier    │      │
│                           │ - Cooldown Manager        │      │
│                           │ - Dispatcher              │      │
│                           └──────────┬───────────────┘      │
│                                      ▼                       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Dashboard (Web UI)         │ Storage (SQLite + Files)  │  │
│  │ - Multi-camera grid        │ - violations table        │  │
│  │ - Alert top bar (nhấp nháy)│ - roi_configs table       │  │
│  │ - ROI drawing tool (admin) │ - Thumbnail files         │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Máy chính đóng 2 vai trò:**
1. **Edge** cho webcam USB cắm trực tiếp (gửi frame qua queue nội bộ)
2. **Server** nhận frame từ edge ngoài (MQTT) + chạy inference + dashboard

---

## 3. Edge Agent

### 3.1 Nhiệm vụ

- Capture frame từ nguồn (RTSP / USB)
- Skip frame thông minh (giảm 30fps → 5fps)
- Crop ROI (tải config từ server)
- Resize về 416×416
- Gửi frame đã xử lý lên server qua MQTT (hoặc queue nội bộ nếu cùng máy)

### 3.2 Chi tiết kỹ thuật

| Mục | Quyết định |
|-----|-----------|
| Ngôn ngữ | Python 3.10+ |
| Capture | OpenCV `cv2.VideoCapture` (RTSP + USB) |
| Frame rate gửi | 5 fps (config được) |
| Skip logic | Motion detection (frame diff > threshold) + fixed skip |
| Resize | 416×416 (YOLO input chuẩn) |
| Giao thức | MQTT (paho-mqtt, QoS 1, binary JPEG payload) |
| Queue nội bộ | `asyncio.Queue` khi webcam USB cắm trực tiếp máy server |
| MQTT Topic | `cv/{camera_id}/frame` — frame data; `cv/{camera_id}/heartbeat` |

### 3.3 Cấu trúc module

```
edge/
  source_manager.py    # Quản lý các nguồn (USB + RTSP)
  frame_processor.py   # Skip frame, crop ROI, resize
  mqtt_publisher.py    # Gửi frame qua MQTT
  local_bridge.py      # Gửi frame nội bộ khi edge = server
  config.yaml          # Cấu hình camera, ROI, MQTT broker
```

---

## 4. Inference Engine

### 4.1 Pipeline

```
Frame từ MQTT/queue → Scheduler (round-robin) → OpenVINO Inference → DetectionResult
```

### 4.2 Model

| Model | Input | Output | Vai trò |
|-------|-------|--------|---------|
| YOLOv8n | (1,3,416,416) | boxes: [xyxy, conf, cls] | Detect person, vest, helmet, boot |
| YOLOv8n-pose | (1,3,416,416) | boxes + keypoints×17 | Pose estimation → detect ngã |

- Export sang ONNX → convert sang OpenVINO IR
- FP16 quantization để tối ưu tốc độ trên CPU Intel

### 4.3 Logic phát hiện

| Tình huống | Cách detect |
|-----------|-------------|
| **NO_HELMET** | Detect `person` + `helmet` → person không có helmet box overlap → vi phạm |
| **NO_VEST** | Detect `person` + `vest` → person không có vest box overlap → vi phạm |
| **NO_BOOT** | Detect `person` + `boot` ở vùng chân dưới → khó nhất, precision thấp nhất |
| **FALL** | Pose estimation: bbox aspect ratio (w/h > 1.2) + head_y > hip_y + shoulder_center_y > hip_y → fall_score (0-1). Score > 0.6 → alert |

### 4.4 Scheduler

- Round-robin giữa các camera
- Batch size = 1 (low latency)
- Dự kiến: ~15-30ms/frame trên CPU Intel với OpenVINO + YOLOv8n

### 4.5 Cấu trúc module

```
inference/
  mqtt_subscriber.py    # Nhận frame từ MQTT
  model_manager.py      # Load model, warm-up, inference
  detector.py           # Logic detect 4 tình huống
  scheduler.py          # Round-robin scheduler
  local_receiver.py     # Nhận frame từ local_bridge (USB mode)
```

---

## 5. Alert Service

### 5.1 Pipeline

```
DetectionResult → ROI Matcher → Violation Classifier → Cooldown → Dispatcher
```

### 5.2 Chi tiết từng bước

| Bước | Module | Mô tả |
|------|--------|-------|
| ROI Matcher | `roi_matcher.py` | Load ROI polygon từ DB, point-in-polygon test (Shapely), chỉ giữ detection có center trong ROI |
| Violation Classifier | `classifier.py` | Nhận DetectionResult đã filter ROI → phân loại → tạo Violation object |
| Cooldown Manager | `cooldown.py` | Mỗi (camera, type) → cooldown 5s. Dedup bằng IoU tracking đơn giản |
| Dispatcher | `dispatcher.py` | Push WebSocket → dashboard + INSERT DB + lưu thumbnail |

### 5.3 Violation Schema

```python
Violation {
    id: str
    camera_id: str
    timestamp: datetime
    type: "FALL" | "NO_HELMET" | "NO_VEST" | "NO_BOOT"
    severity: "HIGH" | "MEDIUM"
    bbox: [x1, y1, x2, y2]
    thumbnail_path: str
}
```

### 5.4 Cấu trúc module

```
alert/
  roi_matcher.py       # Point-in-polygon, load/save ROI
  classifier.py        # Phân loại vi phạm
  cooldown.py          # Dedup + cooldown logic
  dispatcher.py        # Push dashboard + lưu DB
  db.py                # SQLite operations
```

---

## 6. Dashboard & Admin UI

### 6.1 Màn hình chính (Dashboard)

- **Multi-camera grid**: hiển thị preview frame + ROI overlay cho từng camera
- **Alert top bar**: luôn hiển thị, nhấp nháy đỏ khi có violation mới
- **Panel chi tiết**: khi click vào alert → thumbnail phóng to + bbox + timestamp + nút Acknowledge / False Alarm

### 6.2 Admin UI (ROI Tool)

- Chọn camera từ dropdown
- Vẽ polygon trên frame hiện tại bằng Canvas API
  - Click = thêm điểm
  - Drag = di chuyển điểm
  - Right-click = xóa điểm
  - Double-click = đóng polygon
- Lưu → PUT API → DB
- Tự động nạp lại khi chọn camera

### 6.3 Trang lịch sử (`/history`)

- Filter theo camera, loại vi phạm, khoảng thời gian
- Bảng danh sách + thumbnail nhỏ

### 6.4 Chi tiết kỹ thuật

| Mục | Quyết định |
|-----|-----------|
| Frontend | HTML + JS + Canvas API, không framework |
| Realtime | WebSocket (`/ws/dashboard`) |
| CSS | Dark theme, alert animation (nhấp nháy) |
| WebServer | aiohttp hoặc FastAPI |

### 6.5 API Endpoints

| Method | Path | Mô tả |
|--------|------|-------|
| GET | `/api/cameras` | Danh sách camera + trạng thái |
| GET | `/api/roi/{camera_id}` | Lấy ROI polygon |
| PUT | `/api/roi/{camera_id}` | Lưu ROI polygon |
| GET | `/api/violations?camera=&type=&from=&to=` | Lịch sử vi phạm |
| GET | `/api/violations/{id}/thumbnail` | Ảnh thumbnail |
| WS | `/ws/dashboard` | Stream realtime alerts + preview |

### 6.6 Cấu trúc module

```
dashboard/
  server.py            # aiohttp/FastAPI, serve static + API + WebSocket
  static/
    index.html         # Màn hình chính
    admin.html         # Trang admin ROI tool
    history.html       # Trang lịch sử vi phạm
    app.js             # WebSocket client + canvas ROI tool
    style.css          # Dark theme, alert animation
```

---

## 7. Database Schema

```sql
TABLE violations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id       TEXT    NOT NULL,
    type            TEXT    NOT NULL,  -- FALL | NO_HELMET | NO_VEST | NO_BOOT
    severity        TEXT,             -- HIGH | MEDIUM
    bbox            TEXT,             -- JSON [x1,y1,x2,y2]
    thumbnail_path  TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

TABLE roi_configs (
    camera_id   TEXT PRIMARY KEY,
    polygon     TEXT NOT NULL,         -- JSON [[x,y], [x,y], ...]
    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

---

## 8. Yêu Cầu Phi Chức Năng

| Mục | Chỉ tiêu |
|-----|---------|
| Recall | > 70% |
| Độ trễ | 500ms–1s (frame capture → alert hiển thị) |
| Số camera | Hỗ trợ tối thiểu 8 camera đồng thời |
| Model inference | < 50ms/frame (OpenVINO trên CPU Intel) |
| Băng thông edge→server | ~10KB/frame (JPEG 416×416) — giảm ~10x so với 1080p |
| Cooldown | 5s mỗi (camera, violation_type) |
| Browser | Chrome/Firefox/Edge phiên bản gần nhất |

---

## 9. Luồng Dữ Liệu Tổng

```
[Webcam] ──RTSP──► [Edge Agent] ──MQTT──► [Inference Engine]
                                                │
[USB Webcam] ──cv2──► [local_bridge] ──Queue────┘
                                                │
                                                ▼
                                        DetectionResult
                                                │
                                                ▼
                                          ROI Matcher
                                          (center in polygon?)
                                                │
                                     ┌──────────┴──────────┐
                                     │ NO                  │ YES
                                     ▼                     ▼
                                   Bỏ qua          Violation Classifier
                                                         │
                                                         ▼
                                                   Cooldown Check
                                                   (đã alert < 5s?)
                                                         │
                                                ┌────────┴────────┐
                                                │ SKIP            │ PASS
                                                ▼                 ▼
                                              Bỏ qua     ┌─────────────────┐
                                                         │ WS → Dashboard   │
                                                         │ INSERT → DB      │
                                                         │ Save thumbnail   │
                                                         └─────────────────┘
```

---

## 10. Cấu Trúc Thư Mục Dự Kiến

```
CV/
├── edge/
│   ├── source_manager.py
│   ├── frame_processor.py
│   ├── mqtt_publisher.py
│   ├── local_bridge.py
│   └── config.yaml
├── inference/
│   ├── mqtt_subscriber.py
│   ├── model_manager.py
│   ├── detector.py
│   ├── scheduler.py
│   └── local_receiver.py
├── alert/
│   ├── roi_matcher.py
│   ├── classifier.py
│   ├── cooldown.py
│   ├── dispatcher.py
│   └── db.py
├── dashboard/
│   ├── server.py
│   └── static/
│       ├── index.html
│       ├── admin.html
│       ├── history.html
│       ├── app.js
│       └── style.css
├── models/
│   ├── yolov8n.onnx
│   └── yolov8n-pose.onnx
├── tests/
├── docs/
│   ├── spec.md
│   ├── plan.md
│   └── tasks.md
├── requirements.txt
└── README.md
```

---

## 11. Rủi Ro & Giả Định

| Rủi ro | Mức độ | Giảm thiểu |
|--------|--------|-----------|
| NO_BOOT precision thấp (giày khó thấy từ camera trên cao) | Trung bình | Đặt severity MEDIUM, chấp nhận recall thấp hơn |
| Thiếu dữ liệu thực tế để fine-tune | Cao | Khởi đầu với pretrained model, thu thập dữ liệu dần |
| MQTT broker quá tải với nhiều edge | Thấp | Binary JPEG nhỏ (10KB), 5fps → ~50KB/s/camera |
| OpenVINO perf không đủ cho nhiều camera | Trung bình | Có thể giảm tiếp input size hoặc tăng skip frame |

### Giả định

- Camera cố định, góc nhìn không thay đổi trong quá trình vận hành
- ROI được vẽ một lần bởi người dùng, ít thay đổi
- Môi trường công trường có ánh sáng ban ngày (không yêu cầu night vision)
- Mỗi người trong khung hình có kích thước tối thiểu ~50px chiều cao
