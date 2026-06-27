# CV Safety Monitor

Realtime multi-camera safety monitoring system. Detects zone intrusion and PPE violations (helmet/vest/boot) using YOLOv8n + ROI-first optimization on NVIDIA GPU.

**Target hardware:**
- **Server:** Ubuntu 22.04 + NVIDIA GPU (Quadro T2000, RTX, etc.) — Docker container
- **Edge:** Windows — native (USB passthrough impossible in Docker on Windows)

## Architecture

```
 Windows (edge)                           Ubuntu (server)
   USB Cam1 ──ZMQ PUB───:5555──→ Docker container
   USB Cam2 ──ZMQ PUB───:5556──→ ├── cam1_thread → detect_roi() → ROIChecker
                                     ├── cam2_thread → detect_roi() → PPEChecker + ROIChecker
                                     ├── AlertManager (cooldown + DB)
                                     └── WebServer (:8080) → Dashboard (browser)
```

| Layer | Tech |
|-------|------|
| Capture | OpenCV → ZMQ PUB (320×240 @5fps) |
| Detection | YOLOv8n ONNX on `onnxruntime-gpu` (CUDA) |
| ROI | Crop-to-detect: chỉ inference trên vùng ROI bounding box |
| PPE | MobileNetV3 binary classifiers (helmet/vest/boot) |
| UI | PyQt5 (offscreen) + FastAPI/WebSocket (web dashboard) |
| DB | SQLite (violations, ROI config, cameras) |

### ROI-first detection

Không dùng OpenVINO hay TensorRT. Thay vào đó, crop frame về bounding box của ROI zone trước khi detect:

- Full frame 640×640: ~10ms
- ROI crop 50% frame: ~5ms (**-50%**)
- ROI crop 25% frame: ~3ms (**-70%**)

Accuracy trong ROI zone: không đổi (detect trên crop thay vì full frame).

## Setup

### 1. Server (Ubuntu + Docker)

```bash
git clone <repo> && cd CV

# Download YOLOv8n ONNX
mkdir -p gpu/models
wget -O gpu/models/yolov8n.onnx \
  https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.onnx

# Edit config — set gpu_host = server IP (Windows edge will connect to this)
nano edge/config.yaml
# → gpu_host: <SERVER_IP>

# Build & run (requires nvidia-container-toolkit)
sudo docker compose up -d

# Check logs
sudo docker logs -f cv-server

# Firewall
sudo ufw allow 5555/tcp
sudo ufw allow 5556/tcp
sudo ufw allow 8080/tcp
```

### 2. Edge Machine (Windows — native)

```powershell
git clone <repo> && cd CV

python -m venv venv
.\venv\Scripts\activate
pip install -r edge\requirements.txt

# Edit config — set gpu_host to server IP, device_path for USB cameras
notepad edge\config.yaml

python edge\sender.py
```

### 3. Dashboard

Browser → http://<SERVER_IP>:8080

## Project Layout

```
CV/
├── edge/                   # Edge: capture + ZMQ send
│   ├── sender.py           # USB cam → JPEG → ZMQ PUB
│   ├── config.yaml         # Cam config + server IP
│   └── requirements.txt
├── gpu/                    # Server: receive → detect → serve
│   ├── main.py             # PyQt5 app coordinator
│   ├── cam1_thread.py      # Cam1: zone intrusion detection
│   ├── cam2_thread.py      # Cam2: PPE + zone detection
│   ├── detector.py         # YOLOv8n ONNX (CUDA) + detect_roi()
│   ├── roi_checker.py      # ROI bounds + point-in-polygon
│   ├── classifier.py       # MobileNetV3 PPE classifiers
│   ├── ppe_checker.py      # PPE violation logic
│   ├── overlay.py          # Bbox + label drawing
│   ├── alert_manager.py    # Cooldown + DB logging
│   ├── database.py         # SQLite helpers
│   ├── web_server.py       # FastAPI + WebSocket
│   ├── roi_drawer.py       # PyQt ROI drawing overlay
│   ├── camera_widget.py    # PyQt camera widget
│   ├── main_window.py      # PyQt main window
│   └── models/             # YOLOv8n ONNX (gitignored)
├── shared/
│   ├── models.py           # Data models (BBox, Detection, etc.)
├── tests/gpu/              # Phase 1 tests
├── Dockerfile              # GPU server container
├── docker-compose.yml      # Docker Compose for Ubuntu server
├── requirements-gpu.txt
└── edge/requirements.txt
```

## Config

`edge/config.yaml`:

```yaml
gpu_host: 192.168.1.100      # GPU server IP
jpeg_quality: 55              # JPEG compression (0-100)
cameras:
  - id: cam1
    device_path: 0            # USB index
    zmq_port: 5555
    fps: 5
    resolution: [320, 240]
  - id: cam2
    device_path: 1
    zmq_port: 5556
    fps: 5
    resolution: [320, 240]
```
