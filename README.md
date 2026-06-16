# CV Safety Monitor v2

AI camera safety monitor. Detects zone intrusion + PPE violations (helmet/vest/boot) in realtime.

## Architecture

```
┌── Windows Edge (NATIVE) ────┐      ┌── Ubuntu GPU Server (DOCKER) ─┐
│                              │      │                               │
│  edge/sender.py              │      │  Container: cv-server          │
│  ┌──────────┐  ┌─────────┐  │      │  ┌──────────┐  ┌───────────┐  │
│  │ USB Cam1 │→│ ZMQ PUB │──┼──TCP──┼─→│ZMQ SUB   │→│ YOLOv8    │  │
│  │ USB Cam2 │→│ ZMQ PUB │──┼──TCP──┼─→│ cam thread│  │ detect    │  │
│  └──────────┘  └─────────┘  │ :5555 │  └──────────┘  ├───────────┤  │
│                              │ :5556 │              │ ROI check │  │
│  Dependencies: opencv, pyzmq │      │              ├───────────┤  │
│  No Docker (USB passthrough  │      │              │ WebServer │  │
│  impossible on Windows)       │      │              │ :8080     │  │
└──────────────────────────────┘      │              └───────────┘  │
                                       │                             │
                                       │  Dependencies: ONNX, PyQt5  │
                                       │  (offscreen), FastAPI        │
                                       └─────────────────────────────┘
```

| Component | Platform | Chạy bằng |
|---|---|---|
| Edge sender | Windows | `python edge/sender.py` (native) |
| GPU server | Ubuntu | `docker compose -f docker-compose.gpu.yml up` |
| Dashboard | Browser | http://server-ip:8080 |

## Setup

### 1. GPU Server (Ubuntu + Docker)

```bash
# Clone
git clone <repo> && cd CV

# Download model
mkdir -p gpu/models
wget -O gpu/models/yolov8n.onnx \
  https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.onnx

# Build & run
docker compose -f docker-compose.gpu.yml up -d
```

**Check logs:**
```bash
docker logs -f cv-server
```

Mở firewall:
```bash
sudo ufw allow 5555/tcp
sudo ufw allow 5556/tcp
sudo ufw allow 8080/tcp
```

### 2. Windows Edge (Native)

```powershell
# Python 3.10+ required
python -m venv venv
.\venv\Scripts\activate

pip install -r edge\requirements.txt

# Edit config — set server IP
notepad edge\config.yaml
# → gpu_host: <SERVER_IP>
# → device_path: 0 (USB index)

# Run
python edge\sender.py
```

### 3. Open Dashboard

Browser → http://server-ip:8080

## Config

`edge/config.yaml`:

```yaml
gpu_host: 192.168.1.100   # GPU server IP

cameras:
  - id: cam1
    device_path: 0         # USB index (Windows) / /dev/video0 (Linux)
    zmq_port: 5555         # must match docker-compose
    fps: 15
    resolution: [640, 480]
```

## Project Layout

```
CV/
├── edge/               # Edge device: capture + ZMQ send
│   ├── sender.py       # Windows: OpenCV → ZMQ PUB loop
│   ├── config.yaml     # Camera config + server IP
│   └── requirements.txt
├── gpu/                # GPU server: ZMQ SUB → detect → web
│   ├── main.py         # Entry: PyQt5 app + cam threads
│   ├── cam1_thread.py  # CAM1: person in zone
│   ├── cam2_thread.py  # CAM2: PPE classification
│   ├── detector.py     # YOLOv8n ONNX
│   ├── web_server.py   # FastAPI + WebSocket
│   └── models/         # YOLO ONNX files (gitignored)
├── shared/             # Data models
├── Dockerfile.gpu      # Server container
├── Dockerfile.edge     # Linux edge container (ko dùng cho Windows)
├── docker-compose.gpu.yml
└── requirements.txt    # Server deps
```
