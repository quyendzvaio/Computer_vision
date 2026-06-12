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
