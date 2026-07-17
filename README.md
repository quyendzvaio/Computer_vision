# Construction Safety Monitor

Edge-first construction safety monitoring for two or three USB cameras. The
NVIDIA edge device owns capture and AI analytics; the Ubuntu CPU server is only
the control plane and event/media store.

## Implementation status

Phase 1 foundation and the Phase 2 inference core are implemented:

- one generic capture process per enabled camera;
- process-safe single-slot latest-frame buffers;
- camera reconnect health and a supervisor watchdog;
- typed, versioned configuration and normalized ROI schemas;
- one model registry for all cameras with strict execution-provider policy;
- fair bounded detector scheduling and bounded pose-priority scheduling;
- edge liveness, readiness and Prometheus text metrics;
- CPU-only FastAPI/control-plane and Docker Compose skeleton;
- idempotent event, heartbeat, metrics and configuration API contracts.
- RTMDet MMDeploy preprocessing/postprocessing and shared runtime wiring;
- verified RTMPose-s SimCC preprocessing, decoding and full-frame mapping;
- one ByteTrack state machine per camera with `(camera_id, track_id)` identity;
- keypoint-derived head, torso, feet and upper-body ROI evidence geometry;
- a native Windows edge service package using WinSW.

The official RTMPose ONNX and RTMDet source checkpoint are checksum-pinned and
can be fetched locally. RTMDet still requires the official MMDeploy export, and
PPE remains `UNVERIFIABLE` until vetted explicit-negative pretrained weights
are pinned. Models stay disabled by default and realtime performance still
requires a benchmark on the actual Windows edge GPU and USB cameras.

## Architecture

```text
EDGE — Quadro T2000 4 GB
USB cameras → isolated capture processes → latest-frame buffers
            → shared GPU scheduler → tracking/pose/PPE/fall rules
            → event outbox/media → HTTPS

UBUNTU SERVER — CPU only
Nginx → FastAPI → PostgreSQL/MinIO → dashboard/event worker
```

Detection always runs on the full analytics frame. Normalized ROI polygons are
applied later to the relevant head, torso, shoe or fall evidence; the runtime
does not crop the ROI before person detection.

## Validate the edge configuration

```bash
python3 -m edge_runtime.main \
  --config-dir edge_runtime/config \
  --validate-only
```

The example cameras and model entries are deliberately disabled. Replace the
camera placeholders with stable `/dev/v4l/by-id/...` paths and validate model
weights/checksums before setting `enabled: true`.

Configuration is split across:

- `edge_runtime/config/edge.yaml`: device and scheduler settings;
- `edge_runtime/config/cameras.yaml`: independent camera rates and paths;
- `edge_runtime/config/models.yaml`: model path/provider/precision;
- `edge_runtime/config/rules.yaml`: PPE and fall thresholds plus normalized ROI.

On production Linux, install the service template from
`deployment/edge/edge-runtime.service`. The health endpoints default to:

- `GET http://127.0.0.1:8090/health/live`
- `GET http://127.0.0.1:8090/health/ready`
- `GET http://127.0.0.1:8090/metrics`

Readiness remains false until at least one camera and all required pretrained
models are enabled and healthy. There is no silent CUDA-to-CPU fallback.

## Prepare Phase-2 pretrained models

```bash
python3 scripts/fetch_pretrained_models.py --model all
python3 scripts/verify_pretrained_models.py
```

Export RTMDet with `scripts/export_rtmdet_onnx.py` from an official MMDeploy +
MMPose environment, then re-run verification with `--require-detector`. See
`docs/phase2-model-and-geometry.md` for the exact contracts and
`deployment/edge-windows/README.md` for native Windows service installation.

## Run the server stack

```bash
cp deployment/server/.env.example deployment/server/.env
# Replace every example credential and pin the MinIO image.
docker compose \
  --env-file deployment/server/.env \
  -f deployment/server/compose.yaml \
  up --build
```

The Compose stack contains Nginx, API, dashboard, event-worker, PostgreSQL and
MinIO and requests no NVIDIA runtime. PostgreSQL/MinIO adapters and Alembic
migrations arrive in Phase 4; the current API intentionally uses an in-memory
repository and returns HTTP 503 for media presigning.

## Tests

```bash
pytest -q tests/phase1
pytest -q tests/phase1 tests/phase2
pytest -q
```

The broader legacy prototype suite can be run with `pytest -q tests`. Some
legacy tests are known to fail independently of the new runtime and are kept
visible during migration rather than rewritten to validate obsolete behavior.

## Model policy

Only pretrained models are in scope. The intended stack is:

- RTMDet-tiny/nano pretrained person detector, with a lightweight pretrained
  YOLO backend allowed behind the same interface if integration is more stable;
- RTMPose-s pretrained body pose model at 256×192;
- ByteTrack per camera (no learned weights);
- a pretrained PPE detector that exposes explicit negative classes such as
  `bare_head`, `no_safety_vest` and `no_safety_shoes`.

Absence of positive PPE detection is never sufficient to produce a violation.
Unobservable body regions return `UNVERIFIABLE`; missing weights return
`MODEL_UNAVAILABLE`.

See [Phase 1 foundation](docs/phase1-foundation.md),
[Phase 2 model and geometry](docs/phase2-model-and-geometry.md), and
[fall analytics design](docs/fall-design.md) for migration boundaries and the
rule-based temporal design. Hardware validation is defined in the
[Quadro T2000 benchmark plan](docs/benchmark-plan.md).

## Legacy code

The root `main.py` plus `edge/`, `gpu/`, `inference/`, `alert/` and the old
dashboard are deprecated prototypes. They are retained temporarily for safe
migration, but the new `edge_runtime/` and `server/` packages never import them.
Do not run `gpu.database.init_db()` against production data because the legacy
function performs destructive table drops.
