# Phase 1 foundation and migration boundary

## New runtime boundaries

- `edge_runtime/` owns camera processes, shared frame transport, model sessions,
  GPU scheduling and edge health.
- `server/` owns CPU-only API contracts and future persistence adapters.
- `shared/schemas/` is the only domain protocol imported by both deployments.
- `deployment/edge/` and `deployment/server/` have independent dependencies.
- `models/manifest.yaml` records intended pretrained artifacts without claiming
  unverified weights are installed.

The new runtime has no imports from `edge/`, `gpu/`, `inference/`, `alert/` or
the legacy dashboard.

## Phase 1 file migration

Created:

- `edge_runtime/main.py`, `supervisor.py`, `shutdown.py`;
- `edge_runtime/capture/` process, health, registry and latest-frame modules;
- `edge_runtime/inference/` interfaces, registry, ONNX backend and scheduler;
- `edge_runtime/health/` HTTP health and metrics modules;
- `edge_runtime/config/*.yaml` and its typed loader;
- `shared/enums.py`, `protocol.py`, `errors.py`, `shared/schemas/`;
- `server/api/`, `server/event_worker/`, `server/dashboard/`;
- `deployment/edge/` and the CPU-only `deployment/server/` stack;
- `tests/phase1/`.

Modified:

- `README.md` now describes edge-first execution and actual phase status;
- `pyproject.toml` uses Phase 1 plus stable GPU prototype tests as the migration
  gate and configures Ruff/mypy defaults.

Deprecated but retained:

- root `main.py`, `Dockerfile`, `Dockerfile.cpu`, `docker-compose.yml`;
- `edge/`, `gpu/`, `inference/`, `alert/`, `dashboard/`.

No production data or legacy model file is deleted in Phase 1.

## Compatibility and data migration

The legacy repository has two incompatible SQLite schemas sharing `data/cv.db`.
Phase 1 does not open or mutate that database. Before Phase 4:

1. make an immutable backup;
2. inspect which legacy schema is present;
3. import data idempotently into PostgreSQL with Alembic-managed tables;
4. normalize a pixel ROI only when its original frame resolution is known;
5. preserve unresolved pixel polygons as `requires_review` instead of guessing;
6. verify media checksum/upload before deleting any local legacy file.

The edge SQLite database introduced in Phase 4 is an upload outbox only and
must not reuse the legacy server database path.

## Initial pretrained model slots

Model paths and provider policy are configured in
`edge_runtime/config/models.yaml`; descriptive metadata lives in
`models/manifest.yaml`.

| Registry name | Intended pretrained artifact | Current state |
|---|---|---|
| `person-detector` | RTMDet-tiny/nano person detector | disabled; weights absent |
| `human-pose` | RTMPose-s 256×192 body pose | disabled; weights absent |
| `ppe-detector` | explicit positive/negative PPE detector | selection pending validation |

PPE weights will only be accepted if their class map contains explicit negative
evidence or an equivalent negative classifier output. A positive-only detector
cannot infer missing PPE from absence.

## Remaining phases

Phase 2 integrates and validates pretrained models, ByteTrack, keypoint mapping,
body evidence and normalized ROI geometry. Phase 3 adds PPE/fall temporal state.
Phase 4 adds outbox/media/PostgreSQL/MinIO/dashboard. Phase 5 performs replay,
hardware benchmarking and manual threshold adjustment without training.
