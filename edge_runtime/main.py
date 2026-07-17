"""Edge entrypoint with isolated capture and Phase-2 inference wiring."""

import argparse
import json
import logging
import sys

from edge_runtime.config import load_edge_configuration
from edge_runtime.health import EdgeHealthServer, MetricsRegistry, ReadinessSnapshot
from edge_runtime.inference import ModelRegistry
from edge_runtime.phase2_runtime import Phase2Runtime
from edge_runtime.shutdown import shutdown_signals
from edge_runtime.supervisor import EdgeSupervisor
from shared.enums import CameraState, ModelHealth, ModelRole
from shared.errors import ConfigurationError, ModelUnavailableError


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            },
            separators=(",", ":"),
        )


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Edge-first construction safety runtime")
    parser.add_argument("--config-dir", default="edge_runtime/config")
    parser.add_argument("--health-host", default="127.0.0.1")
    parser.add_argument("--health-port", type=int, default=8090)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    log = logging.getLogger("edge-runtime")
    args = build_parser().parse_args(argv)
    try:
        configuration = load_edge_configuration(args.config_dir)
    except ConfigurationError as exc:
        log.error("configuration invalid: %s", exc)
        return 2
    if args.validate_only:
        log.info("configuration valid: version=%s", configuration.version)
        return 0

    metrics = MetricsRegistry()
    registry = ModelRegistry(configuration.models)
    try:
        registry.load_enabled()
    except ModelUnavailableError as exc:
        log.error("one or more enabled models failed to load: %s", exc)

    supervisor = EdgeSupervisor(configuration)
    phase2: Phase2Runtime | None = None
    if registry.names_for_role(ModelRole.DETECTOR):
        try:
            phase2 = Phase2Runtime(configuration, supervisor, registry)
        except (ModelUnavailableError, ValueError) as exc:
            log.error("Phase 2 runtime unavailable: %s", exc)

    def readiness() -> ReadinessSnapshot:
        camera_snapshots = supervisor.health_snapshots()
        enabled_cameras = [camera for camera in configuration.cameras if camera.enabled]
        cameras_ready = bool(enabled_cameras) and all(
            camera_snapshots.get(camera.camera_id)
            and camera_snapshots[camera.camera_id].state == CameraState.RUNNING
            for camera in enabled_cameras
        )
        enabled_models = [model for model in configuration.models if model.enabled]
        statuses = registry.statuses()
        models_ready = bool(enabled_models) and all(
            status.health == ModelHealth.READY
            for status in statuses
            if any(model.name == status.name and model.enabled for model in enabled_models)
        )
        return ReadinessSnapshot(
            ready=cameras_ready and models_ready,
            configuration_loaded=True,
            cameras_ready=cameras_ready,
            models_ready=models_ready,
            details={
                "phase": "phase2",
                "inference": "active" if phase2 is not None else "unavailable",
            },
        )

    def render_metrics() -> str:
        for camera_id, snapshot in supervisor.health_snapshots().items():
            safe_id = camera_id.replace("-", "_").replace(".", "_")
            metrics.set_gauge(f"edge_camera_{safe_id}_capture_fps", snapshot.capture_fps)
            metrics.set_gauge(f"edge_camera_{safe_id}_dropped_frames", snapshot.dropped_frames)
            metrics.set_gauge(f"edge_camera_{safe_id}_capture_failures", snapshot.capture_failures)
            metrics.set_gauge(f"edge_camera_{safe_id}_reconnects", snapshot.reconnect_count)
            metrics.set_gauge(
                f"edge_camera_{safe_id}_resolution_mismatches",
                snapshot.resolution_mismatch_count,
            )
        return metrics.render_prometheus()

    health = EdgeHealthServer(
        readiness,
        render_metrics,
        host=args.health_host,
        port=args.health_port,
    )
    supervisor.start()
    if phase2 is not None:
        phase2.start()
    health.start()
    log.info(
        "edge runtime started: cameras=%s phase2=%s",
        len(supervisor.cameras),
        phase2 is not None,
    )
    try:
        with shutdown_signals() as stop:
            stop.wait()
    finally:
        health.stop()
        if phase2 is not None:
            phase2.stop()
        supervisor.stop()
        registry.close()
        log.info("edge runtime stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
