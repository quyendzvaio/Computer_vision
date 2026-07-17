from datetime import datetime, timezone
from uuid import uuid4

from fastapi.testclient import TestClient

from server.api.main import create_app
from server.api.repositories import InMemoryControlPlaneRepository
from server.api.schemas import ConfigurationResponse


def event_payload(event_id):
    now = datetime.now(timezone.utc).isoformat()
    return {
        "event_id": str(event_id),
        "device_id": "edge-01",
        "camera_id": "cam-01",
        "track_id": 12,
        "roi_id": "zone-a",
        "rule_type": "FALL",
        "status": "CONFIRMED",
        "severity": "CRITICAL",
        "confidence": 0.8,
        "started_at": now,
        "confirmed_at": now,
        "evidence": {"fall_score": 0.8},
    }


def test_event_ingestion_is_idempotent() -> None:
    repository = InMemoryControlPlaneRepository()
    event_id = uuid4()
    with TestClient(create_app(repository)) as client:
        first = client.post(
            "/api/v1/events",
            json=event_payload(event_id),
            headers={"Idempotency-Key": str(event_id)},
        )
        duplicate = client.post(
            "/api/v1/events",
            json=event_payload(event_id),
            headers={"Idempotency-Key": str(event_id)},
        )
    assert first.status_code == 201
    assert duplicate.status_code == 200
    assert duplicate.json()["duplicate"] is True
    assert len(repository.events) == 1


def test_configuration_and_device_telemetry_contracts() -> None:
    repository = InMemoryControlPlaneRepository(
        {"edge-01": ConfigurationResponse(device_id="edge-01", version=3)}
    )
    with TestClient(create_app(repository)) as client:
        config = client.get("/api/v1/devices/edge-01/configuration")
        heartbeat = client.post(
            "/api/v1/devices/edge-01/heartbeat",
            json={"configuration_version": 3, "ready": True},
        )
        metrics = client.post(
            "/api/v1/devices/edge-01/metrics",
            json={"gauges": {"capture_fps": 15.0}, "counters": {}},
        )
    assert config.status_code == 200 and config.json()["version"] == 3
    assert heartbeat.status_code == 202
    assert metrics.status_code == 202


def test_media_adapter_is_explicitly_unavailable_in_phase_one() -> None:
    with TestClient(create_app(InMemoryControlPlaneRepository())) as client:
        response = client.post(
            "/api/v1/media/presign",
            json={
                "event_id": str(uuid4()),
                "media_type": "snapshot",
                "content_type": "image/jpeg",
                "size_bytes": 100,
                "sha256": "0" * 64,
            },
        )
    assert response.status_code == 503
