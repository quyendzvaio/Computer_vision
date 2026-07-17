"""Persistence ports and a deterministic Phase-1 in-memory adapter."""

import threading
from collections.abc import Mapping
from typing import Protocol
from uuid import UUID

from server.api.schemas import ConfigurationResponse
from shared.schemas import DeviceHeartbeat, DeviceMetrics, SafetyEvent


class ControlPlaneRepository(Protocol):
    def ingest_event(self, event: SafetyEvent) -> bool: ...

    def save_heartbeat(self, device_id: str, heartbeat: DeviceHeartbeat) -> None: ...

    def save_metrics(self, device_id: str, metrics: DeviceMetrics) -> None: ...

    def get_configuration(self, device_id: str) -> ConfigurationResponse | None: ...

    def healthy(self) -> bool: ...


class InMemoryControlPlaneRepository:
    """Phase-1 adapter used until the Alembic/PostgreSQL milestone."""

    def __init__(
        self,
        configurations: Mapping[str, ConfigurationResponse] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self.events: dict[UUID, SafetyEvent] = {}
        self.heartbeats: dict[str, DeviceHeartbeat] = {}
        self.metrics: dict[str, DeviceMetrics] = {}
        self.configurations = dict(configurations or {})

    def ingest_event(self, event: SafetyEvent) -> bool:
        with self._lock:
            if event.event_id in self.events:
                return False
            self.events[event.event_id] = event
            return True

    def save_heartbeat(self, device_id: str, heartbeat: DeviceHeartbeat) -> None:
        with self._lock:
            self.heartbeats[device_id] = heartbeat

    def save_metrics(self, device_id: str, metrics: DeviceMetrics) -> None:
        with self._lock:
            self.metrics[device_id] = metrics

    def get_configuration(self, device_id: str) -> ConfigurationResponse | None:
        with self._lock:
            return self.configurations.get(device_id)

    def healthy(self) -> bool:
        return True
