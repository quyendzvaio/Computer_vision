"""Device heartbeat and metrics ingestion schemas."""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DeviceHeartbeat(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    configuration_version: int = Field(ge=0)
    ready: bool
    details: dict[str, Any] = Field(default_factory=dict)


class DeviceMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    gauges: dict[str, float] = Field(default_factory=dict)
    counters: dict[str, int] = Field(default_factory=dict)
