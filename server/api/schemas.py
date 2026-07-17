"""Control-plane request and response schemas."""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class EventIngestResponse(BaseModel):
    event_id: UUID
    accepted: bool
    duplicate: bool


class ConfigurationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device_id: str
    version: int = Field(ge=0)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    configuration: dict[str, Any] = Field(default_factory=dict)


class MediaPresignRequest(BaseModel):
    event_id: UUID
    media_type: str = Field(pattern=r"^(snapshot|clip)$")
    content_type: str = Field(pattern=r"^(image/jpeg|video/mp4)$")
    size_bytes: int = Field(gt=0, le=2_000_000_000)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class HealthResponse(BaseModel):
    status: str
    component: str
