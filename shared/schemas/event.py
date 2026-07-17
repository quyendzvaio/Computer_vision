"""Idempotent safety event protocol."""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from shared.enums import EventStatus, RuleType, Severity
from shared.protocol import PROTOCOL_VERSION


class EventEvidence(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    visible_keypoint_ratio: float | None = Field(default=None, ge=0, le=1)
    roi_overlap_ratio: float | None = Field(default=None, ge=0, le=1)
    roi_overlap_duration_ms: int | None = Field(default=None, ge=0)
    torso_angle: float | None = None
    torso_angular_velocity: float | None = None
    normalized_head_velocity: float | None = None
    normalized_hip_velocity: float | None = None
    fall_score: float | None = Field(default=None, ge=0, le=1)


class SafetyEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: str = PROTOCOL_VERSION
    event_id: UUID = Field(default_factory=uuid4)
    device_id: str = Field(min_length=1, max_length=128)
    camera_id: str = Field(min_length=1, max_length=128)
    track_id: int = Field(ge=0)
    roi_id: str = Field(min_length=1, max_length=128)
    rule_type: RuleType
    status: EventStatus = EventStatus.CONFIRMED
    severity: Severity
    confidence: float = Field(ge=0, le=1)
    started_at: datetime
    confirmed_at: datetime
    evidence: EventEvidence = EventEvidence()

    @field_validator("started_at", "confirmed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("event timestamps must be timezone-aware")
        return value.astimezone(timezone.utc)
