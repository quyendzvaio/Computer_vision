"""Normalized ROI and per-zone rule schemas."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from shared.enums import Severity

UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]
NormalizedPoint = tuple[UnitFloat, UnitFloat]


class ROIRuleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    confirmation_ms: int = Field(default=1_000, ge=0, le=120_000)
    cooldown_ms: int = Field(default=30_000, ge=0, le=3_600_000)


class ROIConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    roi_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")
    camera_id: str = Field(min_length=1, max_length=128)
    polygon: list[NormalizedPoint]
    enabled: bool = True
    priority: int = Field(default=0, ge=0, le=100)
    severity: Severity = Severity.HIGH
    color: str = Field(default="#ff0000", pattern=r"^#[0-9A-Fa-f]{6}$")
    version: int = Field(default=1, ge=1)
    rules: dict[str, ROIRuleConfig] = Field(default_factory=dict)

    @field_validator("polygon")
    @classmethod
    def validate_polygon(cls, points: list[NormalizedPoint]) -> list[NormalizedPoint]:
        if len(points) < 3:
            raise ValueError("ROI polygon requires at least three vertices")
        if len(set(points)) < 3:
            raise ValueError("ROI polygon requires at least three distinct vertices")
        area_twice = sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
        )
        if abs(area_twice) < 1e-8:
            raise ValueError("ROI polygon area must be non-zero")
        return points
