"""Camera configuration schemas."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

PositiveFps = Annotated[float, Field(gt=0, le=120)]


class ReconnectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    initial_delay_ms: int = Field(default=500, ge=100, le=60_000)
    maximum_delay_ms: int = Field(default=30_000, ge=500, le=300_000)
    multiplier: float = Field(default=2.0, ge=1.1, le=10.0)
    read_timeout_ms: int = Field(default=2_000, ge=100, le=30_000)


class CameraConfig(BaseModel):
    """Stable camera identity and independently configurable frame rates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    camera_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")
    device_path: str | int
    enabled: bool = True
    resolution: tuple[int, int] = (1280, 720)
    capture_fps: PositiveFps = 15.0
    analytics_fps: PositiveFps = 6.0
    preview_fps: PositiveFps = 3.0
    clip_fps: PositiveFps = 15.0
    reconnect: ReconnectConfig = ReconnectConfig()

    @field_validator("device_path")
    @classmethod
    def validate_device_path(cls, value: str | int) -> str | int:
        if isinstance(value, str) and not value.strip():
            raise ValueError("device_path must not be empty")
        if isinstance(value, int) and value < 0:
            raise ValueError("camera index must be non-negative")
        return value

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, value: tuple[int, int]) -> tuple[int, int]:
        width, height = value
        if not (160 <= width <= 7680 and 120 <= height <= 4320):
            raise ValueError("resolution is outside the supported range")
        return value
