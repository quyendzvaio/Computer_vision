"""Validated edge runtime configuration and initial temporal thresholds."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from shared.schemas.camera import CameraConfig
from shared.schemas.model import ModelConfig
from shared.schemas.roi import ROIConfig


class SchedulerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_frame_age_ms: int = Field(default=500, ge=50, le=10_000)
    detector_batch_size: int = Field(default=1, ge=1, le=8)
    inference_timeout_ms: int = Field(default=1_000, ge=10, le=30_000)
    idle_poll_ms: int = Field(default=5, ge=1, le=1_000)
    circuit_breaker_failures: int = Field(default=3, ge=1, le=100)
    circuit_breaker_reset_ms: int = Field(default=10_000, ge=100, le=300_000)
    pose_fps_low: float = Field(default=2.0, gt=0, le=30)
    pose_fps_medium: float = Field(default=6.0, gt=0, le=30)
    pose_fps_high: float = Field(default=12.0, gt=0, le=30)


class FallWeights(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    descent: float = Field(default=0.25, ge=0, le=1)
    rotation: float = Field(default=0.20, ge=0, le=1)
    horizontal_motion: float = Field(default=0.10, ge=0, le=1)
    posture: float = Field(default=0.20, ge=0, le=1)
    persistence: float = Field(default=0.15, ge=0, le=1)
    inactivity: float = Field(default=0.10, ge=0, le=1)

    @model_validator(mode="after")
    def validate_sum(self) -> "FallWeights":
        total = sum(self.model_dump().values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError("fall weights must sum to 1.0")
        return self


class FallRuleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    window_ms: int = Field(default=3_500, ge=500, le=10_000)
    minimum_visible_keypoint_ratio: float = Field(default=0.45, ge=0, le=1)
    upright_angle_deg: float = Field(default=25.0, ge=0, le=90)
    abnormal_angle_deg: float = Field(default=55.0, ge=0, le=90)
    normalized_descent_velocity: float = Field(default=0.8, ge=0)
    normalized_horizontal_velocity: float = Field(default=0.75, ge=0)
    angular_velocity_deg_s: float = Field(default=60.0, ge=0)
    candidate_score: float = Field(default=0.55, ge=0, le=1)
    confirmation_score: float = Field(default=0.72, ge=0, le=1)
    recovery_score: float = Field(default=0.35, ge=0, le=1)
    maximum_transition_ms: int = Field(default=1_200, ge=100, le=10_000)
    persistence_ms: int = Field(default=800, ge=100, le=10_000)
    inactivity_ms: int = Field(default=500, ge=0, le=10_000)
    recovery_ms: int = Field(default=1_500, ge=100, le=30_000)
    roi_minimum_overlap_frames: int = Field(default=3, ge=1, le=100)
    roi_minimum_overlap_duration_ms: int = Field(default=250, ge=0, le=10_000)
    roi_minimum_region_overlap_ratio: float = Field(default=0.10, ge=0, le=1)
    weights: FallWeights = FallWeights()

    @model_validator(mode="after")
    def validate_threshold_order(self) -> "FallRuleConfig":
        if self.upright_angle_deg >= self.abnormal_angle_deg:
            raise ValueError("upright angle must be below abnormal angle")
        if not self.recovery_score < self.candidate_score < self.confirmation_score:
            raise ValueError("fall scores must satisfy recovery < candidate < confirmation")
        return self


class PPEConfirmationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    negative_confidence: float = Field(default=0.65, ge=0, le=1)
    positive_confidence: float = Field(default=0.60, ge=0, le=1)
    minimum_positive_samples: int = Field(default=3, ge=1, le=100)
    sample_window: int = Field(default=5, ge=1, le=100)
    recovery_samples: int = Field(default=2, ge=1, le=100)


class EdgeConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = Field(default=1, ge=1)
    device_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")
    cameras: list[CameraConfig] = Field(default_factory=list)
    models: list[ModelConfig] = Field(default_factory=list)
    rois: list[ROIConfig] = Field(default_factory=list)
    scheduler: SchedulerConfig = SchedulerConfig()
    fall: FallRuleConfig = FallRuleConfig()
    ppe: PPEConfirmationConfig = PPEConfirmationConfig()

    @model_validator(mode="after")
    def validate_references(self) -> "EdgeConfiguration":
        camera_ids = [camera.camera_id for camera in self.cameras]
        if len(camera_ids) != len(set(camera_ids)):
            raise ValueError("camera_id values must be unique")
        model_names = [model.name for model in self.models]
        if len(model_names) != len(set(model_names)):
            raise ValueError("model names must be unique")
        unknown = {roi.camera_id for roi in self.rois} - set(camera_ids)
        if unknown:
            raise ValueError(f"ROI references unknown cameras: {sorted(unknown)}")
        return self
