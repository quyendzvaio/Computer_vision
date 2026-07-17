"""Typed schemas shared across deployment boundaries."""

from shared.schemas.camera import CameraConfig, ReconnectConfig
from shared.schemas.configuration import (
    EdgeConfiguration,
    FallRuleConfig,
    FallWeights,
    PPEConfirmationConfig,
    SchedulerConfig,
)
from shared.schemas.event import EventEvidence, SafetyEvent
from shared.schemas.health import DeviceHeartbeat, DeviceMetrics
from shared.schemas.model import ModelConfig
from shared.schemas.roi import ROIConfig, ROIRuleConfig
from shared.schemas.vision import (
    COCO17_KEYPOINT_NAMES,
    BoundingBox,
    Detection,
    Keypoint,
    PoseResult,
    TrackedDetection,
)

__all__ = [
    "CameraConfig",
    "COCO17_KEYPOINT_NAMES",
    "BoundingBox",
    "Detection",
    "DeviceHeartbeat",
    "DeviceMetrics",
    "EdgeConfiguration",
    "EventEvidence",
    "FallRuleConfig",
    "FallWeights",
    "ModelConfig",
    "Keypoint",
    "PPEConfirmationConfig",
    "ReconnectConfig",
    "ROIConfig",
    "ROIRuleConfig",
    "PoseResult",
    "SafetyEvent",
    "SchedulerConfig",
    "TrackedDetection",
]
