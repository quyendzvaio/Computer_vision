"""Domain enums shared by edge and control-plane services."""

from enum import Enum


class _StringEnum(str, Enum):
    """Python 3.10-compatible equivalent of enum.StrEnum."""

    def __str__(self) -> str:
        return self.value


class CameraState(_StringEnum):
    DISABLED = "DISABLED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    DISCONNECTED = "DISCONNECTED"
    STOPPED = "STOPPED"


class ModelHealth(_StringEnum):
    DISABLED = "DISABLED"
    LOADING = "LOADING"
    READY = "READY"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"


class PPEState(_StringEnum):
    PRESENT = "PRESENT"
    MISSING = "MISSING"
    UNVERIFIABLE = "UNVERIFIABLE"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"


class FallState(_StringEnum):
    NORMAL = "NORMAL"
    RAPID_TRANSITION = "RAPID_TRANSITION"
    POSSIBLE_FALL = "POSSIBLE_FALL"
    CONFIRMED_FALL = "CONFIRMED_FALL"
    RECOVERING = "RECOVERING"
    UNVERIFIABLE = "UNVERIFIABLE"


class TrackLifecycle(_StringEnum):
    CREATED = "CREATED"
    ACTIVE = "ACTIVE"
    LOST = "LOST"
    EXPIRED = "EXPIRED"


class RuleType(_StringEnum):
    NO_HELMET = "NO_HELMET"
    NO_VEST = "NO_VEST"
    NO_SAFETY_SHOES = "NO_SAFETY_SHOES"
    FALL = "FALL"


class EventStatus(_StringEnum):
    CONFIRMED = "CONFIRMED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"


class Severity(_StringEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ModelRole(_StringEnum):
    DETECTOR = "detector"
    PPE = "ppe"
    POSE = "pose"


class PosePriority(_StringEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
