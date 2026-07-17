"""Backend-neutral inference contracts."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from edge_runtime.capture.latest_frame_buffer import FramePacket
from shared.enums import ModelHealth, PosePriority


@dataclass(frozen=True, slots=True)
class ModelStatus:
    name: str
    health: ModelHealth
    provider: str
    detail: str = ""


@runtime_checkable
class ModelBackend(Protocol):
    """Low-level model runtime owned exactly once by ModelRegistry."""

    def load(self) -> None: ...

    def infer(self, inputs: np.ndarray) -> list[np.ndarray]: ...

    def close(self) -> None: ...


@runtime_checkable
class FrameInferencePipeline(Protocol):
    """Model-specific pre/inference/post pipeline consumed by the scheduler."""

    def infer_frames(self, frames: Sequence[FramePacket]) -> Sequence[Any]: ...


@dataclass(frozen=True, slots=True)
class ScheduledResult:
    camera_id: str
    frame_id: int
    captured_at_unix_ns: int
    frame_age_ms: float
    output: Any
    frame: FramePacket | None = None


@dataclass(frozen=True, slots=True)
class PoseRequest:
    camera_id: str
    track_id: int
    frame: FramePacket
    bbox_xyxy: tuple[float, float, float, float]
    priority: PosePriority
    requested_at_monotonic_ns: int
