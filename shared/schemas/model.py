"""Pretrained model manifest schemas."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from shared.enums import ModelRole


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=128)
    role: ModelRole
    version: str = Field(min_length=1, max_length=64)
    path: Path
    backend: str = Field(default="onnxruntime", pattern=r"^[a-z0-9_-]+$")
    provider: str = "CUDAExecutionProvider"
    precision: str = Field(default="fp32", pattern=r"^(fp32|fp16|int8)$")
    input_shape: tuple[int, int]
    enabled: bool = True
    warmup_runs: int = Field(default=2, ge=0, le=20)
    timeout_ms: int = Field(default=1_000, ge=10, le=30_000)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    output_format: str | None = Field(default=None, max_length=64)
    score_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    keypoint_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
