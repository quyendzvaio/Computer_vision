import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from edge_runtime.inference.model_registry import ModelRegistry
from edge_runtime.inference.onnx_backend import OnnxRuntimeBackend
from shared.enums import ModelHealth, ModelRole
from shared.errors import ModelUnavailableError
from shared.schemas import ModelConfig


class FakeBackend:
    load_count = 0

    def __init__(self, _configuration: ModelConfig) -> None:
        pass

    def load(self) -> None:
        FakeBackend.load_count += 1

    def infer(self, inputs):
        return [inputs]

    def close(self) -> None:
        pass


def model_config(**overrides) -> ModelConfig:
    values = {
        "name": "detector",
        "role": ModelRole.DETECTOR,
        "version": "test",
        "path": Path("models/yolov8n.onnx"),
        "backend": "fake",
        "provider": "CUDAExecutionProvider",
        "input_shape": (416, 416),
    }
    values.update(overrides)
    return ModelConfig(**values)


def test_registry_loads_model_exactly_once() -> None:
    FakeBackend.load_count = 0
    registry = ModelRegistry([model_config()], factories={"fake": FakeBackend})
    first = registry.load("detector")
    second = registry.load("detector")
    assert first is second
    assert FakeBackend.load_count == 1
    assert registry.statuses()[0].health == ModelHealth.READY


def test_onnx_backend_never_silently_falls_back_to_cpu(monkeypatch) -> None:
    fake_ort = SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    backend = OnnxRuntimeBackend(model_config(backend="onnxruntime"))
    with pytest.raises(ModelUnavailableError, match="CUDAExecutionProvider unavailable"):
        backend.load()
