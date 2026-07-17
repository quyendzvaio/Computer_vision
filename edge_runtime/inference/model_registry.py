"""Load every enabled pretrained model at most once per edge runtime."""

import threading
from collections.abc import Callable
from dataclasses import dataclass

from edge_runtime.inference.interfaces import ModelBackend, ModelStatus
from edge_runtime.inference.onnx_backend import OnnxRuntimeBackend
from shared.enums import ModelHealth, ModelRole
from shared.errors import ModelUnavailableError
from shared.schemas import ModelConfig

BackendFactory = Callable[[ModelConfig], ModelBackend]


@dataclass(slots=True)
class _ModelRecord:
    configuration: ModelConfig
    health: ModelHealth = ModelHealth.DISABLED
    detail: str = ""
    backend: ModelBackend | None = None


class ModelRegistry:
    """Single owner of model sessions; load operations are idempotent."""

    def __init__(
        self,
        configurations: list[ModelConfig],
        factories: dict[str, BackendFactory] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._factories: dict[str, BackendFactory] = {
            "onnxruntime": OnnxRuntimeBackend,
            **(factories or {}),
        }
        self._records = {
            configuration.name: _ModelRecord(
                configuration=configuration,
                health=(ModelHealth.UNAVAILABLE if configuration.enabled else ModelHealth.DISABLED),
            )
            for configuration in configurations
        }
        if len(self._records) != len(configurations):
            raise ValueError("model names must be unique")

    def load_enabled(self) -> None:
        for name, record in self._records.items():
            if record.configuration.enabled:
                self.load(name)

    def load(self, name: str) -> ModelBackend:
        with self._lock:
            record = self._record(name)
            if not record.configuration.enabled:
                raise ModelUnavailableError(f"model {name} is disabled")
            if record.health == ModelHealth.READY and record.backend is not None:
                return record.backend
            factory = self._factories.get(record.configuration.backend)
            if factory is None:
                record.health = ModelHealth.UNAVAILABLE
                record.detail = f"unknown backend: {record.configuration.backend}"
                raise ModelUnavailableError(record.detail)
            record.health = ModelHealth.LOADING
            record.detail = ""
            backend = factory(record.configuration)
            try:
                backend.load()
            except Exception as exc:
                record.health = ModelHealth.UNAVAILABLE
                record.detail = str(exc)
                record.backend = None
                raise ModelUnavailableError(
                    f"model {name} failed to load: {exc}"
                ) from exc
            record.backend = backend
            record.health = ModelHealth.READY
            return backend

    def get(self, name: str) -> ModelBackend:
        with self._lock:
            record = self._record(name)
            if record.health != ModelHealth.READY or record.backend is None:
                raise ModelUnavailableError(f"model {name} is not ready: {record.detail}")
            return record.backend

    def configuration(self, name: str) -> ModelConfig:
        """Return the immutable contract paired with a loaded backend."""
        with self._lock:
            return self._record(name).configuration

    def names_for_role(self, role: ModelRole) -> list[str]:
        return [
            name
            for name, record in self._records.items()
            if record.configuration.role == role and record.configuration.enabled
        ]

    def statuses(self) -> list[ModelStatus]:
        with self._lock:
            return [
                ModelStatus(
                    name=name,
                    health=record.health,
                    provider=record.configuration.provider,
                    detail=record.detail,
                )
                for name, record in self._records.items()
            ]

    def close(self) -> None:
        with self._lock:
            for record in self._records.values():
                if record.backend is not None:
                    record.backend.close()
                record.backend = None
                record.health = (
                    ModelHealth.UNAVAILABLE
                    if record.configuration.enabled
                    else ModelHealth.DISABLED
                )

    def _record(self, name: str) -> _ModelRecord:
        try:
            return self._records[name]
        except KeyError as exc:
            raise KeyError(f"unknown model: {name}") from exc
