"""Strict ONNX Runtime backend with no implicit CPU provider fallback."""

import hashlib
from pathlib import Path

import numpy as np

from shared.errors import ModelUnavailableError
from shared.schemas import ModelConfig


class OnnxRuntimeBackend:
    def __init__(self, configuration: ModelConfig) -> None:
        self.configuration = configuration
        self._session = None
        self._input_name = ""

    @property
    def session(self):
        if self._session is None:
            raise ModelUnavailableError(f"model {self.configuration.name} is not loaded")
        return self._session

    def load(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModelUnavailableError("onnxruntime is not installed") from exc

        model_path = Path(self.configuration.path)
        if not model_path.is_file():
            raise ModelUnavailableError(f"model file does not exist: {model_path}")
        available = ort.get_available_providers()
        provider = self.configuration.provider
        if provider not in available:
            raise ModelUnavailableError(
                f"required provider {provider} unavailable; available={available}"
            )
        expected_sha256 = self.configuration.sha256
        sidecar = model_path.with_suffix(model_path.suffix + ".sha256")
        if expected_sha256 is None and sidecar.is_file():
            expected_sha256 = sidecar.read_text(encoding="ascii").strip().split()[0]
        if expected_sha256 is None:
            raise ModelUnavailableError(
                f"model checksum is not configured and sidecar is missing: {sidecar}"
            )
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ModelUnavailableError(
                f"checksum mismatch for {model_path}: expected "
                f"{expected_sha256}, got {digest}"
            )

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if provider != "CPUExecutionProvider":
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        try:
            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=[provider],
            )
        except Exception as exc:
            raise ModelUnavailableError(
                f"failed to load {self.configuration.name} with {provider}: {exc}"
            ) from exc
        active = self._session.get_providers()
        if not active or active[0] != provider:
            self.close()
            raise ModelUnavailableError(
                f"provider mismatch for {self.configuration.name}: active={active}"
            )
        self._input_name = self._session.get_inputs()[0].name
        self._warmup()

    def _warmup(self) -> None:
        if self._session is None or self.configuration.warmup_runs == 0:
            return
        input_meta = self._session.get_inputs()[0]
        shape = [
            dimension if isinstance(dimension, int) else 1
            for dimension in input_meta.shape
        ]
        if len(shape) != 4:
            raise ModelUnavailableError(
                f"expected a 4D input for {self.configuration.name}, got {shape}"
            )
        tensor = np.zeros(shape, dtype=np.float32)
        for _ in range(self.configuration.warmup_runs):
            self._session.run(None, {self._input_name: tensor})

    def infer(self, inputs: np.ndarray) -> list[np.ndarray]:
        outputs = self.session.run(None, {self._input_name: inputs})
        return [np.asarray(output) for output in outputs]

    @property
    def input_shape(self) -> tuple[object, ...]:
        return tuple(self.session.get_inputs()[0].shape)

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(output.name for output in self.session.get_outputs())

    def close(self) -> None:
        self._session = None
        self._input_name = ""
