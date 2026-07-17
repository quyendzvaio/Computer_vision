"""Typed errors that do not depend on FastAPI or persistence libraries."""


class ConfigurationError(ValueError):
    """Configuration failed validation."""


class ModelUnavailableError(RuntimeError):
    """A required model could not be loaded with the configured provider."""


class InferenceTimeoutError(TimeoutError):
    """An inference request exceeded its configured deadline."""


class StaleFrameError(RuntimeError):
    """A frame is older than the analytics freshness budget."""
