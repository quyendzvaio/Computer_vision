"""Shared model ownership and bounded multi-camera inference scheduling."""

from edge_runtime.inference.model_registry import ModelRegistry
from edge_runtime.inference.scheduler import SharedInferenceScheduler

__all__ = ["ModelRegistry", "SharedInferenceScheduler"]
