"""Load the split edge YAML files into one validated configuration."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from shared.errors import ConfigurationError
from shared.schemas import EdgeConfiguration


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"configuration file {path} must contain a mapping")
    return value


def load_edge_configuration(config_dir: str | Path) -> EdgeConfiguration:
    """Load edge.yaml plus optional cameras/models/rules overlays."""

    root = Path(config_dir)
    merged = _read_yaml(root / "edge.yaml")
    overlay_keys = {
        "cameras.yaml": "cameras",
        "models.yaml": "models",
        "rules.yaml": None,
    }
    for filename, key in overlay_keys.items():
        overlay = _read_yaml(root / filename)
        if not overlay:
            continue
        if key is not None and key not in overlay:
            overlay = {key: overlay}
        merged.update(overlay)
    try:
        return EdgeConfiguration.model_validate(merged)
    except ValidationError as exc:
        raise ConfigurationError(str(exc)) from exc
