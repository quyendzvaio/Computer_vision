"""Test edge sender config loading."""
import os
import tempfile

import pytest
import yaml

from edge.sender import load_config


def test_load_config_minimal():
    """Load a minimal valid config."""
    data = {
        "gpu_host": "192.168.1.100",
        "cameras": [
            {"id": "cam1", "device_path": "/dev/video0", "zmq_port": 5555, "fps": 30, "resolution": [640, 480]},
        ],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        path = f.name
    try:
        cfg = load_config(path)
        assert cfg["gpu_host"] == "192.168.1.100"
        assert len(cfg["cameras"]) == 1
        assert cfg["cameras"][0]["zmq_port"] == 5555
    finally:
        os.unlink(path)


def test_load_config_empty_cameras():
    """Handle camera list empty gracefully."""
    data = {"gpu_host": "127.0.0.1", "cameras": []}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        path = f.name
    try:
        cfg = load_config(path)
        assert len(cfg["cameras"]) == 0
    finally:
        os.unlink(path)
