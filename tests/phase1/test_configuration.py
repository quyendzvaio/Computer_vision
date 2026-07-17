from pathlib import Path

import pytest
from pydantic import ValidationError

from edge_runtime.config import load_edge_configuration
from shared.schemas import EdgeConfiguration, ROIConfig


def test_split_configuration_loads() -> None:
    config = load_edge_configuration(Path("edge_runtime/config"))
    assert config.device_id == "edge-01"
    assert len(config.cameras) == 2
    assert len(config.models) == 3
    assert config.scheduler.detector_batch_size == 1
    assert config.fall.weights.model_dump() == {
        "descent": 0.25,
        "rotation": 0.20,
        "horizontal_motion": 0.10,
        "posture": 0.20,
        "persistence": 0.15,
        "inactivity": 0.10,
    }


def test_roi_coordinates_must_be_normalized() -> None:
    with pytest.raises(ValidationError):
        ROIConfig(
            roi_id="zone-a",
            camera_id="cam-01",
            polygon=[(0.1, 0.1), (640, 0.2), (0.2, 0.8)],
        )


def test_configuration_rejects_duplicate_camera_ids() -> None:
    camera = {
        "camera_id": "cam-01",
        "device_path": 0,
        "resolution": [640, 480],
    }
    with pytest.raises(ValidationError, match="camera_id values must be unique"):
        EdgeConfiguration(device_id="edge-01", cameras=[camera, camera])


def test_fall_threshold_order_is_validated() -> None:
    with pytest.raises(ValidationError, match="recovery < candidate < confirmation"):
        EdgeConfiguration(
            device_id="edge-01",
            fall={"recovery_score": 0.6, "candidate_score": 0.5},
        )
