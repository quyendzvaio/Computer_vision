"""End-to-end integration tests for the full pipeline."""
import json
import asyncio
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


@pytest.mark.asyncio
async def test_alert_pipeline_end_to_end(temp_dir):
    """Full pipeline from detection to dispatch with in-memory DB."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import alert.db as db_module
    original_path = db_module.DB_PATH
    db_module.DB_PATH = temp_dir / "test.db"
    db_module.init_db()

    try:
        from alert.roi_matcher import ROIMatcher
        from alert.classifier import ViolationClassifier
        from alert.cooldown import CooldownManager
        from alert.dispatcher import Dispatcher
        from alert import AlertPipeline
        from shared.models import DetectionResult, DetectedObject, BBox

        # Setup pipeline with no cooldown
        roi = ROIMatcher(db=db_module)  # No ROI configs -> allow all
        classifier = ViolationClassifier(confidence_threshold=0.4)
        cooldown = CooldownManager(cooldown_seconds=0)
        dispatcher = Dispatcher(db=db_module, ws_manager=None, thumbnail_dir=str(temp_dir / "thumbs"))

        pipeline = AlertPipeline(roi, classifier, cooldown, dispatcher)

        # Create a detection with person but no helmet and no vest
        result = DetectionResult(
            camera_id="cam-01",
            objects=[
                DetectedObject(bbox=BBox(100, 100, 200, 300), cls="person", conf=0.9),
            ],
        )

        violations = pipeline.process(result, frame_bgr=None)
        types = {v.type for v in violations}

        assert "NO_HELMET" in types
        assert "NO_VEST" in types
        assert len(violations) >= 2

        # Verify DB insert
        rows = db_module.get_violations()
        assert len(rows) >= 2

    finally:
        db_module.DB_PATH = original_path


@pytest.mark.asyncio
async def test_scheduler_to_detector_flow():
    """Scheduler -> Detector -> AlertPipeline integration."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from inference.scheduler import Scheduler
    from alert.cooldown import CooldownManager
    from shared.models import Violation

    # Create mock detector that returns known result
    mock_mm = MagicMock()
    mock_mm.preprocess_jpeg.return_value = None
    mock_mm.detect.return_value = []

    from inference.detector import Detector
    detector = Detector(mock_mm)

    # Setup scheduler
    scheduler = Scheduler()
    scheduler.register_camera("cam-01")
    scheduler.add_frame("cam-01", b"fake_jpeg")

    # Poll and detect
    result = scheduler.poll()
    assert result is not None
    camera_id, jpeg_bytes = result
    assert camera_id == "cam-01"

    detection = detector.run(jpeg_bytes, camera_id)
    assert detection.camera_id == "cam-01"
    assert len(detection.objects) == 0  # Mock returns empty


def test_config_roundtrip():
    """Config YAML can be parsed and camera list is correct."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import yaml

    config_path = Path("edge/config.yaml")
    if not config_path.exists():
        pytest.skip("config.yaml not found")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert "cameras" in config
    assert "mqtt" in config
    assert "frame" in config
    assert isinstance(config["cameras"], list)
    for cam in config["cameras"]:
        assert "id" in cam
        assert "source" in cam
