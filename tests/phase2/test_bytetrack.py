from edge_runtime.tracking import ByteTracker, PerCameraTrackerManager
from shared.schemas import BoundingBox, Detection


def detection(x: float, score: float) -> Detection:
    return Detection(BoundingBox(x, 10, x + 40, 100), score, 0, "person")


def test_track_identity_is_scoped_per_camera():
    manager = PerCameraTrackerManager(["cam-a", "cam-b"])
    first = manager.update("cam-a", [detection(10, 0.9)])[0]
    second = manager.update("cam-b", [detection(10, 0.9)])[0]
    assert first.identity == ("cam-a", 1)
    assert second.identity == ("cam-b", 1)
    assert first.identity != second.identity


def test_low_score_second_stage_keeps_existing_track_but_does_not_create_one():
    tracker = ByteTracker("cam-a", track_threshold=0.5, low_threshold=0.1)
    assert tracker.update([detection(10, 0.9)])[0].track_id == 1
    assert tracker.update([detection(12, 0.3)])[0].track_id == 1
    fresh = ByteTracker("cam-b", track_threshold=0.5, low_threshold=0.1)
    assert fresh.update([detection(12, 0.3)]) == []
