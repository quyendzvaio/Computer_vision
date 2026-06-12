import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from shared.models import Violation, BBox


def make_violation(camera_id="cam-01", vtype="NO_HELMET"):
    return Violation(
        camera_id=camera_id,
        type=vtype,
        severity="HIGH",
        bbox=BBox(100, 100, 200, 300),
    )


def test_dispatcher_inserts_to_db():
    """Dispatcher should call db.insert_violation for each violation."""
    from alert.dispatcher import Dispatcher

    mock_db = MagicMock()
    mock_db.insert_violation.return_value = 42

    dispatcher = Dispatcher(db=mock_db, ws_manager=None)

    v = make_violation()
    dispatcher.dispatch(v, frame_bgr=None)

    mock_db.insert_violation.assert_called_once()
    args = mock_db.insert_violation.call_args[0]
    assert args[0] == "cam-01"
    assert args[1] == "NO_HELMET"
    assert args[2] == "HIGH"


def test_dispatcher_saves_thumbnail(temp_dir):
    """Dispatcher should save thumbnail JPEG to data/thumbnails/."""
    from alert.dispatcher import Dispatcher

    mock_db = MagicMock()
    mock_db.insert_violation.return_value = 42

    thumb_dir = temp_dir / "thumbnails"
    dispatcher = Dispatcher(
        db=mock_db,
        ws_manager=None,
        thumbnail_dir=str(thumb_dir),
    )

    # Create a fake BGR frame (100x100 blue image)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :] = (255, 0, 0)  # BGR blue

    v = make_violation()
    dispatcher.dispatch(v, frame_bgr=frame)

    # Check that thumbnail path was set
    assert v.thumbnail_path != ""
    assert Path(v.thumbnail_path).exists()
    # Check thumbnail path was passed to DB insert (4th positional arg = thumbnail_path)
    assert mock_db.insert_violation.call_args[0][4] == v.thumbnail_path


def test_dispatcher_broadcasts_to_websocket():
    """Dispatcher should send violation JSON to all WebSocket clients."""
    from alert.dispatcher import Dispatcher

    mock_db = MagicMock()
    mock_db.insert_violation.return_value = 42
    mock_ws = MagicMock()

    dispatcher = Dispatcher(db=mock_db, ws_manager=mock_ws)

    v = make_violation()
    dispatcher.dispatch(v, frame_bgr=None)

    mock_ws.broadcast.assert_called_once()
    broadcast_data = mock_ws.broadcast.call_args[0][0]
    parsed = json.loads(broadcast_data) if isinstance(broadcast_data, str) else broadcast_data
    assert parsed["type"] == "violation"
    assert parsed["violation"]["camera_id"] == "cam-01"
    assert parsed["violation"]["type"] == "NO_HELMET"


def test_dispatcher_no_websocket_does_not_crash():
    """Dispatcher should work fine without a WebSocket manager (no-op broadcast)."""
    from alert.dispatcher import Dispatcher

    mock_db = MagicMock()
    mock_db.insert_violation.return_value = 42

    dispatcher = Dispatcher(db=mock_db, ws_manager=None)

    v = make_violation()
    # Should not raise
    dispatcher.dispatch(v, frame_bgr=None)
    assert mock_db.insert_violation.called
