import multiprocessing as mp
import time

import numpy as np

from edge_runtime.capture.latest_frame_buffer import LatestFrameBuffer
from edge_runtime.supervisor import EdgeSupervisor
from shared.schemas import EdgeConfiguration


def _publish_frame(buffer: LatestFrameBuffer) -> None:
    buffer.publish(np.full((8, 8, 3), 5, dtype=np.uint8))


def _hang() -> None:
    while True:
        time.sleep(1)


def test_separate_process_failure_does_not_block_peer_buffer() -> None:
    context = mp.get_context("spawn")
    buffer = LatestFrameBuffer("cam-ok", 8, 8, context)
    hung = context.Process(target=_hang)
    writer = context.Process(target=_publish_frame, args=(buffer,))
    hung.start()
    writer.start()
    writer.join(timeout=5)
    try:
        assert writer.exitcode == 0
        packet = buffer.read_latest(max_age_ms=2_000)
        assert packet is not None
        assert int(packet.frame.mean()) == 5
        assert hung.is_alive()
    finally:
        hung.terminate()
        hung.join(timeout=2)


def test_supervisor_does_not_hardcode_camera_count() -> None:
    configuration = EdgeConfiguration(
        device_id="edge-01",
        cameras=[
            {"camera_id": "a", "device_path": 0, "resolution": [640, 480]},
            {"camera_id": "b", "device_path": 1, "resolution": [640, 480]},
            {
                "camera_id": "c",
                "device_path": 2,
                "resolution": [640, 480],
                "enabled": False,
            },
        ],
    )
    supervisor = EdgeSupervisor(configuration, context=mp.get_context("spawn"))
    assert set(supervisor.cameras) == {"a", "b"}
