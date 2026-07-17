"""Networking tests: ZMQ edge<->server, config loading, reconnection."""
import json
import os
import tempfile
import time
from multiprocessing import Process

import cv2
import numpy as np
import pytest
import zmq
import yaml

from edge.sender import load_config, setup_camera


# --- Config loading ---

def test_load_config_cpu_variant():
    cfg = load_config("edge/config.cpu.yaml")
    assert cfg["jpeg_quality"] == 60

def test_load_config_minimal():
    data = {"gpu_host": "10.0.0.1", "cameras": [
        {"id": "cam1", "device_path": "0", "zmq_port": 5555,
         "fps": 5, "resolution": [320, 240]},
    ]}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        path = f.name
    try:
        cfg = load_config(path)
        assert cfg["gpu_host"] == "10.0.0.1"
    finally:
        os.unlink(path)

def test_load_config_empty_cameras():
    data = {"gpu_host": "127.0.0.1", "cameras": []}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        path = f.name
    try:
        cfg = load_config(path)
        assert len(cfg["cameras"]) == 0
    finally:
        os.unlink(path)


# --- Camera setup ---

def test_setup_camera_invalid_device():
    cap = setup_camera({"id": "test", "device_path": "/dev/video9999",
                        "resolution": [320, 240], "fps": 5})
    assert cap is None


# --- ZMQ PUB/SUB ---

@pytest.fixture
def zmq_sub():
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVTIMEO, 2000)
    port = sub.bind_to_random_port("tcp://127.0.0.1")
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    yield port, sub
    sub.close()
    ctx.term()

def _jpeg_bytes(w=320, h=240):
    frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
    return buf.tobytes()


class TestZMQ:

    def test_jpeg_roundtrip(self, zmq_sub):
        port, sub = zmq_sub
        ctx = zmq.Context()
        pub = ctx.socket(zmq.PUB)
        pub.connect(f"tcp://127.0.0.1:{port}")
        time.sleep(0.05)
        data = _jpeg_bytes()
        pub.send(data)
        recv = sub.recv()
        assert recv == data
        frame = cv2.imdecode(np.frombuffer(recv, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert frame is not None and frame.shape == (240, 320, 3)
        pub.close()
        ctx.term()

    def test_hwm_drops_frames(self, zmq_sub):
        """RCVHWM=2 drops oldest frames, doesn't block publisher."""
        port, sub = zmq_sub
        sub.setsockopt(zmq.RCVHWM, 2)
        ctx = zmq.Context()
        pub = ctx.socket(zmq.PUB)
        pub.setsockopt(zmq.SNDHWM, 2)
        pub.connect(f"tcp://127.0.0.1:{port}")
        time.sleep(0.05)
        for _ in range(20):
            pub.send(_jpeg_bytes())
        count = 0
        while True:
            try:
                sub.recv()
                count += 1
            except zmq.Again:
                break
        assert 1 <= count < 20
        pub.close()
        ctx.term()

    def test_rcvtimeo_disconnected(self):
        """SUB with RCVTIMEO raises zmq.Again when no publisher."""
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.setsockopt(zmq.RCVTIMEO, 100)
        sub.bind_to_random_port("tcp://127.0.0.1")
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        with pytest.raises(zmq.Again):
            sub.recv()
        sub.close()
        ctx.term()


# --- Edge sender process (multiprocess) ---

def _sender_process(port):
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 2)
    pub.connect(f"tcp://127.0.0.1:{port}")
    for _ in range(10):
        pub.send(_jpeg_bytes())
        time.sleep(0.05)
    pub.close()
    ctx.term()

def test_edge_process_sends_jpeg():
    """Full edge sender process -> SUB receives valid JPEG."""
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVTIMEO, 5000)
    port = sub.bind_to_random_port("tcp://127.0.0.1")
    sub.setsockopt_string(zmq.SUBSCRIBE, "")

    p = Process(target=_sender_process, args=(port,))
    p.start()

    count = 0
    while count < 2:
        try:
            data = sub.recv()
            if cv2.imdecode(np.frombuffer(data, dtype=np.uint8),
                            cv2.IMREAD_COLOR) is not None:
                count += 1
        except zmq.Again:
            break

    p.terminate()
    p.join(timeout=2)
    sub.close()
    ctx.term()
    assert count >= 1


# --- JSON message format tests ---

def test_preview_msg_json():
    from edge.capture_loop import _preview_msg
    parsed = json.loads(_preview_msg("cam1", "abcd"))
    assert parsed == {"type": "preview", "camera_id": "cam1",
                      "frame_base64": "abcd"}

def test_violation_msg_json():
    from edge.capture_loop import _violation_msg
    from shared.models import Violation, BBox
    from datetime import datetime
    v = Violation(id=42, camera_id="cam1", type="NO_HELMET",
                  severity="HIGH", bbox=BBox(10, 20, 30, 40),
                  timestamp=datetime(2025, 1, 1, 12, 0, 0))
    parsed = json.loads(_violation_msg(v))
    assert parsed["type"] == "violation"
    assert parsed["violation"]["id"] == 42
    assert parsed["violation"]["type"] == "NO_HELMET"
    assert parsed["violation"]["bbox"] == [10, 20, 30, 40]
