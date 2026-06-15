"""Edge device: capture USB cameras and send raw BGR frames via ZeroMQ PUB.

Connects to GPU machine (stable endpoint). ZMQ PUB with HWM=2 prevents
queue buildup when GPU is slow. Sends raw BGR bytes — no JPEG encode.
"""
import logging

import cv2
import yaml
import zmq

logging.basicConfig(level=logging.INFO, format="[Edge] %(message)s")
log = logging.getLogger("edge")


def load_config(path: str = "edge/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_camera(cfg: dict):
    """Open USB camera and set resolution. Returns VideoCapture or None."""
    path = cfg.get("device_path", "")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        log.error("Cannot open camera: %s", path)
        return None
    w, h = cfg.get("resolution", [640, 480])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, cfg.get("fps", 30))
    log.info("Camera %s opened: %dx%d", cfg["id"], int(cap.get(3)), int(cap.get(4)))
    return cap


def sender_loop():
    cfg = load_config()
    ctx = zmq.Context()
    cameras = []

    gpu_host = cfg.get("gpu_host", "127.0.0.1")

    for cam_cfg in cfg.get("cameras", []):
        cap = setup_camera(cam_cfg)
        if cap is None:
            continue
        pub = ctx.socket(zmq.PUB)
        pub.setsockopt(zmq.SNDHWM, 2)
        pub.connect(f"tcp://{gpu_host}:{cam_cfg['zmq_port']}")
        cameras.append((cam_cfg["id"], cap, pub))
        log.info("Camera %s → tcp://%s:%d", cam_cfg["id"], gpu_host, cam_cfg["zmq_port"])

    if not cameras:
        log.error("No cameras available. Exiting.")
        return

    log.info("Edge sender running with %d camera(s)", len(cameras))

    while True:
        for cam_id, cap, pub in cameras:
            ret, frame = cap.read()
            if not ret:
                log.warning("Camera %s: frame read failed", cam_id)
                continue
            pub.send(frame.tobytes())


if __name__ == "__main__":
    try:
        sender_loop()
    except KeyboardInterrupt:
        log.info("Edge sender stopped by user")
