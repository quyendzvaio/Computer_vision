"""Edge device: capture USB cameras, JPEG-encode, send via ZeroMQ PUB.

JPEG compression ~10-20× smaller than raw BGR. FPS strictly capped
to configured rate. ZMQ PUB with HWM=2 prevents queue buildup.
"""
import logging
import time

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
    fps = cfg.get("fps", 30)

    # USB cameras often ignore CAP_PROP — read first frame to force init, then try again
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (actual_w, actual_h) != (w, h):
        log.warning("Camera %s: wanted %dx%d, got %dx%d — forcing resize", cfg["id"], w, h, actual_w, actual_h)

    log.info("Camera %s opened: %dx%d (cfg %dx%d)", cfg["id"], actual_w, actual_h, w, h)
    return cap


def sender_loop():
    cfg = load_config()
    ctx = zmq.Context()
    cameras = []

    gpu_host = cfg.get("gpu_host", "127.0.0.1")
    jpeg_quality = cfg.get("jpeg_quality", 60)

    for cam_cfg in cfg.get("cameras", []):
        cap = setup_camera(cam_cfg)
        if cap is None:
            continue
        pub = ctx.socket(zmq.PUB)
        pub.setsockopt(zmq.SNDHWM, 2)
        pub.connect(f"tcp://{gpu_host}:{cam_cfg['zmq_port']}")
        cameras.append((cam_cfg["id"], cap, pub, cam_cfg.get("fps", 30), jpeg_quality))
        log.info("Camera %s → tcp://%s:%d @ %dfps", cam_cfg["id"], gpu_host, cam_cfg["zmq_port"], cam_cfg.get("fps", 30))

    if not cameras:
        log.error("No cameras available. Exiting.")
        return

    log.info("Edge sender running with %d camera(s)", len(cameras))

    last_time = {cam_id: 0.0 for cam_id, _, _, _, _ in cameras}

    while True:
        for cam_id, cap, pub, fps, quality in cameras:
            # FPS capping
            now = time.monotonic()
            interval = 1.0 / fps
            if now - last_time[cam_id] < interval:
                continue
            last_time[cam_id] = now

            ret, frame = cap.read()
            if not ret:
                log.warning("Camera %s: frame read failed", cam_id)
                continue

            # JPEG encode — shrink bandwidth 10-20×
            ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ret:
                log.warning("Camera %s: JPEG encode failed", cam_id)
                continue

            pub.send(buf.tobytes())


if __name__ == "__main__":
    try:
        sender_loop()
    except KeyboardInterrupt:
        log.info("Edge sender stopped by user")
