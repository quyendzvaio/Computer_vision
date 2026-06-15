"""Edge device simulator for GPU machine testing.

Reads a video file and publishes raw BGR frames via ZMQ PUB,
simulating the real edge device. Use for integration testing
without needing edge hardware.
"""
import argparse
import time

import cv2
import zmq


def main():
    parser = argparse.ArgumentParser(description="Edge device simulator")
    parser.add_argument("video", help="Path to video file (.mp4)")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ port (default: 5555)")
    parser.add_argument("--fps", type=int, default=30, help="Target send FPS (default: 30)")
    parser.add_argument("--target-host", default="127.0.0.1", help="GPU machine host")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[Sim] Cannot open video: {args.video}")
        return

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 2)
    pub.connect(f"tcp://{args.target_host}:{args.port}")

    frame_interval = 1.0 / args.fps
    frame_count = 0

    print(f"[Sim] Publishing {args.video} -> tcp://{args.target_host}:{args.port} @ {args.fps}fps")

    try:
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            pub.send(frame.tobytes())
            frame_count += 1

            elapsed = time.perf_counter() - t0
            sleep = max(0, frame_interval - elapsed)
            time.sleep(sleep)

            if frame_count % 100 == 0:
                print(f"[Sim] Sent {frame_count} frames")

    except KeyboardInterrupt:
        print(f"[Sim] Stopped after {frame_count} frames")

    cap.release()


if __name__ == "__main__":
    main()
