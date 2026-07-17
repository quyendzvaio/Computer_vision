#!/usr/bin/env python3
"""Discover stable Linux camera paths and probe their current video mode."""

import argparse
import json
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edge_runtime.capture.device_registry import DeviceRegistry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true", help="open each camera and read one frame")
    args = parser.parse_args()

    if platform.system() != "Linux":
        print(
            json.dumps(
                {
                    "platform": platform.system(),
                    "devices": [],
                    "note": "use camera indices",
                }
            )
        )
        return 0

    devices = DeviceRegistry.discover_linux()
    output = []
    for device in devices:
        item: dict[str, object] = {
            "stable_path": device.stable_path,
            "resolved_path": device.resolved_path,
        }
        if args.probe:
            import cv2

            capture = cv2.VideoCapture(device.stable_path)
            try:
                ok, frame = capture.read()
                item["opened"] = capture.isOpened()
                item["frame_read"] = bool(ok and frame is not None)
                if ok and frame is not None:
                    item["resolution"] = [int(frame.shape[1]), int(frame.shape[0])]
            finally:
                capture.release()
        output.append(item)
    print(json.dumps({"platform": "Linux", "devices": output}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
