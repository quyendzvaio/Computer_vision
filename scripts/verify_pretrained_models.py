#!/usr/bin/env python3
"""Offline structural verification; never changes edge provider policy."""

import argparse
import hashlib
from pathlib import Path

import onnxruntime as ort


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--require-detector", action="store_true")
    args = parser.parse_args()
    pose = args.root / "models/rtmpose/rtmpose_s_body7_256x192.onnx"
    expected_pose = "9aeb635b83f86aea45cf45d85798f7eba1a162de8e0d721c44e54fe5eebaf47d"
    if not pose.is_file() or hashlib.sha256(pose.read_bytes()).hexdigest() != expected_pose:
        raise SystemExit("RTMPose artifact missing or checksum mismatch")
    session = ort.InferenceSession(str(pose), providers=["CPUExecutionProvider"])
    inputs = [(value.name, value.shape) for value in session.get_inputs()]
    outputs = [(value.name, value.shape) for value in session.get_outputs()]
    if inputs != [("input", ["batch", 3, 256, 192])]:
        raise SystemExit(f"unexpected RTMPose input: {inputs}")
    if [name for name, _ in outputs] != ["simcc_x", "simcc_y"]:
        raise SystemExit(f"unexpected RTMPose outputs: {outputs}")
    print(f"RTMPose verified: inputs={inputs}, outputs={outputs}")
    detector = args.root / "models/detector/rtmdet_nano_320_person.onnx"
    if detector.is_file():
        detector_session = ort.InferenceSession(str(detector), providers=["CPUExecutionProvider"])
        names = [value.name for value in detector_session.get_outputs()]
        if names != ["dets", "labels"]:
            raise SystemExit(f"unexpected RTMDet outputs: {names}")
        print(f"RTMDet verified: outputs={names}")
    elif args.require_detector:
        raise SystemExit("RTMDet ONNX has not been exported")
    else:
        print("RTMDet ONNX pending reproducible MMDeploy export")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
