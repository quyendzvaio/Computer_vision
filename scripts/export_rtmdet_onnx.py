#!/usr/bin/env python3
"""Reproducible wrapper around OpenMMLab's official MMDeploy export command."""

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import onnxruntime as ort


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmdeploy-root", type=Path, required=True)
    parser.add_argument("--mmpose-root", type=Path, required=True)
    parser.add_argument("--demo-image", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    checkpoint = (
        args.project_root
        / "models/source/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth"
    )
    required = [
        args.mmdeploy_root / "tools/deploy.py",
        args.mmdeploy_root / "configs/mmdet/detection/detection_onnxruntime_static.py",
        args.mmpose_root / "projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py",
        checkpoint,
        args.demo_image,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing export inputs: {missing}")
    work_dir = args.project_root / "models/detector/mmdeploy-work"
    command = [
        sys.executable,
        str(required[0]),
        str(required[1]),
        str(required[2]),
        str(checkpoint),
        str(args.demo_image),
        "--work-dir",
        str(work_dir),
        "--device",
        "cpu",
        "--dump-info",
    ]
    subprocess.run(command, cwd=args.mmdeploy_root, check=True)
    exported = work_dir / "end2end.onnx"
    session = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
    output_names = [output.name for output in session.get_outputs()]
    if output_names != ["dets", "labels"]:
        raise RuntimeError(f"unexpected RTMDet output contract: {output_names}")
    destination = args.project_root / "models/detector/rtmdet_nano_320_person.onnx"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, destination)
    checksum = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(".onnx.sha256").write_text(checksum + "\n", encoding="ascii")
    print(f"verified {destination} sha256={checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
