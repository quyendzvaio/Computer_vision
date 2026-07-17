#!/usr/bin/env python3
"""Download only pinned OpenMMLab artifacts and verify SHA-256 before use."""

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import yaml


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def download(url: str, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and digest(destination) == expected_sha256:
        print(f"verified existing artifact: {destination}")
        return
    with tempfile.NamedTemporaryFile(delete=False, dir=destination.parent) as temporary:
        temporary_path = Path(temporary.name)
    try:
        print(f"downloading {url}")
        with (
            urllib.request.urlopen(url, timeout=120) as response,
            temporary_path.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
        actual = digest(temporary_path)
        if actual != expected_sha256:
            raise RuntimeError(f"checksum mismatch: expected {expected_sha256}, got {actual}")
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def fetch_pose(model: dict, root: Path) -> None:
    with tempfile.TemporaryDirectory() as directory:
        archive = Path(directory) / "model.zip"
        download(model["archive_url"], archive, model["archive_sha256"])
        with zipfile.ZipFile(archive) as bundle:
            matches = [
                name
                for name in bundle.namelist()
                if name.endswith(model["archive_member_suffix"])
            ]
            if len(matches) != 1:
                raise RuntimeError(f"expected one ONNX member, got {matches}")
            destination = root / model["runtime_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(matches[0]) as source, tempfile.NamedTemporaryFile(
                delete=False, dir=destination.parent
            ) as temporary:
                shutil.copyfileobj(source, temporary)
                temporary_path = Path(temporary.name)
            try:
                actual = digest(temporary_path)
                if actual != model["runtime_sha256"]:
                    raise RuntimeError(f"runtime ONNX checksum mismatch: {actual}")
                temporary_path.replace(destination)
            finally:
                temporary_path.unlink(missing_ok=True)
            print(f"installed and verified: {destination}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("human-pose", "person-detector", "all"),
        default="all",
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    manifest = yaml.safe_load((args.root / "models/manifest.yaml").read_text(encoding="utf-8"))
    models = manifest["models"]
    if args.model in ("human-pose", "all"):
        fetch_pose(models["human-pose"], args.root)
    if args.model in ("person-detector", "all"):
        detector = models["person-detector"]
        download(
            detector["checkpoint_url"],
            args.root / detector["checkpoint_path"],
            detector["checkpoint_sha256"],
        )
        print("RTMDet checkpoint verified; export it with scripts/export_rtmdet_onnx.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
