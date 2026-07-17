"""Stable USB device discovery without assuming /dev/video enumeration order."""

import glob
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CameraDevice:
    stable_path: str
    resolved_path: str


class DeviceRegistry:
    @staticmethod
    def discover_linux() -> list[CameraDevice]:
        devices: list[CameraDevice] = []
        for stable in sorted(glob.glob("/dev/v4l/by-id/*")):
            devices.append(
                CameraDevice(stable_path=stable, resolved_path=os.path.realpath(stable))
            )
        return devices

    @staticmethod
    def resolve(device_path: str | int) -> str | int:
        """Validate stable Linux paths while retaining numeric dev indices."""

        if isinstance(device_path, int):
            return device_path
        path = Path(device_path)
        if device_path.startswith("/dev/v4l/by-id/") and path.is_symlink():
            return str(path)
        return device_path
