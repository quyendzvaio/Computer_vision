"""Per-(camera, violation_type) cooldown manager to prevent alert spam."""
from datetime import datetime
from typing import Dict, Tuple


class CooldownManager:
    """Rate-limits alerts so the same violation from the same camera
    only fires once per cooldown period (default 5 seconds)."""

    def __init__(self, cooldown_seconds: float = 5.0):
        self._cooldown = cooldown_seconds
        self._last_alert: Dict[Tuple[str, str], datetime] = {}

    def should_alert(self, camera_id: str, violation_type: str) -> bool:
        """Returns True if this alert should fire.
        Returns False if it's within the cooldown window for this (camera, type)."""
        key = (camera_id, violation_type)
        now = datetime.now()

        if key in self._last_alert:
            elapsed = (now - self._last_alert[key]).total_seconds()
            if elapsed < self._cooldown:
                return False

        self._last_alert[key] = now
        return True

    def reset(self) -> None:
        """Clear all cooldown state (useful for testing or config changes)."""
        self._last_alert.clear()

    def reset_for_camera(self, camera_id: str) -> None:
        """Clear cooldown for all violation types on one camera."""
        keys_to_remove = [k for k in self._last_alert if k[0] == camera_id]
        for k in keys_to_remove:
            del self._last_alert[k]
