"""Tests for alert.cooldown — CooldownManager per-(camera, violation_type) rate limiting."""
from datetime import datetime, timedelta


def test_should_alert_first_time():
    """First alert for a (camera, type) should always pass."""
    from alert.cooldown import CooldownManager

    cm = CooldownManager(cooldown_seconds=5)
    assert cm.should_alert("cam-01", "NO_HELMET") is True


def test_should_alert_within_cooldown():
    """Second alert within cooldown window should be blocked."""
    from alert.cooldown import CooldownManager

    cm = CooldownManager(cooldown_seconds=5)

    # First alert passes
    cm.should_alert("cam-01", "NO_HELMET")

    # Immediate second alert should be blocked
    assert cm.should_alert("cam-01", "NO_HELMET") is False


def test_should_alert_after_cooldown():
    """Alert should pass again after cooldown period expires."""
    from alert.cooldown import CooldownManager

    cm = CooldownManager(cooldown_seconds=1)

    # First alert passes
    cm.should_alert("cam-01", "FALL")

    # Manually age the last-alert time by 2 seconds
    cm._last_alert[("cam-01", "FALL")] = datetime.now() - timedelta(seconds=2)

    assert cm.should_alert("cam-01", "FALL") is True


def test_different_types_independent_cooldown():
    """Different violation types on same camera have independent cooldowns."""
    from alert.cooldown import CooldownManager

    cm = CooldownManager(cooldown_seconds=5)

    assert cm.should_alert("cam-01", "NO_HELMET") is True
    # Different type — should not be blocked
    assert cm.should_alert("cam-01", "NO_VEST") is True
    # Same type — blocked
    assert cm.should_alert("cam-01", "NO_HELMET") is False


def test_different_cameras_independent_cooldown():
    """Different cameras have independent cooldowns."""
    from alert.cooldown import CooldownManager

    cm = CooldownManager(cooldown_seconds=5)

    assert cm.should_alert("cam-01", "NO_HELMET") is True
    assert cm.should_alert("cam-02", "NO_HELMET") is True
    # cam-01 same type — blocked
    assert cm.should_alert("cam-01", "NO_HELMET") is False


def test_reset():
    """reset() should clear all cooldown state."""
    from alert.cooldown import CooldownManager

    cm = CooldownManager(cooldown_seconds=10)
    cm.should_alert("cam-01", "FALL")
    assert cm.should_alert("cam-01", "FALL") is False

    cm.reset()
    assert cm.should_alert("cam-01", "FALL") is True
