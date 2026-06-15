"""Test cooldown manager and alert manager."""
import sqlite3
import time
import pytest
from gpu.alert_manager import CooldownManager, AlertManager


def test_cooldown_allows_first():
    cd = CooldownManager(default_cooldown=1.0)
    assert cd.can_alert(("cam1", 0, "PERSON_IN_ZONE")) is True


def test_cooldown_blocks_immediate_repeat():
    cd = CooldownManager(default_cooldown=5.0)
    assert cd.can_alert(("cam1", 0, "PERSON_IN_ZONE")) is True
    assert cd.can_alert(("cam1", 0, "PERSON_IN_ZONE")) is False


def test_cooldown_expires():
    cd = CooldownManager(default_cooldown=0.1)
    assert cd.can_alert(("cam1", 0, "PERSON_IN_ZONE")) is True
    time.sleep(0.15)
    assert cd.can_alert(("cam1", 0, "PERSON_IN_ZONE")) is True


def test_cooldown_different_key_not_blocked():
    cd = CooldownManager(default_cooldown=5.0)
    assert cd.can_alert(("cam1", 0, "NO_HELMET")) is True
    assert cd.can_alert(("cam1", 1, "NO_HELMET")) is True


@pytest.fixture
def alert_manager():
    from gpu.database import init_db
    conn = init_db()
    alerts = []
    def callback(vid, cam, vtype, zone, pidx, timestamp):
        alerts.append((vid, vtype, zone))
    am = AlertManager(conn, on_alert=callback, cooldown=30.0)
    return am, conn, alerts


def test_alert_manager_zone(alert_manager):
    am, conn, alerts = alert_manager
    fired = am.process_zone_alerts("cam1", [
        {"type": "PERSON_IN_ZONE", "zone_name": "Zone A", "person_idx": 0},
    ], force=True)
    assert len(fired) == 1
    assert len(alerts) == 1
    assert alerts[0][1] == "PERSON_IN_ZONE"


def test_alert_manager_cooldown(alert_manager):
    am, conn, alerts = alert_manager
    fired1 = am.process_zone_alerts("cam1", [
        {"type": "PERSON_IN_ZONE", "zone_name": "Zone A", "person_idx": 0},
    ], force=False)
    assert len(fired1) == 1
    fired2 = am.process_zone_alerts("cam1", [
        {"type": "PERSON_IN_ZONE", "zone_name": "Zone A", "person_idx": 0},
    ], force=False)
    assert len(fired2) == 0
