"""Dependency-free Prometheus text metrics for the edge foundation."""

import threading


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._gauges: dict[str, float] = {}
        self._counters: dict[str, int] = {}

    def set_gauge(self, name: str, value: float) -> None:
        self._validate_name(name)
        with self._lock:
            self._gauges[name] = float(value)

    def increment(self, name: str, value: int = 1) -> None:
        self._validate_name(name)
        if value < 0:
            raise ValueError("counter increments must be non-negative")
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def render_prometheus(self) -> str:
        with self._lock:
            gauges = (
                f"# TYPE {name} gauge\n{name} {value}"
                for name, value in sorted(self._gauges.items())
            )
            counters = (
                f"# TYPE {name} counter\n{name} {value}"
                for name, value in sorted(self._counters.items())
            )
            lines = [*gauges, *counters]
        return "\n".join(lines) + "\n"

    @staticmethod
    def _validate_name(name: str) -> None:
        if not name or not all(character.isalnum() or character in "_:" for character in name):
            raise ValueError(f"invalid Prometheus metric name: {name!r}")
