"""Deterministic exponential reconnect policy."""

from dataclasses import dataclass

from shared.schemas import ReconnectConfig


@dataclass(slots=True)
class ReconnectPolicy:
    config: ReconnectConfig
    attempts: int = 0

    def next_delay_seconds(self) -> float:
        delay_ms = min(
            self.config.maximum_delay_ms,
            self.config.initial_delay_ms * (self.config.multiplier ** self.attempts),
        )
        self.attempts += 1
        return delay_ms / 1_000.0

    def reset(self) -> None:
        self.attempts = 0
