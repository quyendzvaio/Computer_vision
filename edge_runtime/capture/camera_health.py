"""Process-safe camera health counters."""

import multiprocessing as mp
import time
from dataclasses import dataclass

from shared.enums import CameraState

_STATE_TO_INT = {state: index for index, state in enumerate(CameraState)}
_INT_TO_STATE = {value: key for key, value in _STATE_TO_INT.items()}


@dataclass(frozen=True, slots=True)
class CameraHealthSnapshot:
    state: CameraState
    captured_frames: int
    capture_failures: int
    reconnect_count: int
    resolution_mismatch_count: int
    dropped_frames: int
    capture_fps: float
    last_success_monotonic_ns: int
    last_success_unix_ns: int
    width: int
    height: int


class CameraHealth:
    """Small shared-memory state object passed to one capture process."""

    def __init__(self, context: mp.context.BaseContext | None = None) -> None:
        ctx = context or mp.get_context()
        self._lock = ctx.Lock()
        self._state = ctx.Value("i", _STATE_TO_INT[CameraState.STOPPED], lock=False)
        self._captured = ctx.Value("Q", 0, lock=False)
        self._failures = ctx.Value("Q", 0, lock=False)
        self._reconnects = ctx.Value("Q", 0, lock=False)
        self._resolution_mismatches = ctx.Value("Q", 0, lock=False)
        self._dropped = ctx.Value("Q", 0, lock=False)
        self._fps = ctx.Value("d", 0.0, lock=False)
        self._last_mono = ctx.Value("Q", 0, lock=False)
        self._last_unix = ctx.Value("Q", 0, lock=False)
        self._width = ctx.Value("i", 0, lock=False)
        self._height = ctx.Value("i", 0, lock=False)

    def set_state(self, state: CameraState) -> None:
        with self._lock:
            self._state.value = _STATE_TO_INT[state]

    def record_success(self, width: int, height: int, fps: float) -> None:
        with self._lock:
            self._captured.value += 1
            self._fps.value = fps
            self._last_mono.value = time.monotonic_ns()
            self._last_unix.value = time.time_ns()
            self._width.value = width
            self._height.value = height
            self._state.value = _STATE_TO_INT[CameraState.RUNNING]

    def record_failure(self) -> None:
        with self._lock:
            self._failures.value += 1
            self._state.value = _STATE_TO_INT[CameraState.DEGRADED]

    def record_reconnect(self) -> None:
        with self._lock:
            self._reconnects.value += 1
            self._state.value = _STATE_TO_INT[CameraState.STARTING]

    def record_resolution_mismatch(self) -> None:
        with self._lock:
            self._resolution_mismatches.value += 1
            self._state.value = _STATE_TO_INT[CameraState.DEGRADED]

    def set_dropped_frames(self, count: int) -> None:
        with self._lock:
            self._dropped.value = count

    def snapshot(self) -> CameraHealthSnapshot:
        with self._lock:
            return CameraHealthSnapshot(
                state=_INT_TO_STATE[self._state.value],
                captured_frames=self._captured.value,
                capture_failures=self._failures.value,
                reconnect_count=self._reconnects.value,
                resolution_mismatch_count=self._resolution_mismatches.value,
                dropped_frames=self._dropped.value,
                capture_fps=self._fps.value,
                last_success_monotonic_ns=self._last_mono.value,
                last_success_unix_ns=self._last_unix.value,
                width=self._width.value,
                height=self._height.value,
            )
