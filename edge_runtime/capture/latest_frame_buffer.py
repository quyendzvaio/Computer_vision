"""A process-safe, bounded latest-frame buffer with explicit freshness metadata."""

import ctypes
import multiprocessing as mp
import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FramePacket:
    camera_id: str
    frame_id: int
    captured_at_unix_ns: int
    captured_at_monotonic_ns: int
    frame: np.ndarray

    @property
    def age_ms(self) -> float:
        return max(0.0, (time.monotonic_ns() - self.captured_at_monotonic_ns) / 1_000_000)


class LatestFrameBuffer:
    """One shared frame slot; a new frame overwrites unconsumed stale data."""

    def __init__(
        self,
        camera_id: str,
        max_width: int,
        max_height: int,
        context: mp.context.BaseContext | None = None,
    ) -> None:
        if max_width <= 0 or max_height <= 0:
            raise ValueError("buffer dimensions must be positive")
        ctx = context or mp.get_context()
        self.camera_id = camera_id
        self.max_width = max_width
        self.max_height = max_height
        self._capacity = max_width * max_height * 3
        self._data = ctx.RawArray(ctypes.c_ubyte, self._capacity)
        self._lock = ctx.Lock()
        self._sequence = ctx.Value("Q", 0, lock=False)
        self._consumed_sequence = ctx.Value("Q", 0, lock=False)
        self._unix_ns = ctx.Value("Q", 0, lock=False)
        self._monotonic_ns = ctx.Value("Q", 0, lock=False)
        self._height = ctx.Value("i", 0, lock=False)
        self._width = ctx.Value("i", 0, lock=False)
        self._dropped = ctx.Value("Q", 0, lock=False)
        self._stale = ctx.Value("Q", 0, lock=False)

    @property
    def dropped_frames(self) -> int:
        with self._lock:
            return int(self._dropped.value)

    @property
    def sequence(self) -> int:
        with self._lock:
            return int(self._sequence.value)

    @property
    def stale_frames(self) -> int:
        with self._lock:
            return int(self._stale.value)

    def publish(
        self,
        frame: np.ndarray,
        *,
        captured_at_unix_ns: int | None = None,
        captured_at_monotonic_ns: int | None = None,
    ) -> int:
        """Atomically replace the slot and return the assigned frame sequence."""

        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a uint8 HxWx3 BGR array")
        height, width = frame.shape[:2]
        if width > self.max_width or height > self.max_height:
            raise ValueError(
                f"frame {width}x{height} exceeds buffer "
                f"{self.max_width}x{self.max_height}"
            )
        contiguous = np.ascontiguousarray(frame)
        size = contiguous.nbytes
        unix_ns = captured_at_unix_ns or time.time_ns()
        monotonic_ns = captured_at_monotonic_ns or time.monotonic_ns()
        with self._lock:
            if self._sequence.value > self._consumed_sequence.value:
                self._dropped.value += 1
            np.frombuffer(self._data, dtype=np.uint8, count=size)[:] = contiguous.reshape(-1)
            self._height.value = height
            self._width.value = width
            self._unix_ns.value = unix_ns
            self._monotonic_ns.value = monotonic_ns
            self._sequence.value += 1
            return int(self._sequence.value)

    def read_latest(
        self,
        *,
        after_sequence: int = 0,
        max_age_ms: float | None = None,
    ) -> FramePacket | None:
        """Copy the newest frame or return None when unchanged/stale."""

        with self._lock:
            sequence = int(self._sequence.value)
            if sequence == 0 or sequence <= after_sequence:
                return None
            age_ms = (time.monotonic_ns() - self._monotonic_ns.value) / 1_000_000
            if max_age_ms is not None and age_ms > max_age_ms:
                self._stale.value += 1
                self._consumed_sequence.value = sequence
                return None
            height = int(self._height.value)
            width = int(self._width.value)
            size = height * width * 3
            frame = np.frombuffer(self._data, dtype=np.uint8, count=size).copy()
            frame = frame.reshape(height, width, 3)
            self._consumed_sequence.value = sequence
            return FramePacket(
                camera_id=self.camera_id,
                frame_id=sequence,
                captured_at_unix_ns=int(self._unix_ns.value),
                captured_at_monotonic_ns=int(self._monotonic_ns.value),
                frame=frame,
            )
