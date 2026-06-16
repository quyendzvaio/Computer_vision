"""Shared memory IPC between capture (P1) and inference (P2) processes.

Double-buffer protocol — P1 writes to FREE buffer, P2 reads from FILLED buffer,
sets LOCKED while holding result. P1 reads EXACT frame back for overlay,
then marks FREE again.

State machine per buffer: FREE (0) → FILLED (1) → LOCKED (2) → FREE (0)

Ctrl segment (64 bytes):
  [0:1]   state_a      — uint8  0=FREE 1=FILLED 2=LOCKED
  [1:2]   state_b      — uint8
  [2:4]   pad
  [4:8]   height_a     — int32
  [8:12]  width_a      — int32
  [12:16] height_b     — int32
  [16:20] width_b      — int32
  [20:24] result_for_buf — int32  0/1 which buffer latest result is for (-1=none)
  [24:32] result_seq   — uint64  incremented by P2 after write_result
  [32:40] frame_counter — uint64  incremented by P1, tags each written frame
  [40:64] reserved
"""
import json
from enum import IntEnum
from multiprocessing import shared_memory
from typing import Optional

import numpy as np


class BufState(IntEnum):
    FREE = 0
    FILLED = 1
    LOCKED = 2


class FrameBuffer:
    """Double-buffer shared memory — 2 frame slots, lock protocol for perfect overlay."""

    CTRL_SIZE = 64
    RESULT_SIZE = 8192

    def __init__(self, camera_id: str, max_width: int = 1920, max_height: int = 1080):
        self.camera_id = camera_id
        self.max_width = max_width
        self.max_height = max_height
        self.frame_nbytes = max_width * max_height * 3
        self._last_result_seq = 0

        prefix = f"cv_{camera_id.replace('-', '_')}"

        def _create_or_attach(name, size):
            try:
                return shared_memory.SharedMemory(
                    name=name, create=True, size=size)
            except FileExistsError:
                # Attach existing (P2 after P1)
                return shared_memory.SharedMemory(name=name, create=False)
            except Exception:
                # Unlink stale and retry
                try:
                    shm = shared_memory.SharedMemory(name=name, create=False)
                    shm.close()
                    shm.unlink()
                except (FileNotFoundError, Exception):
                    pass
                return shared_memory.SharedMemory(
                    name=name, create=True, size=size)

        self._shm_frame_a = _create_or_attach(
            f"{prefix}_frame_a", self.frame_nbytes)
        self._shm_frame_b = _create_or_attach(
            f"{prefix}_frame_b", self.frame_nbytes)
        self._shm_result = _create_or_attach(
            f"{prefix}_result", self.RESULT_SIZE)
        self._shm_ctrl = _create_or_attach(
            f"{prefix}_ctrl", self.CTRL_SIZE)

        # Fresh init: both buffers FREE, no result
        # Only if P1 created fresh (check if frame_counter is 0)
        if self._get_u64(32) == 0:
            self._shm_ctrl.buf[0] = BufState.FREE
            self._shm_ctrl.buf[1] = BufState.FREE
            self._set_i32(20, -1)  # result_for_buf = -1 (none)
            self._set_u64(24, 0)   # result_seq = 0
            self._set_u64(32, 0)   # frame_counter = 0

    # ── low-level helpers ──────────────────────────────────────

    @property
    def _state_a(self) -> BufState:
        return BufState(self._shm_ctrl.buf[0])

    @_state_a.setter
    def _state_a(self, v: BufState):
        self._shm_ctrl.buf[0] = v

    @property
    def _state_b(self) -> BufState:
        return BufState(self._shm_ctrl.buf[1])

    @_state_b.setter
    def _state_b(self, v: BufState):
        self._shm_ctrl.buf[1] = v

    def _get_u64(self, offset: int) -> int:
        return int.from_bytes(
            self._shm_ctrl.buf[offset:offset + 8], 'little')

    def _set_u64(self, offset: int, value: int):
        self._shm_ctrl.buf[offset:offset + 8] = \
            value.to_bytes(8, 'little')

    def _get_i32(self, offset: int) -> int:
        return int.from_bytes(
            self._shm_ctrl.buf[offset:offset + 4], 'little', signed=True)

    def _set_i32(self, offset: int, value: int):
        self._shm_ctrl.buf[offset:offset + 4] = \
            value.to_bytes(4, 'little', signed=True)

    # ── Frame: P1 write → P2 read ─────────────────────────────

    def pick_free(self) -> Optional[int]:
        """P1: find a FREE buffer. Returns 0/1 or None if both locked."""
        if self._state_a == BufState.FREE:
            return 0
        if self._state_b == BufState.FREE:
            return 1
        return None

    def write_frame(self, buf_idx: int, frame: np.ndarray):
        """P1: write frame to buffer, THEN set state → FILLED.

        Data must be fully committed before signaling P2, even on TSO x86.
        """
        h, w = frame.shape[:2]
        if h > self.max_height or w > self.max_width:
            raise ValueError(
                f"Frame {w}x{h} exceeds {self.max_width}x{self.max_height}")

        # 1. Write dimensions for this buffer
        if buf_idx == 0:
            self._set_i32(4, h)
            self._set_i32(8, w)
        else:
            self._set_i32(12, h)
            self._set_i32(16, w)

        # 2. Write pixel data (must complete before state change)
        shm = self._shm_frame_a if buf_idx == 0 else self._shm_frame_b
        nbytes = h * w * 3
        shm.buf[:nbytes] = frame.tobytes()

        # 3. Signal P2 — state change is the "memory barrier"
        if buf_idx == 0:
            self._state_a = BufState.FILLED
        else:
            self._state_b = BufState.FILLED

    def find_filled(self) -> Optional[int]:
        """P2: find a FILLED buffer. Returns 0/1 or None."""
        if self._state_a == BufState.FILLED:
            return 0
        if self._state_b == BufState.FILLED:
            return 1
        return None

    def acquire(self, buf_idx: int):
        """P2: lock buffer → LOCKED (P1 won't overwrite)."""
        if buf_idx == 0:
            self._state_a = BufState.LOCKED
        else:
            self._state_b = BufState.LOCKED

    def read_frame(self, buf_idx: int) -> Optional[np.ndarray]:
        """P2 or P1: read frame back from buffer.

        P2 calls after acquire(). P1 calls when consuming result.
        """
        if buf_idx == 0:
            h = self._get_i32(4)
            w = self._get_i32(8)
            shm = self._shm_frame_a
        else:
            h = self._get_i32(12)
            w = self._get_i32(16)
            shm = self._shm_frame_b

        if h <= 0 or w <= 0:
            return None

        nbytes = h * w * 3
        arr = np.frombuffer(shm.buf[:nbytes], dtype=np.uint8).copy()
        return arr.reshape(h, w, 3)

    def release(self, buf_idx: int):
        """P1: mark buffer as FREE after consuming result + overlay."""
        if buf_idx == 0:
            self._state_a = BufState.FREE
        else:
            self._state_b = BufState.FREE

    # ── Result: P2 write, P1 read ──────────────────────────────

    def write_result(self, result_json: str, buf_idx: int):
        """P2: write detection result JSON + flag which buffer it's for."""
        encoded = result_json.encode('utf-8')
        n = min(len(encoded), self.RESULT_SIZE - 1)
        self._shm_result.buf[:n] = encoded[:n]
        self._shm_result.buf[n] = 0  # null terminator

        self._set_i32(20, buf_idx)
        self._set_u64(24, self._get_u64(24) + 1)  # result_seq++

    def read_result(self) -> Optional[dict]:
        """P1: read latest result. Returns dict or None if no new result."""
        result_seq = self._get_u64(24)
        if result_seq == self._last_result_seq:
            return None  # No new result

        self._last_result_seq = result_seq

        b = bytes(self._shm_result.buf[:self.RESULT_SIZE])
        null = b.find(b'\x00')
        if null <= 2:
            return None

        try:
            data = json.loads(b[:null].decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

        buf_idx = self._get_i32(20)
        data["_buffer_index"] = buf_idx
        return data

    # ── Result buffer check (no parsing) ───────────────────────

    def has_new_result(self) -> bool:
        return self._get_u64(24) != self._last_result_seq

    def result_for_buffer(self) -> int:
        """Which buffer the latest result is for (0/1)."""
        return self._get_i32(20)

    # ── Lifecycle ──────────────────────────────────────────────

    def close(self):
        self._shm_frame_a.close()
        self._shm_frame_b.close()
        self._shm_result.close()
        self._shm_ctrl.close()

    def unlink(self):
        for shm in [self._shm_frame_a, self._shm_frame_b,
                     self._shm_result, self._shm_ctrl]:
            try:
                shm.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
