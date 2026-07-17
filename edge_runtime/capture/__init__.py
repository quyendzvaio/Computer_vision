"""Isolated camera capture processes and single-slot frame transport."""

from edge_runtime.capture.latest_frame_buffer import FramePacket, LatestFrameBuffer
from edge_runtime.capture.worker import CaptureWorker

__all__ = ["CaptureWorker", "FramePacket", "LatestFrameBuffer"]
