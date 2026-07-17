"""MMDeploy RTMDet person detector preprocessing and end-to-end decoding."""

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from edge_runtime.capture.latest_frame_buffer import FramePacket
from edge_runtime.inference.interfaces import ModelBackend
from shared.schemas import BoundingBox, Detection, ModelConfig


@dataclass(frozen=True, slots=True)
class LetterboxTransform:
    scale: float
    original_width: int
    original_height: int


class RTMDetPipeline:
    """Implements the official 320x320 RTMDet-nano person contract.

    The MMDetection test pipeline resizes with preserved aspect ratio and pads
    the bottom/right edge. The MMDeploy end-to-end graph owns NMS and returns
    ``dets`` (xyxy + score) and ``labels``.
    """

    _MEAN = np.asarray([103.53, 116.28, 123.675], dtype=np.float32)
    _STD = np.asarray([57.375, 57.12, 58.395], dtype=np.float32)

    def __init__(self, backend: ModelBackend, configuration: ModelConfig) -> None:
        if configuration.output_format != "mmdeploy-dets-labels":
            raise ValueError("RTMDet requires output_format=mmdeploy-dets-labels")
        self.backend = backend
        self.configuration = configuration
        self.input_height, self.input_width = configuration.input_shape

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, LetterboxTransform]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("RTMDet input must be a BGR HxWx3 image")
        original_height, original_width = image.shape[:2]
        scale = min(
            self.input_width / original_width,
            self.input_height / original_height,
        )
        resized_width = max(1, min(self.input_width, round(original_width * scale)))
        resized_height = max(1, min(self.input_height, round(original_height * scale)))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        canvas[:resized_height, :resized_width] = resized
        tensor = (canvas.astype(np.float32) - self._MEAN) / self._STD
        tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1))
        return tensor, LetterboxTransform(scale, original_width, original_height)

    def postprocess(
        self,
        outputs: Sequence[np.ndarray],
        transform: LetterboxTransform,
    ) -> list[Detection]:
        if len(outputs) != 2:
            raise ValueError(f"expected RTMDet dets/labels outputs, got {len(outputs)}")
        dets, labels = np.asarray(outputs[0]), np.asarray(outputs[1])
        if dets.ndim == 3:
            dets = dets[0]
        if labels.ndim == 2:
            labels = labels[0]
        if dets.ndim != 2 or dets.shape[1] != 5 or labels.ndim != 1:
            raise ValueError(f"invalid RTMDet output shapes: {dets.shape}, {labels.shape}")
        if len(dets) != len(labels):
            raise ValueError("RTMDet dets/labels length mismatch")

        results: list[Detection] = []
        for row, label in zip(dets, labels):
            score = float(row[4])
            if int(label) != 0 or score < self.configuration.score_threshold:
                continue
            x1, y1, x2, y2 = (float(value) / transform.scale for value in row[:4])
            x1 = min(max(x1, 0.0), float(transform.original_width))
            x2 = min(max(x2, 0.0), float(transform.original_width))
            y1 = min(max(y1, 0.0), float(transform.original_height))
            y2 = min(max(y2, 0.0), float(transform.original_height))
            if x2 <= x1 or y2 <= y1:
                continue
            results.append(Detection(BoundingBox(x1, y1, x2, y2), score, 0, "person"))
        return results

    def infer_frames(self, frames: Sequence[FramePacket]) -> Sequence[list[Detection]]:
        if not frames:
            return []
        prepared = [self.preprocess(packet.frame) for packet in frames]
        input_shape = getattr(self.backend, "input_shape", ())
        if len(frames) > 1 and input_shape and input_shape[0] == 1:
            return [
                self.postprocess(self.backend.infer(item[0][None]), item[1])
                for item in prepared
            ]
        tensor = np.stack([item[0] for item in prepared])
        outputs = self.backend.infer(tensor)
        if not outputs or outputs[0].shape[0] != len(frames):
            raise ValueError("RTMDet batch mapping mismatch")
        return [
            self.postprocess([output[index : index + 1] for output in outputs], transform)
            for index, (_, transform) in enumerate(prepared)
        ]
