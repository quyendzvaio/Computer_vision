"""RTMPose-s top-down affine preprocessing and SimCC postprocessing."""

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from edge_runtime.inference.interfaces import ModelBackend, PoseRequest
from shared.schemas import (
    COCO17_KEYPOINT_NAMES,
    BoundingBox,
    Keypoint,
    ModelConfig,
    PoseResult,
)


@dataclass(frozen=True, slots=True)
class PoseTransform:
    inverse_affine: np.ndarray


class RTMPosePipeline:
    _MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
    _STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)
    _BBOX_PADDING = 1.25
    _SIMCC_SPLIT_RATIO = 2.0

    def __init__(self, backend: ModelBackend, configuration: ModelConfig) -> None:
        if configuration.output_format != "simcc-coco17":
            raise ValueError("RTMPose requires output_format=simcc-coco17")
        self.backend = backend
        self.configuration = configuration
        self.input_height, self.input_width = configuration.input_shape

    def preprocess(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
    ) -> tuple[np.ndarray, PoseTransform]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("RTMPose input must be a BGR HxWx3 image")
        center = np.asarray([(bbox.x1 + bbox.x2) / 2, (bbox.y1 + bbox.y2) / 2])
        width = max(bbox.width, 1.0)
        height = max(bbox.height, 1.0)
        target_aspect = self.input_width / self.input_height
        if width > height * target_aspect:
            height = width / target_aspect
        else:
            width = height * target_aspect
        width *= self._BBOX_PADDING
        height *= self._BBOX_PADDING

        source = np.float32(
            [
                center,
                center + np.asarray([0.0, -height / 2]),
                center + np.asarray([width / 2, 0.0]),
            ]
        )
        destination = np.float32(
            [
                [self.input_width / 2, self.input_height / 2],
                [self.input_width / 2, 0.0],
                [self.input_width, self.input_height / 2],
            ]
        )
        affine = cv2.getAffineTransform(source, destination)
        crop = cv2.warpAffine(
            image,
            affine,
            (self.input_width, self.input_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor = (rgb - self._MEAN) / self._STD
        tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1))
        return tensor, PoseTransform(cv2.invertAffineTransform(affine))

    def postprocess(
        self,
        outputs: Sequence[np.ndarray],
        transform: PoseTransform,
        request: PoseRequest,
    ) -> PoseResult:
        if len(outputs) != 2:
            raise ValueError("RTMPose expects simcc_x and simcc_y")
        simcc_x, simcc_y = (np.asarray(output) for output in outputs)
        if simcc_x.ndim == 3:
            simcc_x = simcc_x[0]
        if simcc_y.ndim == 3:
            simcc_y = simcc_y[0]
        expected_x = self.input_width * 2
        expected_y = self.input_height * 2
        if simcc_x.shape != (17, expected_x) or simcc_y.shape != (17, expected_y):
            raise ValueError(f"invalid SimCC shapes: {simcc_x.shape}, {simcc_y.shape}")

        x_index = np.argmax(simcc_x, axis=1)
        y_index = np.argmax(simcc_y, axis=1)
        confidence = np.minimum(np.max(simcc_x, axis=1), np.max(simcc_y, axis=1))
        crop_points = np.column_stack((x_index, y_index)).astype(np.float32)
        crop_points /= self._SIMCC_SPLIT_RATIO
        homogeneous = np.column_stack((crop_points, np.ones(17, dtype=np.float32)))
        frame_points = homogeneous @ transform.inverse_affine.T
        frame_height, frame_width = request.frame.frame.shape[:2]
        keypoints: list[Keypoint] = []
        for name, (x, y), score in zip(COCO17_KEYPOINT_NAMES, frame_points, confidence):
            score_value = float(np.clip(score, 0.0, 1.0))
            visible = bool(
                score_value >= self.configuration.keypoint_threshold
                and 0.0 <= x < frame_width
                and 0.0 <= y < frame_height
            )
            keypoints.append(
                Keypoint(
                    name=name,
                    x=float(x) if visible else None,
                    y=float(y) if visible else None,
                    confidence=score_value,
                    visible=visible,
                )
            )
        return PoseResult(
            camera_id=request.camera_id,
            track_id=request.track_id,
            bbox=BoundingBox(*request.bbox_xyxy),
            keypoints=tuple(keypoints),
        )

    def infer_requests(self, requests: Sequence[PoseRequest]) -> Sequence[PoseResult]:
        if not requests:
            return []
        prepared = [
            self.preprocess(request.frame.frame, BoundingBox(*request.bbox_xyxy))
            for request in requests
        ]
        outputs = self.backend.infer(np.stack([item[0] for item in prepared]))
        if not outputs or outputs[0].shape[0] != len(requests):
            raise ValueError("RTMPose batch mapping mismatch")
        return [
            self.postprocess(
                [output[index : index + 1] for output in outputs],
                prepared[index][1],
                request,
            )
            for index, request in enumerate(requests)
        ]
