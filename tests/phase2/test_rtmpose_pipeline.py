import time

import numpy as np

from edge_runtime.capture.latest_frame_buffer import FramePacket
from edge_runtime.inference.interfaces import PoseRequest
from edge_runtime.inference.pipelines import RTMPosePipeline
from shared.enums import ModelRole, PosePriority
from shared.schemas import BoundingBox, ModelConfig


class FakePoseBackend:
    def load(self):
        return None

    def infer(self, inputs):
        batch = inputs.shape[0]
        x = np.zeros((batch, 17, 384), np.float32)
        y = np.zeros((batch, 17, 512), np.float32)
        x[:, :, 192] = 0.9
        y[:, :, 256] = 0.8
        x[:, 16] = 0.0
        y[:, 16] = 0.0
        return [x, y]

    def close(self):
        return None


def test_rtmpose_simcc_is_inverse_mapped_and_invisible_is_not_zero_zero():
    configuration = ModelConfig(
        name="human-pose",
        role=ModelRole.POSE,
        version="test",
        path="unused.onnx",
        input_shape=(256, 192),
        output_format="simcc-coco17",
        keypoint_threshold=0.3,
    )
    pipeline = RTMPosePipeline(FakePoseBackend(), configuration)
    frame = FramePacket(
        "cam-1",
        7,
        time.time_ns(),
        time.monotonic_ns(),
        np.zeros((480, 640, 3), np.uint8),
    )
    request = PoseRequest(
        "cam-1",
        12,
        frame,
        (200, 100, 400, 420),
        PosePriority.MEDIUM,
        time.monotonic_ns(),
    )
    result = pipeline.infer_requests([request])[0]
    assert result.identity == ("cam-1", 12)
    nose = result.keypoint("nose")
    assert nose.visible and nose.x is not None and nose.y is not None
    assert abs(nose.x - 300) < 1
    assert abs(nose.y - 260) < 1
    ankle = result.keypoint("right_ankle")
    assert not ankle.visible
    assert ankle.x is None and ankle.y is None


def test_rtmpose_affine_accepts_partially_out_of_frame_person_box():
    configuration = ModelConfig(
        name="human-pose",
        role=ModelRole.POSE,
        version="test",
        path="unused.onnx",
        input_shape=(256, 192),
        output_format="simcc-coco17",
    )
    pipeline = RTMPosePipeline(FakePoseBackend(), configuration)
    tensor, _ = pipeline.preprocess(np.zeros((100, 100, 3), np.uint8), BoundingBox(-20, 5, 50, 120))
    assert tensor.shape == (3, 256, 192)
    assert np.isfinite(tensor).all()
