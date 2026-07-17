import time

import numpy as np

from edge_runtime.capture.latest_frame_buffer import FramePacket
from edge_runtime.inference.pipelines import RTMDetPipeline
from shared.enums import ModelRole
from shared.schemas import ModelConfig


class FakeDetectorBackend:
    input_shape = (1, 3, 320, 320)

    def load(self):
        return None

    def infer(self, inputs):
        batch = inputs.shape[0]
        dets = np.zeros((batch, 2, 5), dtype=np.float32)
        labels = np.zeros((batch, 2), dtype=np.int64)
        dets[:, 0] = [32, 64, 160, 256, 0.9]
        dets[:, 1] = [0, 0, 10, 10, 0.1]
        return [dets, labels]

    def close(self):
        return None


def configuration() -> ModelConfig:
    return ModelConfig(
        name="person-detector",
        role=ModelRole.DETECTOR,
        version="test",
        path="unused.onnx",
        input_shape=(320, 320),
        output_format="mmdeploy-dets-labels",
        score_threshold=0.35,
    )


def packet(camera_id: str) -> FramePacket:
    return FramePacket(
        camera_id,
        1,
        time.time_ns(),
        time.monotonic_ns(),
        np.zeros((360, 640, 3), np.uint8),
    )


def test_rtmdet_preprocessing_matches_official_normalization_and_padding():
    pipeline = RTMDetPipeline(FakeDetectorBackend(), configuration())
    tensor, transform = pipeline.preprocess(np.zeros((360, 640, 3), np.uint8))
    assert tensor.shape == (3, 320, 320)
    assert transform.scale == 0.5
    # Resized image occupies 180 rows; bottom padding is value 114.
    np.testing.assert_allclose(
        tensor[:, 200, 0],
        (114 - pipeline._MEAN) / pipeline._STD,
    )


def test_rtmdet_maps_mmdeploy_boxes_back_to_each_full_frame():
    pipeline = RTMDetPipeline(FakeDetectorBackend(), configuration())
    results = pipeline.infer_frames([packet("cam-a"), packet("cam-b")])
    assert len(results) == 2
    assert results[0][0].bbox.as_xyxy() == (64.0, 128.0, 320.0, 360.0)
    assert results[1][0].class_name == "person"
