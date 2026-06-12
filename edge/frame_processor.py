"""Frame processor: motion-based skip, ROI crop, resize, JPEG encode."""
import cv2
import numpy as np
from typing import Optional, List, Tuple


class FrameProcessor:
    """Processes raw camera frames before sending to inference:
    1. Motion-based frame skipping (reduces 30fps -> ~5fps effective)
    2. ROI crop to bounding rectangle
    3. Resize to target dimensions (default 416x416)
    4. JPEG encoding for efficient transport
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (416, 416),
        motion_threshold: float = 0.05,
        jpeg_quality: int = 70,
    ):
        self.target_size = target_size
        self.motion_threshold = motion_threshold
        self.jpeg_quality = jpeg_quality
        self._prev_frame_gray: Optional[np.ndarray] = None

    def should_skip(self, frame: np.ndarray) -> bool:
        """Returns True if this frame should be skipped due to lack of motion.
        Always returns False on the first frame (no previous frame to compare)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_frame_gray is None:
            self._prev_frame_gray = gray
            return False

        # Resize to small size for fast diff comparison
        small = cv2.resize(gray, (160, 120))
        prev_small = cv2.resize(self._prev_frame_gray, (160, 120))

        diff = cv2.absdiff(small, prev_small)
        motion_ratio = np.count_nonzero(diff > 15) / diff.size

        self._prev_frame_gray = gray
        return motion_ratio < self.motion_threshold

    def crop_to_roi(
        self, frame: np.ndarray, roi_polygon: List[Tuple[float, float]]
    ) -> np.ndarray:
        """Crop frame to the bounding rectangle of the ROI polygon, then resize."""
        if not roi_polygon:
            return self.resize(frame)

        xs = [p[0] for p in roi_polygon]
        ys = [p[1] for p in roi_polygon]

        x1 = max(0, int(min(xs)))
        y1 = max(0, int(min(ys)))
        x2 = min(frame.shape[1], int(max(xs)))
        y2 = min(frame.shape[0], int(max(ys)))

        if x2 <= x1 or y2 <= y1:
            return self.resize(frame)

        cropped = frame[y1:y2, x1:x2]
        return self.resize(cropped)

    def resize(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to target dimensions."""
        return cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)

    def encode_jpeg(self, frame: np.ndarray, quality: Optional[int] = None) -> bytes:
        """Encode a BGR frame as JPEG bytes."""
        q = quality if quality is not None else self.jpeg_quality
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, q])
        return buf.tobytes()

    def process(
        self,
        frame: np.ndarray,
        roi_polygon: Optional[List[Tuple[float, float]]] = None,
    ) -> Optional[bytes]:
        """Full processing pipeline for one frame.
        Returns JPEG bytes if the frame passes motion detection, None if skipped."""
        if self.should_skip(frame):
            return None

        if roi_polygon:
            processed = self.crop_to_roi(frame, roi_polygon)
        else:
            processed = self.resize(frame)

        return self.encode_jpeg(processed)
