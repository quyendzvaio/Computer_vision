"""Alert service — pipeline from DetectionResult to dispatched Violation."""
from typing import List, Optional

import numpy as np

from alert.roi_matcher import ROIMatcher
from alert.classifier import ViolationClassifier
from alert.cooldown import CooldownManager
from alert.dispatcher import Dispatcher
from shared.models import DetectionResult, Violation


class AlertPipeline:
    """Orchestrates the four-stage alert pipeline:
    1. ROI filtering
    2. Violation classification
    3. Cooldown deduplication
    4. Dispatch (DB + thumbnail + WebSocket)
    """

    def __init__(
        self,
        roi_matcher: ROIMatcher,
        classifier: ViolationClassifier,
        cooldown: CooldownManager,
        dispatcher: Dispatcher,
    ):
        self.roi = roi_matcher
        self.classifier = classifier
        self.cooldown = cooldown
        self.dispatcher = dispatcher

    def process(
        self, result: DetectionResult, frame_bgr: Optional[np.ndarray] = None
    ) -> List[Violation]:
        """Run a DetectionResult through the full alert pipeline.
        Returns the list of violations that were actually dispatched."""
        dispatched: List[Violation] = []

        violations = self.classifier.classify(result)

        for violation in violations:
            # 1. ROI check — is_in_roi takes List[float], so pass bbox as list
            if not self.roi.is_in_roi(violation.camera_id, violation.bbox.to_list()):
                continue

            # 2. Cooldown check
            if not self.cooldown.should_alert(violation.camera_id, violation.type):
                continue

            # 3. Dispatch
            self.dispatcher.dispatch(violation, frame_bgr=frame_bgr)
            dispatched.append(violation)

        return dispatched
