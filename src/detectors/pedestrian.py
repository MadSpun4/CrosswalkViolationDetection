from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..config import Settings


@dataclass
class PedestrianDetection:
    bbox_xyxy: Tuple[int, int, int, int]
    confidence: float


class PedestrianDetector:
    """Pedestrian detector (YOLO via PyTorch/Ultralytics if configured).

    If `YOLO_MODEL` is empty, the detector returns None (stub).
    """

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.model = None

        if (self.s.yolo_model or "").strip():
            # Lazy import so the template can run even if user removes ultralytics
            from ultralytics import YOLO  # type: ignore

            self.model = YOLO(self.s.yolo_model)

    def detect(self, frame_bgr: np.ndarray) -> Optional[PedestrianDetection]:
        if self.model is None:
            return None

        # Ultralytics expects RGB by default; it accepts BGR too but we convert explicitly for clarity
        frame_rgb = frame_bgr[..., ::-1]

        results = self.model.predict(frame_rgb, verbose=False)
        if not results:
            return None

        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return None

        # COCO "person" class is 0 in many YOLO configs; this may vary for custom models.
        # Here we choose the highest-confidence box among all classes and rely on later refinement.
        boxes = r0.boxes
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        idx = int(np.argmax(confs))
        x1, y1, x2, y2 = xyxy[idx].astype(int).tolist()
        conf = float(confs[idx])

        return PedestrianDetection((x1, y1, x2, y2), conf)
