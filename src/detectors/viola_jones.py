from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from ..config import Settings


ROI = Tuple[int, int, int, int]


@dataclass
class ViolaJonesRegion:
    bbox_xyxy: ROI


class ViolaJonesDetector:
    """Находит ROI-кандидаты для пешеходов"""

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        cascade_path = settings.viola_cascade_path.strip()
        if not cascade_path:
            cascade_path = cv2.data.haarcascades + "haarcascade_fullbody.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)

    @property
    def available(self) -> bool:
        return not self._cascade.empty()

    def detect_regions(self, frame_bgr: np.ndarray) -> List[ViolaJonesRegion]:
        if not self.available:
            return []

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        boxes = self._cascade.detectMultiScale(
            gray,
            scaleFactor=float(self.s.viola_scale_factor),
            minNeighbors=int(self.s.viola_min_neighbors),
            minSize=(int(self.s.viola_min_width), int(self.s.viola_min_height)),
        )

        h, w = frame_bgr.shape[:2]
        regions: List[ViolaJonesRegion] = []
        pad = float(self.s.viola_padding)
        for x, y, bw, bh in boxes:
            px = int(round(bw * pad))
            py = int(round(bh * pad))
            x1 = max(0, int(x) - px)
            y1 = max(0, int(y) - py)
            x2 = min(w, int(x + bw) + px)
            y2 = min(h, int(y + bh) + py)
            if x2 > x1 and y2 > y1:
                regions.append(ViolaJonesRegion((x1, y1, x2, y2)))

        return regions
