from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from ..config import Settings


@dataclass
class TrafficLightState:
    is_red: bool
    red_score: float        # keep compatibility: mean grayscale brightness (0..255)
    red_fraction: float     # fraction of "red" pixels in ROI (0..1)


class TrafficLightDetector:
    """ROI-based pedestrian traffic light detector.

    Thesis baseline: use ROI + mean grayscale brightness threshold T. 
    In practice ROI may include bright background or green lamp, producing false "red".
    To stay stable for the UI calibration workflow, we add a *red chroma gate*:
      - compute fraction of pixels in ROI that fall into "red" HSV ranges with enough S/V
      - require both: mean brightness >= T and red_fraction >= P

    This still follows the thesis assumption that operator selects ROI at the signal location,
    but makes the system robust to imperfect ROI selection.
    """

    def __init__(self, settings: Settings) -> None:
        self.s = settings

    @staticmethod
    def _clip_roi(roi, w, h):
        x1, y1, x2, y2 = roi
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def detect(
        self,
        frame_bgr: np.ndarray,
        roi_override: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[TrafficLightState]:
        roi = roi_override if roi_override is not None else self.s.traffic_light_roi
        if roi is None:
            return None

        h, w = frame_bgr.shape[:2]
        roi2 = self._clip_roi(roi, w, h)
        if roi2 is None:
            return None
        x1, y1, x2, y2 = roi2
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Brightness (grayscale)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))

        # "Red" fraction (HSV)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        s_t = int(self.s.tl_hsv_s_threshold)
        v_t = int(self.s.tl_hsv_v_threshold)

        lower1 = np.array([0, s_t, v_t], dtype=np.uint8)
        upper1 = np.array([10, 255, 255], dtype=np.uint8)
        lower2 = np.array([160, s_t, v_t], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        red_fraction = float(np.count_nonzero(mask)) / float(mask.size) if mask.size else 0.0

        T = float(self.s.tl_brightness_threshold)
        P = float(self.s.tl_red_fraction_threshold)

        is_red = (mean_brightness >= T) and (red_fraction >= P)

        return TrafficLightState(is_red=is_red, red_score=mean_brightness, red_fraction=red_fraction)
