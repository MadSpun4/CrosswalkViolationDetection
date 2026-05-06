from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from ..config import Settings


@dataclass
class TrafficLightState:
    is_red: bool
    red_score: float        # яркость или доля красного
    is_green: bool = False
    green_score: float = 0.0
    mean_brightness: float = 0.0
    signal_color: str = "unknown"        # значения: red, green, unknown
    violation_color: str = "red"         # запрещающий цвет
    is_violation_signal: bool = False
    detection_mode: str = "color"


class TrafficLightDetector:
    """Определяет сигнал светофора внутри ROI"""

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
        invert_violation: bool = False,
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

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        violation_color = "green" if invert_violation else "red"

        mode = (self.s.tl_detection_mode or "color").strip().lower()
        if mode == "brightness":
            # Режим яркости по формуле:
            # Y = 0.299R + 0.587G + 0.114B
            T = float(self.s.tl_brightness_threshold)
            is_red = mean_brightness >= T
            signal_color = "red" if is_red else "green"
            is_green = not is_red
            return TrafficLightState(
                is_red=is_red,
                red_score=mean_brightness,
                is_green=is_green,
                green_score=0.0 if is_red else mean_brightness,
                mean_brightness=mean_brightness,
                signal_color=signal_color,
                violation_color=violation_color,
                is_violation_signal=(signal_color == violation_color),
                detection_mode="brightness",
            )

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat_t = int(self.s.tl_color_min_saturation)
        val_t = int(self.s.tl_color_min_value)

        # Красный в HSV лежит в двух диапазонах
        red_mask_1 = cv2.inRange(hsv, (0, sat_t, val_t), (10, 255, 255))
        red_mask_2 = cv2.inRange(hsv, (170, sat_t, val_t), (179, 255, 255))
        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

        # Зелёный ищем по hue, насыщенности и яркости
        green_mask = cv2.inRange(hsv, (35, sat_t, val_t), (95, 255, 255))

        area = float(crop.shape[0] * crop.shape[1])
        red_fraction = float(cv2.countNonZero(red_mask)) / area if area > 0 else 0.0
        green_fraction = float(cv2.countNonZero(green_mask)) / area if area > 0 else 0.0

        fraction_t = float(self.s.tl_color_fraction_threshold)
        margin = max(1.0, float(self.s.tl_color_margin))
        red_active = red_fraction >= fraction_t and red_fraction >= green_fraction * margin
        green_active = green_fraction >= fraction_t and green_fraction >= red_fraction * margin

        signal_color = "unknown"
        if red_active and not green_active:
            signal_color = "red"
        elif green_active and not red_active:
            signal_color = "green"
        elif red_active and green_active:
            signal_color = "red" if red_fraction >= green_fraction else "green"

        is_red = signal_color == "red"
        is_green = signal_color == "green"
        return TrafficLightState(
            is_red=is_red,
            red_score=red_fraction,
            is_green=is_green,
            green_score=green_fraction,
            mean_brightness=mean_brightness,
            signal_color=signal_color,
            violation_color=violation_color,
            is_violation_signal=(signal_color == violation_color),
            detection_mode="color",
        )
