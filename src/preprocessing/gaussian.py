from __future__ import annotations

import cv2
import numpy as np


def gaussian_blur(frame_bgr: np.ndarray, k: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(frame_bgr, (k, k), 0)
