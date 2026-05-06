from __future__ import annotations

import cv2
import numpy as np


def histogram_equalization(frame_bgr: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img)
    y_eq = cv2.equalizeHist(y)
    out = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)
