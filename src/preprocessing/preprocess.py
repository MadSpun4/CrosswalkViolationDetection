from __future__ import annotations

import cv2
import numpy as np

from ..config import Settings
from .homomorphic import homomorphic_filter
from .hist_eq import histogram_equalization
from .gaussian import gaussian_blur


class Preprocessor:
    def __init__(self, settings: Settings) -> None:
        self.s = settings

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        out = frame_bgr

        if self.s.enable_homomorphic:
            out = homomorphic_filter(out)

        if self.s.enable_hist_eq:
            out = histogram_equalization(out)

        if self.s.enable_gaussian_blur:
            k = int(self.s.gaussian_kernel)
            if k % 2 == 0:
                k += 1
            out = gaussian_blur(out, k)

        return out
