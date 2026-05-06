from __future__ import annotations

import numpy as np

from typing import Optional

from ..config import Settings
from .homomorphic import homomorphic_filter
from .hist_eq import histogram_equalization
from .gaussian import gaussian_blur


class Preprocessor:
    def __init__(self, settings: Settings) -> None:
        self.s = settings

    def apply(
        self,
        frame_bgr: np.ndarray,
        enable_homomorphic: Optional[bool] = None,
        enable_hist_eq: Optional[bool] = None,
        enable_gaussian_blur: Optional[bool] = None,
        gaussian_kernel: Optional[int] = None,
    ) -> np.ndarray:
        out = frame_bgr

        use_homomorphic = self.s.enable_homomorphic if enable_homomorphic is None else enable_homomorphic
        use_hist_eq = self.s.enable_hist_eq if enable_hist_eq is None else enable_hist_eq
        use_gaussian = self.s.enable_gaussian_blur if enable_gaussian_blur is None else enable_gaussian_blur
        kernel = self.s.gaussian_kernel if gaussian_kernel is None else gaussian_kernel

        if use_homomorphic:
            out = homomorphic_filter(out)

        if use_hist_eq:
            out = histogram_equalization(out)

        if use_gaussian:
            k = int(kernel)
            if k % 2 == 0:
                k += 1
            out = gaussian_blur(out, k)

        return out
