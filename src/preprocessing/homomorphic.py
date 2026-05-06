from __future__ import annotations

import cv2
import numpy as np


def homomorphic_filter(frame_bgr: np.ndarray, gamma_l: float = 0.7, gamma_h: float = 1.5, c: float = 1.0, d0: float = 30.0) -> np.ndarray:
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img)

    y = y.astype(np.float32) + 1.0
    log_y = np.log(y)

    M, N = log_y.shape
    y_fft = np.fft.fft2(log_y)
    y_fft_shift = np.fft.fftshift(y_fft)

    u = np.arange(M) - M / 2
    v = np.arange(N) - N / 2
    V, U = np.meshgrid(v, u)
    D = np.sqrt(U**2 + V**2)

    H = (gamma_h - gamma_l) * (1 - np.exp(-c * (D**2) / (d0**2))) + gamma_l

    filtered = H * y_fft_shift
    filtered_ishift = np.fft.ifftshift(filtered)
    y_ifft = np.fft.ifft2(filtered_ishift)
    y_exp = np.exp(np.real(y_ifft))

    y_out = np.clip(y_exp - 1.0, 0, 255).astype(np.uint8)

    out = cv2.merge([y_out, cr, cb])
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)
