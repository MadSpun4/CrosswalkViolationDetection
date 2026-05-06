from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, List, Tuple
import json
import os


Point = Tuple[int, int]
ROI = Tuple[int, int, int, int]


@dataclass
class Processing:
    preprocessing_enabled: Optional[bool] = None
    enable_homomorphic: Optional[bool] = None
    enable_hist_eq: Optional[bool] = None
    enable_gaussian_blur: Optional[bool] = None
    gaussian_kernel: Optional[int] = None
    yolo_conf: Optional[float] = None
    process_stride: Optional[int] = None
    display_preprocessed: Optional[bool] = None
    traffic_light_inverted: Optional[bool] = None
    processing_mode: str = "neural"


@dataclass
class Calibration:
    # Полигон перехода, минимум 3 точки
    crosswalk: Optional[List[Point]] = None
    traffic_light_roi: Optional[ROI] = None
    processing: Processing = field(default_factory=Processing)


class CalibrationStore:
    """Хранит разметку и параметры анализа"""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._lock = Lock()
        self._cal = Calibration()
        self.load()

    def load(self) -> None:
        with self._lock:
            if not self.path.exists():
                return
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                cross = data.get("crosswalk") or None
                roi = data.get("traffic_light_roi") or None
                pr = data.get("processing") or {}
                self._cal = Calibration(
                    crosswalk=[(int(x), int(y)) for x, y in cross] if cross else None,
                    traffic_light_roi=tuple(roi) if roi else None,
                    processing=Processing(
                        preprocessing_enabled=self._none_or_bool(pr.get("preprocessing_enabled")),
                        enable_homomorphic=self._none_or_bool(pr.get("enable_homomorphic")),
                        enable_hist_eq=self._none_or_bool(pr.get("enable_hist_eq")),
                        enable_gaussian_blur=self._none_or_bool(pr.get("enable_gaussian_blur")),
                        gaussian_kernel=self._none_or_int(pr.get("gaussian_kernel")),
                        yolo_conf=self._none_or_float(pr.get("yolo_conf")),
                        process_stride=self._none_or_int(pr.get("process_stride")),
                        display_preprocessed=self._none_or_bool(pr.get("display_preprocessed")),
                        traffic_light_inverted=self._none_or_bool(pr.get("traffic_light_inverted")),
                        processing_mode=str(pr.get("processing_mode", "neural")),
                    ),
                )
            except Exception:
                self._cal = Calibration()

    @staticmethod
    def _none_or_bool(v: Any) -> Optional[bool]:
        return None if v is None else bool(v)

    @staticmethod
    def _none_or_int(v: Any) -> Optional[int]:
        return None if v is None else int(v)

    @staticmethod
    def _none_or_float(v: Any) -> Optional[float]:
        return None if v is None else float(v)

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "crosswalk": self._cal.crosswalk,
                "traffic_light_roi": self._cal.traffic_light_roi,
                "processing": {
                    "preprocessing_enabled": self._cal.processing.preprocessing_enabled,
                    "enable_homomorphic": self._cal.processing.enable_homomorphic,
                    "enable_hist_eq": self._cal.processing.enable_hist_eq,
                    "enable_gaussian_blur": self._cal.processing.enable_gaussian_blur,
                    "gaussian_kernel": self._cal.processing.gaussian_kernel,
                    "yolo_conf": self._cal.processing.yolo_conf,
                    "process_stride": self._cal.processing.process_stride,
                    "display_preprocessed": self._cal.processing.display_preprocessed,
                    "traffic_light_inverted": self._cal.processing.traffic_light_inverted,
                    "processing_mode": self._cal.processing.processing_mode,
                },
            }
            self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self) -> Calibration:
        with self._lock:
            return Calibration(
                crosswalk=list(self._cal.crosswalk) if self._cal.crosswalk else None,
                traffic_light_roi=tuple(self._cal.traffic_light_roi) if self._cal.traffic_light_roi else None,
                processing=Processing(
                    preprocessing_enabled=self._cal.processing.preprocessing_enabled,
                    enable_homomorphic=self._cal.processing.enable_homomorphic,
                    enable_hist_eq=self._cal.processing.enable_hist_eq,
                    enable_gaussian_blur=self._cal.processing.enable_gaussian_blur,
                    gaussian_kernel=self._cal.processing.gaussian_kernel,
                    yolo_conf=self._cal.processing.yolo_conf,
                    process_stride=self._cal.processing.process_stride,
                    display_preprocessed=self._cal.processing.display_preprocessed,
                    traffic_light_inverted=self._cal.processing.traffic_light_inverted,
                    processing_mode=self._cal.processing.processing_mode,
                ),
            )

    def set_crosswalk(self, pts: List[Point]) -> None:
        if len(pts) < 3:
            raise ValueError("crosswalk polygon must contain at least 3 points")
        with self._lock:
            self._cal.crosswalk = [(int(x), int(y)) for x, y in pts]
        self.save()

    def set_traffic_light_roi(self, roi: ROI) -> None:
        x1, y1, x2, y2 = [int(v) for v in roi]
        with self._lock:
            self._cal.traffic_light_roi = (x1, y1, x2, y2)
        self.save()

    def set_processing(self, payload: Dict[str, Any]) -> None:
        allowed_modes = {"neural", "background", "combined"}
        with self._lock:
            pr = self._cal.processing
            if "preprocessing_enabled" in payload:
                pr.preprocessing_enabled = bool(payload["preprocessing_enabled"])
            if "enable_homomorphic" in payload:
                pr.enable_homomorphic = bool(payload["enable_homomorphic"])
            if "enable_hist_eq" in payload:
                pr.enable_hist_eq = bool(payload["enable_hist_eq"])
            if "enable_gaussian_blur" in payload:
                pr.enable_gaussian_blur = bool(payload["enable_gaussian_blur"])
            if "gaussian_kernel" in payload:
                k = max(1, int(payload["gaussian_kernel"]))
                pr.gaussian_kernel = k if k % 2 == 1 else k + 1
            if "yolo_conf" in payload:
                pr.yolo_conf = max(0.0, min(1.0, float(payload["yolo_conf"])))
            if "process_stride" in payload:
                pr.process_stride = max(1, int(payload["process_stride"]))
            if "display_preprocessed" in payload:
                pr.display_preprocessed = bool(payload["display_preprocessed"])
            if "traffic_light_inverted" in payload:
                pr.traffic_light_inverted = bool(payload["traffic_light_inverted"])
            if "processing_mode" in payload:
                mode = str(payload["processing_mode"]).strip().lower()
                if mode in allowed_modes:
                    pr.processing_mode = mode
        self.save()

    def reset(self) -> None:
        with self._lock:
            self._cal = Calibration()
        self.save()


def get_store_from_env() -> CalibrationStore:
    path = os.getenv("RUNTIME_CONFIG", "/app/runtime/calibration.json")
    return CalibrationStore(path)


def effective_processing(cal: Calibration, settings: Any) -> Dict[str, Any]:
    pr = cal.processing
    return {
        "preprocessing_enabled": True if pr.preprocessing_enabled is None else pr.preprocessing_enabled,
        "enable_homomorphic": settings.enable_homomorphic if pr.enable_homomorphic is None else pr.enable_homomorphic,
        "enable_hist_eq": settings.enable_hist_eq if pr.enable_hist_eq is None else pr.enable_hist_eq,
        "enable_gaussian_blur": settings.enable_gaussian_blur if pr.enable_gaussian_blur is None else pr.enable_gaussian_blur,
        "gaussian_kernel": settings.gaussian_kernel if pr.gaussian_kernel is None else pr.gaussian_kernel,
        "yolo_conf": settings.yolo_conf if pr.yolo_conf is None else pr.yolo_conf,
        "process_stride": settings.process_stride if pr.process_stride is None else pr.process_stride,
        "display_preprocessed": settings.display_preprocessed if pr.display_preprocessed is None else pr.display_preprocessed,
        "traffic_light_inverted": settings.traffic_light_inverted if pr.traffic_light_inverted is None else pr.traffic_light_inverted,
        "processing_mode": pr.processing_mode or "neural",
    }
