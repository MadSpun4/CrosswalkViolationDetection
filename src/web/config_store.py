from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Optional, List, Tuple
import json
import os


Point = Tuple[int, int]
ROI = Tuple[int, int, int, int]


@dataclass
class ManualRed:
    enabled: bool = False
    x: int = 0
    y: int = 0
    radius: int = 14  # pixels in frame coordinates


@dataclass
class Calibration:
    # Crosswalk polygon: any number of points >= 3 (operator-defined)
    crosswalk: Optional[List[Point]] = None
    traffic_light_roi: Optional[ROI] = None
    manual_red: ManualRed = field(default_factory=ManualRed)


class CalibrationStore:
    """Stores only calibration parameters (ROI/polygon + manual simulation flags), not frames/video."""

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
                mr = data.get("manual_red") or {}
                self._cal = Calibration(
                    crosswalk=[(int(x), int(y)) for x, y in cross] if cross else None,
                    traffic_light_roi=tuple(roi) if roi else None,
                    manual_red=ManualRed(
                        enabled=bool(mr.get("enabled", False)),
                        x=int(mr.get("x", 0)),
                        y=int(mr.get("y", 0)),
                        radius=int(mr.get("radius", 14)),
                    ),
                )
            except Exception:
                self._cal = Calibration()

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "crosswalk": self._cal.crosswalk,
                "traffic_light_roi": self._cal.traffic_light_roi,
                "manual_red": {
                    "enabled": self._cal.manual_red.enabled,
                    "x": self._cal.manual_red.x,
                    "y": self._cal.manual_red.y,
                    "radius": self._cal.manual_red.radius,
                },
            }
            self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self) -> Calibration:
        with self._lock:
            return Calibration(
                crosswalk=list(self._cal.crosswalk) if self._cal.crosswalk else None,
                traffic_light_roi=tuple(self._cal.traffic_light_roi) if self._cal.traffic_light_roi else None,
                manual_red=ManualRed(
                    enabled=self._cal.manual_red.enabled,
                    x=self._cal.manual_red.x,
                    y=self._cal.manual_red.y,
                    radius=self._cal.manual_red.radius,
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

    def set_manual_red(self, enabled: bool, x: int, y: int, radius: int = 14) -> None:
        with self._lock:
            self._cal.manual_red.enabled = bool(enabled)
            self._cal.manual_red.x = int(x)
            self._cal.manual_red.y = int(y)
            self._cal.manual_red.radius = int(radius)
        self.save()

    def disable_manual_red(self) -> None:
        with self._lock:
            self._cal.manual_red.enabled = False
        self.save()

    def reset(self) -> None:
        with self._lock:
            self._cal = Calibration()
        self.save()


def get_store_from_env() -> CalibrationStore:
    path = os.getenv("RUNTIME_CONFIG", "/app/runtime/calibration.json")
    return CalibrationStore(path)
