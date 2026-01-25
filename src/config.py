from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional, Tuple, List


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return float(v)


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None else v


def _parse_roi(s: str) -> Optional[Tuple[int, int, int, int]]:
    s = (s or "").strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("TRAFFIC_LIGHT_ROI must be 'x1,y1,x2,y2'")
    return tuple(int(x) for x in parts)  # type: ignore[return-value]


def _parse_polygon(s: str) -> Optional[List[Tuple[int, int]]]:
    s = (s or "").strip()
    if not s:
        return None
    pts = []
    for chunk in s.split(";"):
        xy = [p.strip() for p in chunk.split(",")]
        if len(xy) != 2:
            raise ValueError("CROSSWALK_POLYGON must be 'x1,y1;x2,y2;...'")
        pts.append((int(xy[0]), int(xy[1])))
    return pts


@dataclass(frozen=True)
class Settings:
    video_source: str
    traffic_light_roi: Optional[Tuple[int, int, int, int]]
    crosswalk_polygon: Optional[List[Tuple[int, int]]]

    enable_homomorphic: bool
    enable_hist_eq: bool
    enable_gaussian_blur: bool
    gaussian_kernel: int

    yolo_model: str
    yolo_conf: float

    # Traffic light detection thresholds
    tl_brightness_threshold: float          # grayscale mean threshold
    tl_red_fraction_threshold: float        # fraction of "red-ish" pixels in ROI
    tl_hsv_s_threshold: int                 # saturation gate for red mask
    tl_hsv_v_threshold: int                 # value gate for red mask

    # Pedestrian detection throttling (does NOT throttle UI / per-frame logic)
    ped_detect_stride: int                  # 1 = run detector each frame, 4 = every 4th frame
    ped_hold_frames: int                    # how long to keep last bbox if detector skipped (in frames)

    alert_mode: str
    alert_cooldown_sec: float

    @staticmethod
    def from_env() -> "Settings":
        # Allow python-dotenv if user runs outside Docker
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass

        # Backward compatibility: PROCESS_STRIDE from older versions maps to PED_DETECT_STRIDE.
        ped_stride = _env_int("PED_DETECT_STRIDE", 0)
        if ped_stride <= 0:
            ped_stride = _env_int("PROCESS_STRIDE", 1)

        return Settings(
            video_source=_env_str("VIDEO_SOURCE", "/app/input_data/test.mp4"),
            traffic_light_roi=_parse_roi(os.getenv("TRAFFIC_LIGHT_ROI", "")),
            crosswalk_polygon=_parse_polygon(os.getenv("CROSSWALK_POLYGON", "")),

            enable_homomorphic=_env_bool("ENABLE_HOMOMORPHIC", True),
            enable_hist_eq=_env_bool("ENABLE_HIST_EQ", True),
            enable_gaussian_blur=_env_bool("ENABLE_GAUSSIAN_BLUR", True),
            gaussian_kernel=int(_env_str("GAUSSIAN_KERNEL", "5")),

            yolo_model=_env_str("YOLO_MODEL", ""),
            yolo_conf=_env_float("YOLO_CONF", 0.35),

            # Default values align with the thesis scaffolding, but we add a red-color gate
            # to prevent "bright but not red" false positives inside ROI.
            tl_brightness_threshold=_env_float("TL_BRIGHTNESS_T", 60.0),
            tl_red_fraction_threshold=_env_float("TL_RED_FRACTION_T", 0.010),
            tl_hsv_s_threshold=_env_int("TL_HSV_S_T", 90),
            tl_hsv_v_threshold=_env_int("TL_HSV_V_T", 90),

            ped_detect_stride=max(1, int(ped_stride)),
            ped_hold_frames=max(0, _env_int("PED_HOLD_FRAMES", 12)),

            alert_mode=_env_str("ALERT_MODE", "console"),
            alert_cooldown_sec=_env_float("ALERT_COOLDOWN_SEC", 2.0),
        )
