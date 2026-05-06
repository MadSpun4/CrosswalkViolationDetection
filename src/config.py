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
    yolo_person_class: int

    # ROI-кандидаты Viola-Jones
    viola_cascade_path: str
    viola_scale_factor: float
    viola_min_neighbors: int
    viola_min_width: int
    viola_min_height: int
    viola_padding: float

    # Пороги светофора
    tl_brightness_threshold: float          # средняя яркость ROI
    tl_detection_mode: str                  # режим: brightness | color
    tl_color_fraction_threshold: float      # доля красных/зелёных пикселей
    tl_color_margin: float                  # перевес ведущего цвета
    tl_color_min_saturation: int            # минимум насыщенности HSV
    tl_color_min_value: int                 # минимум яркости HSV
    traffic_light_inverted: bool            # зелёный считается запретом

    # Пропуск кадров
    process_stride: int                     # 1 = каждый кадр
    display_preprocessed: bool              # показывать кадр после предобработки

    @staticmethod
    def from_env() -> "Settings":
        # Поддержка запуска вне Docker.
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass

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
            yolo_person_class=_env_int("YOLO_PERSON_CLASS", 0),

            viola_cascade_path=_env_str("VIOLA_CASCADE_PATH", ""),
            viola_scale_factor=_env_float("VIOLA_SCALE_FACTOR", 1.1),
            viola_min_neighbors=_env_int("VIOLA_MIN_NEIGHBORS", 3),
            viola_min_width=_env_int("VIOLA_MIN_WIDTH", 24),
            viola_min_height=_env_int("VIOLA_MIN_HEIGHT", 48),
            viola_padding=_env_float("VIOLA_PADDING", 0.35),

            tl_brightness_threshold=_env_float("TL_BRIGHTNESS_T", 60.0),
            tl_detection_mode=_env_str("TL_DETECTION_MODE", "color").strip().lower(),
            tl_color_fraction_threshold=_env_float("TL_COLOR_FRACTION_T", 0.003),
            tl_color_margin=_env_float("TL_COLOR_MARGIN", 1.25),
            tl_color_min_saturation=max(0, min(255, _env_int("TL_COLOR_MIN_S", 70))),
            tl_color_min_value=max(0, min(255, _env_int("TL_COLOR_MIN_V", 80))),
            traffic_light_inverted=_env_bool("TRAFFIC_LIGHT_INVERTED", False),

            process_stride=max(1, _env_int("PROCESS_STRIDE", 1)),
            display_preprocessed=_env_bool("DISPLAY_PREPROCESSED", True),
        )
