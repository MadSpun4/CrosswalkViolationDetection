from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import cv2

from .config import Settings
from .preprocessing.preprocess import Preprocessor
from .detectors.pedestrian import PedestrianDetector, PedestrianDetection
from .detectors.traffic_light import TrafficLightDetector, TrafficLightState
from .detectors.viola_jones import ViolaJonesDetector


Point = Tuple[int, int]


@dataclass
class FrameResult:
    alert: bool
    message: str
    pedestrians: List[PedestrianDetection] = field(default_factory=list)
    traffic_light: Optional[TrafficLightState] = None
    pedestrian_in_crosswalk: Optional[bool] = None
    pedestrian_crosswalk_states: List[bool] = field(default_factory=list)
    viola_regions: List[Tuple[int, int, int, int]] = field(default_factory=list)
    display_frame_bgr: Optional[np.ndarray] = None
    pedestrian_detection_ran: bool = False


def _point_in_polygon(pt: Point, polygon: List[Point]) -> bool:
    poly = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(poly, pt, False) >= 0


def _bottom_edge_in_polygon(bbox_xyxy: Tuple[int, int, int, int], polygon: List[Point]) -> bool:
    """Проверяет нижний край рамки по трём точкам"""
    x1, y1, x2, y2 = bbox_xyxy
    # Берём низ рамки
    y = int(y2)
    pts = [
        (int(x1), y),
        (int((x1 + x2) / 2), y),
        (int(x2), y),
    ]
    inside = sum(1 for p in pts if _point_in_polygon(p, polygon))
    return inside >= 2


class Pipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.preprocessor = Preprocessor(settings)
        self.viola_detector = ViolaJonesDetector(settings)
        self.ped_detector = PedestrianDetector(settings)
        self.tl_detector = TrafficLightDetector(settings)

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        traffic_light_roi: Optional[Tuple[int, int, int, int]] = None,
        crosswalk_polygon: Optional[List[Point]] = None,
        runtime_processing: Optional[Dict[str, Any]] = None,
    ) -> FrameResult:
        runtime_processing = runtime_processing or {}
        polygon = crosswalk_polygon if crosswalk_polygon is not None else self.settings.crosswalk_polygon
        preprocessing_enabled = bool(runtime_processing.get("preprocessing_enabled", True))
        display_preprocessed = bool(runtime_processing.get("display_preprocessed", self.settings.display_preprocessed))
        traffic_light_inverted = bool(runtime_processing.get("traffic_light_inverted", self.settings.traffic_light_inverted))
        yolo_conf = float(runtime_processing.get("yolo_conf", self.settings.yolo_conf))

        # Светофор по ROI
        tl = self.tl_detector.detect(
            frame_bgr,
            roi_override=traffic_light_roi,
            invert_violation=traffic_light_inverted,
        )

        should_preprocess = preprocessing_enabled and (
            display_preprocessed or (tl is not None and tl.is_violation_signal)
        )
        frame_pp = frame_bgr
        if should_preprocess:
            frame_pp = self.preprocessor.apply(
                frame_bgr,
                enable_homomorphic=runtime_processing.get("enable_homomorphic"),
                enable_hist_eq=runtime_processing.get("enable_hist_eq"),
                enable_gaussian_blur=runtime_processing.get("enable_gaussian_blur"),
                gaussian_kernel=runtime_processing.get("gaussian_kernel"),
            )

        display_frame = frame_pp if (display_preprocessed and preprocessing_enabled) else frame_bgr

        if tl is None or not tl.is_violation_signal:
            return FrameResult(
                alert=False,
                message="",
                traffic_light=tl,
                display_frame_bgr=display_frame,
            )

        # Кандидаты для YOLO
        viola_regions = [r.bbox_xyxy for r in self.viola_detector.detect_regions(frame_pp)]
        if not viola_regions:
            return FrameResult(
                alert=False,
                message="",
                traffic_light=tl,
                viola_regions=[],
                display_frame_bgr=display_frame,
            )

        # YOLO на выбранном кадре
        pedestrians = self.ped_detector.detect_all(
            frame_pp,
            candidate_rois=viola_regions,
            conf_threshold=yolo_conf,
        )
        detection_ran = True

        # Положение пешеходов относительно перехода
        crosswalk_states: List[bool] = []
        if polygon and len(polygon) >= 3:
            crosswalk_states = [
                _bottom_edge_in_polygon(ped.bbox_xyxy, polygon)
                for ped in pedestrians
                if ped.confidence >= yolo_conf
            ]

        ped_in_crosswalk: Optional[bool] = None
        if polygon and len(polygon) >= 3:
            ped_in_crosswalk = any(crosswalk_states) if crosswalk_states else False

        # Нарушение: пешеход в зоне при запрещающем сигнале
        if pedestrians and polygon and len(polygon) >= 3 and any(crosswalk_states):
            return FrameResult(
                alert=True,
                message="VIOLATION ACTIVE",
                pedestrians=pedestrians,
                traffic_light=tl,
                pedestrian_in_crosswalk=ped_in_crosswalk,
                pedestrian_crosswalk_states=crosswalk_states,
                viola_regions=viola_regions,
                display_frame_bgr=display_frame,
                pedestrian_detection_ran=detection_ran,
            )

        return FrameResult(
            alert=False,
            message="",
            pedestrians=pedestrians,
            traffic_light=tl,
            pedestrian_in_crosswalk=ped_in_crosswalk,
            pedestrian_crosswalk_states=crosswalk_states,
            viola_regions=viola_regions,
            display_frame_bgr=display_frame,
            pedestrian_detection_ran=detection_ran,
        )
