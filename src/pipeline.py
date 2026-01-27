from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import cv2

from .config import Settings
from .preprocessing.preprocess import Preprocessor
from .detectors.pedestrian import PedestrianDetector, PedestrianDetection
from .detectors.traffic_light import TrafficLightDetector, TrafficLightState
from .alerting.factory import build_alerter


Point = Tuple[int, int]


@dataclass
class FrameResult:
    alert: bool
    message: str
    pedestrian: Optional[PedestrianDetection] = None
    traffic_light: Optional[TrafficLightState] = None
    pedestrian_in_crosswalk: Optional[bool] = None

    # Debug/telemetry fields for UI
    ped_detector_ran: bool = False
    preprocessed_bgr: Optional[np.ndarray] = None


def _point_in_polygon(pt: Point, polygon: List[Point]) -> bool:
    poly = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(poly, pt, False) >= 0


def _bottom_edge_in_polygon(bbox_xyxy: Tuple[int, int, int, int], polygon: List[Point]) -> bool:
    """Pedestrian is considered in crosswalk if the bottom edge of bbox lies inside polygon.

    We approximate the bottom edge by 3 points (left/center/right) and require >=2 inside.
    """
    x1, y1, x2, y2 = bbox_xyxy
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
        self.ped_detector = PedestrianDetector(settings)
        self.tl_detector = TrafficLightDetector(settings)
        self.alerter = build_alerter(settings)

        # Internal state for detector throttling
        self._frame_i = 0
        self._last_ped: Optional[PedestrianDetection] = None
        self._last_ped_age = 10**9  # frames since last detection

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        traffic_light_roi: Optional[Tuple[int, int, int, int]] = None,
        crosswalk_polygon: Optional[List[Point]] = None,
        include_preprocessed: bool = False,
    ) -> FrameResult:
        """Process one frame.

        Notes for real-time:
        - Preprocessing + traffic-light ROI check + rule evaluation run every call (per frame).
        - Heavy pedestrian detection is throttled by PED_DETECT_STRIDE.
        - If include_preprocessed=True, FrameResult.preprocessed_bgr will be set to the preprocessed frame (for UI debug stream).
        """
        self._frame_i += 1

        # 1) Preprocess
        frame_pp = self.preprocessor.apply(frame_bgr)

        # 2) Traffic light state (cheap; per-frame)
        tl = self.tl_detector.detect(frame_pp, roi_override=traffic_light_roi)

        # 3) Pedestrian detection (throttled)
        stride = max(1, int(self.settings.ped_detect_stride))
        do_detect = ((self._frame_i % stride) == 0)
        ped_detector_ran = bool(do_detect and (self.ped_detector.model is not None))

        ped = self._last_ped
        if do_detect:
            ped = self.ped_detector.detect(frame_pp)
            self._last_ped = ped
            self._last_ped_age = 0
        else:
            self._last_ped_age += 1
            if self.settings.ped_hold_frames >= 0 and self._last_ped_age > self.settings.ped_hold_frames:
                ped = None
                self._last_ped = None

        # 4) Pedestrian-in-crosswalk (bottom edge rule)
        ped_in_crosswalk: Optional[bool] = None
        if ped is not None and crosswalk_polygon and len(crosswalk_polygon) >= 3:
            ped_in_crosswalk = _bottom_edge_in_polygon(ped.bbox_xyxy, crosswalk_polygon)

        # 5) Violation = (pedestrian in crosswalk) AND (pedestrian light is RED)
        alert = (
            ped is not None
            and ped.confidence >= self.settings.yolo_conf
            and (ped_in_crosswalk is True)
            and (tl is not None and tl.is_red)
        )
        msg = "VIOLATION: pedestrian in crosswalk while pedestrian light is RED" if alert else ""

        return FrameResult(
            alert=bool(alert),
            message=msg,
            pedestrian=ped,
            traffic_light=tl,
            pedestrian_in_crosswalk=ped_in_crosswalk,
            ped_detector_ran=ped_detector_ran,
            preprocessed_bgr=(frame_pp if include_preprocessed else None),
        )
