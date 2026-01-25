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


def _point_in_polygon(pt: Point, polygon: List[Point]) -> bool:
    poly = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(poly, pt, False) >= 0


def _bottom_edge_in_polygon(bbox_xyxy: Tuple[int, int, int, int], polygon: List[Point]) -> bool:
    """Returns True when the *bottom edge* of bbox is inside polygon.

    User requirement: treat a pedestrian as being in the crosswalk when the lower boundary of the bbox
    lies inside the n-gon. In practice we approximate the edge by three points:
      - bottom-left, bottom-center, bottom-right
    and require >= 2 points to be inside.
    """
    x1, y1, x2, y2 = bbox_xyxy
    # Use y2 (bottom) but keep it within reasonable range
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
        self._last_ped_age = 10**9  # frames since last detection (large on init)

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        traffic_light_roi: Optional[Tuple[int, int, int, int]] = None,
        crosswalk_polygon: Optional[List[Point]] = None,
    ) -> FrameResult:
        self._frame_i += 1

        # 1) Preprocess (can be disabled via env flags)
        frame_pp = self.preprocessor.apply(frame_bgr)

        # 2) Traffic light state (always cheap; evaluated per-frame)
        tl = self.tl_detector.detect(frame_pp, roi_override=traffic_light_roi)

        # 3) Pedestrian detection throttling:
        #    - does NOT throttle per-frame logic or UI
        #    - runs the heavy detector once per `ped_detect_stride` frames
        stride = max(1, int(self.settings.ped_detect_stride))
        do_detect = ((self._frame_i % stride) == 0)

        ped = self._last_ped
        if do_detect:
            ped = self.ped_detector.detect(frame_pp)
            self._last_ped = ped
            self._last_ped_age = 0
        else:
            self._last_ped_age += 1
            # Drop stale bbox if too old (to avoid "ghost" pedestrian forever)
            if self.settings.ped_hold_frames >= 0 and self._last_ped_age > self.settings.ped_hold_frames:
                ped = None
                self._last_ped = None

        # 4) Pedestrian-in-crosswalk
        ped_in_crosswalk: Optional[bool] = None
        if ped is not None and crosswalk_polygon and len(crosswalk_polygon) >= 3:
            ped_in_crosswalk = _bottom_edge_in_polygon(ped.bbox_xyxy, crosswalk_polygon)

        # 5) Violation classification (per thesis):
        #    Violation = pedestrian in crosswalk AND pedestrian light is RED.
        #    Safe = no pedestrian OR green (or no configured ROI/crosswalk).
        #    If we cannot determine crosswalk membership (polygon not set), we do NOT raise violation.
        #    This keeps behaviour aligned with the "position relative to boundaries" property.
        if (
            ped is not None
            and ped.confidence >= self.settings.yolo_conf
            and (ped_in_crosswalk is True)
            and (tl is not None and tl.is_red)
        ):
            return FrameResult(
                alert=True,
                message="VIOLATION: pedestrian in crosswalk while pedestrian light is RED",
                pedestrian=ped,
                traffic_light=tl,
                pedestrian_in_crosswalk=ped_in_crosswalk,
            )

        return FrameResult(
            alert=False,
            message="",
            pedestrian=ped,
            traffic_light=tl,
            pedestrian_in_crosswalk=ped_in_crosswalk,
        )
