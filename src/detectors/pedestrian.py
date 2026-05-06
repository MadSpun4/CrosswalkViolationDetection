from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..config import Settings


@dataclass
class PedestrianDetection:
    bbox_xyxy: Tuple[int, int, int, int]
    confidence: float


class PedestrianDetector:
    """Детектор пешеходов на YOLO"""

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.model = None

        if (self.s.yolo_model or "").strip():
            # Импортирутся только при включённой YOLO-модели
            from ultralytics import YOLO  # type: ignore

            self.model = YOLO(self.s.yolo_model)

    @staticmethod
    def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = float(iw * ih)
        if inter <= 0:
            return 0.0
        area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
        area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0

    @classmethod
    def _dedupe(cls, detections: List[PedestrianDetection], iou_threshold: float = 0.6) -> List[PedestrianDetection]:
        kept: List[PedestrianDetection] = []
        for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
            if all(cls._iou(det.bbox_xyxy, old.bbox_xyxy) < iou_threshold for old in kept):
                kept.append(det)
        return kept

    def detect_all(
        self,
        frame_bgr: np.ndarray,
        candidate_rois: Optional[List[Tuple[int, int, int, int]]] = None,
        conf_threshold: Optional[float] = None,
    ) -> List[PedestrianDetection]:
        if self.model is None:
            return []

        conf_t = float(self.s.yolo_conf if conf_threshold is None else conf_threshold)
        rois = candidate_rois or [(0, 0, frame_bgr.shape[1], frame_bgr.shape[0])]
        detections: List[PedestrianDetection] = []

        for x1r, y1r, x2r, y2r in rois:
            crop = frame_bgr[y1r:y2r, x1r:x2r]
            if crop.size == 0:
                continue

            # Явный перевод BGR в RGB
            crop_rgb = crop[..., ::-1]
            results = self.model.predict(
                crop_rgb,
                verbose=False,
                conf=conf_t,
                classes=[int(self.s.yolo_person_class)],
            )
            if not results:
                continue

            r0 = results[0]
            if r0.boxes is None or len(r0.boxes) == 0:
                continue

            boxes = r0.boxes
            confs = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros_like(confs)

            for conf, cls_id, box in zip(confs, classes, xyxy):
                if float(conf) < conf_t:
                    continue
                if int(cls_id) != int(self.s.yolo_person_class):
                    continue
                x1, y1, x2, y2 = box.astype(int).tolist()
                detections.append(
                    PedestrianDetection(
                        (x1 + x1r, y1 + y1r, x2 + x1r, y2 + y1r),
                        float(conf),
                    )
                )

        return self._dedupe(detections)
