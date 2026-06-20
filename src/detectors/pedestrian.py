from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..config import Settings


ROI = Tuple[int, int, int, int]


@dataclass
class PedestrianDetection:
    bbox_xyxy: ROI
    confidence: float


class PedestrianDetector:
    """Детектор пешеходов на YOLO."""
    
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.model = None

        if (self.s.yolo_model or "").strip():
            # Импортируется только при включённой YOLO-модели.
            from ultralytics import YOLO  # type: ignore

            self.model = YOLO(self.s.yolo_model)

    @staticmethod
    def _iou(a: ROI, b: ROI) -> float:
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

    @staticmethod
    def _clip_roi(roi: ROI, width: int, height: int) -> Optional[ROI]:
        x1, y1, x2, y2 = roi
        x1 = max(0, min(width, int(x1)))
        y1 = max(0, min(height, int(y1)))
        x2 = max(0, min(width, int(x2)))
        y2 = max(0, min(height, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    @staticmethod
    def _bbox_area(roi: ROI) -> int:
        x1, y1, x2, y2 = roi
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def _merge_pair(a: ROI, b: ROI) -> ROI:
        return (
            min(a[0], b[0]),
            min(a[1], b[1]),
            max(a[2], b[2]),
            max(a[3], b[3]),
        )

    @classmethod
    def _merge_all(cls, rois: List[ROI]) -> ROI:
        if not rois:
            raise ValueError("Cannot merge empty ROI list")
        current = rois[0]
        for roi in rois[1:]:
            current = cls._merge_pair(current, roi)
        return current

    @staticmethod
    def _gap(a: ROI, b: ROI) -> Tuple[int, int]:
        """Расстояние между прямоугольниками по X и Y.
        Если прямоугольники пересекаются по оси, расстояние по этой оси равно 0.
        """
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        dx = max(0, max(ax1, bx1) - min(ax2, bx2))
        dy = max(0, max(ay1, by1) - min(ay2, by2))
        return dx, dy

    @classmethod
    def _should_merge(cls, a: ROI, b: ROI, iou_threshold: float, gap_threshold: int) -> bool:
        if cls._iou(a, b) >= iou_threshold:
            return True
        dx, dy = cls._gap(a, b)
        return dx <= gap_threshold and dy <= gap_threshold

    @classmethod
    def _merge_rois(
        cls,
        rois: List[ROI],
        frame_shape: Tuple[int, int],
        iou_threshold: float,
        gap_threshold: int,
    ) -> List[ROI]:
        """Объединяет пересекающиеся и близко расположенные ROI."""
        height, width = frame_shape
        clipped: List[ROI] = []
        seen: set[ROI] = set()
        for roi in rois:
            clipped_roi = cls._clip_roi(roi, width=width, height=height)
            if clipped_roi is None or clipped_roi in seen:
                continue
            clipped.append(clipped_roi)
            seen.add(clipped_roi)

        if len(clipped) <= 1:
            return clipped

        merged = clipped[:]
        changed = True
        while changed:
            changed = False
            next_rois: List[ROI] = []
            used = [False] * len(merged)

            for i, base in enumerate(merged):
                if used[i]:
                    continue
                current = base
                used[i] = True

                for j in range(i + 1, len(merged)):
                    if used[j]:
                        continue
                    candidate = merged[j]
                    if cls._should_merge(current, candidate, iou_threshold, gap_threshold):
                        current = cls._merge_pair(current, candidate)
                        used[j] = True
                        changed = True

                next_rois.append(current)

            merged = next_rois

        # Сначала более крупные области, чтобы стабильнее удалять дубли.
        return sorted(merged, key=cls._bbox_area, reverse=True)

    def _prepare_rois(
        self,
        frame_bgr: np.ndarray,
        candidate_rois: Optional[List[ROI]],
    ) -> List[ROI]:
        height, width = frame_bgr.shape[:2]
        if candidate_rois is None:
            return [(0, 0, width, height)]
        if len(candidate_rois) == 0:
            return []

        merge_iou = float(getattr(self.s, "yolo_roi_merge_iou", 0.10))
        merge_gap = int(getattr(self.s, "yolo_roi_merge_gap", 24))
        max_rois = max(1, int(getattr(self.s, "yolo_max_rois_per_frame", 1)))
        rois = self._merge_rois(
            candidate_rois,
            frame_shape=(height, width),
            iou_threshold=max(0.0, merge_iou),
            gap_threshold=max(0, merge_gap),
        )

        # Объединяем в одну общую ROI
        if len(rois) > max_rois:
            if max_rois == 1:
                return [self._merge_all(rois)]

            kept = rois[:max_rois - 1]
            kept.append(self._merge_all(rois[max_rois - 1:]))
            return sorted(kept, key=self._bbox_area, reverse=True)

        return rois

    def detect_all(
        self,
        frame_bgr: np.ndarray,
        candidate_rois: Optional[List[ROI]] = None,
        conf_threshold: Optional[float] = None,
    ) -> List[PedestrianDetection]:
        if self.model is None:
            return []

        conf_t = float(self.s.yolo_conf if conf_threshold is None else conf_threshold)
        rois = self._prepare_rois(frame_bgr, candidate_rois)
        if not rois:
            return []

        crops_rgb: List[np.ndarray] = []
        crop_offsets: List[ROI] = []
        for x1r, y1r, x2r, y2r in rois:
            crop = frame_bgr[y1r:y2r, x1r:x2r]
            if crop.size == 0:
                continue

            crops_rgb.append(crop[..., ::-1])
            crop_offsets.append((x1r, y1r, x2r, y2r))

        if not crops_rgb:
            return []

        detections: List[PedestrianDetection] = []
        batch_size = max(1, int(getattr(self.s, "yolo_batch_size", 16)))

        for start in range(0, len(crops_rgb), batch_size):
            batch_crops = crops_rgb[start:start + batch_size]
            batch_offsets = crop_offsets[start:start + batch_size]

            # Один пакетный вызов YOLO для группы ROI
            results = self.model.predict(
                batch_crops,
                verbose=False,
                conf=conf_t,
                classes=[int(self.s.yolo_person_class)],
                batch=len(batch_crops),
            )
            if not results:
                continue

            for result, (x1r, y1r, _, _) in zip(results, batch_offsets):
                if result.boxes is None or len(result.boxes) == 0:
                    continue

                boxes = result.boxes
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
