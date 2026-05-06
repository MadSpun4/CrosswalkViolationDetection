from __future__ import annotations

import os
import time
from threading import Thread, Event, Lock
from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np

from ..config import Settings
from ..pipeline import Pipeline
from .config_store import CalibrationStore, effective_processing


def _is_rtsp(src: str) -> bool:
    s = (src or "").lower().strip()
    return s.startswith("rtsp://") or s.startswith("rtsps://")


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None and v.strip() != "" else default
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None and v.strip() != "" else default
    except Exception:
        return default


class FrameHub:
    """Фоновая обработка кадров и MJPEG-поток"""

    def __init__(self, settings: Settings, store: CalibrationStore) -> None:
        self.settings = settings
        self.store = store
        self.pipeline = Pipeline(settings)

        # Параметры потока
        self.output_max_fps = _env_float("OUTPUT_MAX_FPS", 0.0)
        self.jpeg_quality = max(30, min(95, _env_int("JPEG_QUALITY", 80)))
        self.stream_max_width = max(0, _env_int("STREAM_MAX_WIDTH", 0))
        self.stream_max_height = max(0, _env_int("STREAM_MAX_HEIGHT", 0))

        self._lock = Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._new_frame = Event()
        self._stop = Event()
        self._paused = Event()
        self._restart = Event()
        self._thread: Optional[Thread] = None

        # Статус
        self._frame_w = 0
        self._frame_h = 0
        self._fps_src = 0.0
        self._fps_out = 0.0
        self._fps_proc = 0.0           # FPS анализа
        self._fps_ped_det = 0.0        # FPS YOLO
        self._paused_flag = False
        self._last_err: Optional[str] = None

        self._last_alert_ts: Optional[float] = None
        self._last_alert_msg: str = ""
        self._alert_active_until = 0.0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def pause(self) -> None:
        self._paused.set()
        self._paused_flag = True

    def resume(self) -> None:
        self._paused.clear()
        self._paused_flag = False

    def restart(self) -> None:
        self._restart.set()

    def get_latest(self, timeout: float = 1.0) -> Optional[bytes]:
        self._new_frame.wait(timeout=timeout)
        self._new_frame.clear()
        with self._lock:
            return self._latest_jpeg

    def get_status(self) -> Dict[str, Any]:
        cal = self.store.get()
        proc = effective_processing(cal, self.settings)
        with self._lock:
            now = time.time()
            alert_remaining = max(0.0, self._alert_active_until - now)
            return {
                "frame_w": self._frame_w,
                "frame_h": self._frame_h,
                "source_fps": self._fps_src,
                "output_fps": self._fps_out,
                "process_fps": self._fps_proc,
                "ped_detect_fps": self._fps_ped_det,
                "process_stride": proc["process_stride"],
                "processing": proc,
                "output_max_fps": self.output_max_fps,
                "jpeg_quality": self.jpeg_quality,
                "stream_max_width": self.stream_max_width,
                "stream_max_height": self.stream_max_height,
                "paused": self._paused_flag,
                "last_error": self._last_err,
                "video_source": self.settings.video_source,
                "last_alert_ts": self._last_alert_ts,
                "last_alert_msg": self._last_alert_msg,
                "alert_active": alert_remaining > 0.0,
                "alert_remaining_sec": alert_remaining,
            }

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(self.settings.video_source)
        if not cap.isOpened():
            return None
        return cap

    def _run(self) -> None:
        cap = self._open_capture()
        if cap is None:
            self._last_err = f"Cannot open video source: {self.settings.video_source!r}"
            return

        is_rtsp = _is_rtsp(self.settings.video_source)
        src_is_file = (not is_rtsp) and os.path.exists(self.settings.video_source)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1e-3 or fps > 240:
            fps = 25.0
        self._fps_src = float(fps)
        frame_dt = 1.0 / float(fps)

        next_ts = time.time()      # темп файла
        out_next_ts = time.time()  # лимит вывода

        # Окна расчёта FPS
        t0_out = time.time()
        t0_proc = time.time()
        t0_det = time.time()
        c_out = 0
        c_proc = 0
        c_det = 0

        last_result = None
        source_frame_i = 0

        while not self._stop.is_set():
            if self._paused.is_set():
                time.sleep(0.05)
                continue

            if self._restart.is_set() and src_is_file:
                self._restart.clear()
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                last_result = None
                source_frame_i = 0

            ok, frame = cap.read()
            if not ok:
                if src_is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    source_frame_i = 0
                    continue
                cap.release()
                time.sleep(0.5)
                cap = self._open_capture()
                if cap is None:
                    time.sleep(0.5)
                    continue
                continue

            source_frame_i += 1

            h, w = frame.shape[:2]
            with self._lock:
                self._frame_w = int(w)
                self._frame_h = int(h)

            # Темп для файла
            if src_is_file:
                now = time.time()
                if now < next_ts:
                    time.sleep(max(0.0, next_ts - now))
                next_ts = max(next_ts + frame_dt, time.time())

            cal = self.store.get()
            proc = effective_processing(cal, self.settings)
            process_stride = max(1, int(proc.get("process_stride", 1)))
            if ((source_frame_i - 1) % process_stride) != 0:
                continue

            # Анализ выбранного кадра
            try:
                last_result = self.pipeline.process_frame(
                    frame,
                    traffic_light_roi=cal.traffic_light_roi,
                    crosswalk_polygon=cal.crosswalk,
                    runtime_processing=proc,
                )
                if last_result is not None and getattr(last_result, "pedestrian_detection_ran", False):
                    c_det += 1

                now_alert = time.time()
                if last_result is not None and getattr(last_result, "alert", False):
                    with self._lock:
                        was_active = self._alert_active_until > now_alert
                        self._alert_active_until = now_alert + 3.0
                        self._last_alert_msg = last_result.message
                        if not was_active:
                            self._last_alert_ts = now_alert
                else:
                    with self._lock:
                        if self._alert_active_until <= now_alert:
                            self._last_alert_msg = ""
            except Exception as e:
                self._last_err = f"pipeline error: {e!r}"

            c_proc += 1
            if time.time() - t0_proc >= 2.0:
                with self._lock:
                    self._fps_proc = c_proc / (time.time() - t0_proc)
                t0_proc = time.time()
                c_proc = 0

            if time.time() - t0_det >= 2.0:
                with self._lock:
                    self._fps_ped_det = c_det / (time.time() - t0_det)
                t0_det = time.time()
                c_det = 0

            # Лимит вывода
            if self.output_max_fps and self.output_max_fps > 0:
                now = time.time()
                if now < out_next_ts:
                    continue
                out_next_ts = now + (1.0 / self.output_max_fps)

            # Разметка и JPEG
            display_frame = getattr(last_result, "display_frame_bgr", None) if last_result is not None else None
            if display_frame is None:
                display_frame = frame
            with self._lock:
                alert_active = self._alert_active_until > time.time()
            frame_vis = self._draw_overlays(display_frame, cal, last_result, alert_active)
            frame_vis = self._resize_for_stream(frame_vis)

            ok2, buf = cv2.imencode(
                ".jpg",
                frame_vis,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if ok2:
                with self._lock:
                    self._latest_jpeg = buf.tobytes()
                self._new_frame.set()

            c_out += 1
            if time.time() - t0_out >= 2.0:
                with self._lock:
                    self._fps_out = c_out / (time.time() - t0_out)
                t0_out = time.time()
                c_out = 0

        cap.release()

    def _resize_for_stream(self, frame: np.ndarray) -> np.ndarray:
        if self.stream_max_width <= 0 and self.stream_max_height <= 0:
            return frame

        h, w = frame.shape[:2]
        max_w = self.stream_max_width if self.stream_max_width > 0 else w
        max_h = self.stream_max_height if self.stream_max_height > 0 else h

        scale = min(max_w / w, max_h / h, 1.0)
        if scale >= 1.0:
            return frame

        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

    def _draw_overlays(
        self,
        frame: np.ndarray,
        cal,
        result,
        alert_active: bool = False,
    ) -> np.ndarray:
        vis = frame.copy()

        # Полигон перехода
        if cal.crosswalk and len(cal.crosswalk) >= 3:
            pts = np.array(cal.crosswalk, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            for (x, y) in cal.crosswalk:
                cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)

        # ROI светофора
        if cal.traffic_light_roi:
            x1, y1, x2, y2 = cal.traffic_light_roi
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # ROI-кандидаты Viola-Jones
        if result is not None:
            for x1, y1, x2, y2 in getattr(result, "viola_regions", []) or []:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 180, 0), 1)

        # Рамки пешеходов
        if result is not None:
            pedestrians = getattr(result, "pedestrians", []) or []
            states = getattr(result, "pedestrian_crosswalk_states", []) or []
            for idx, ped in enumerate(pedestrians):
                x1, y1, x2, y2 = ped.bbox_xyxy
                in_crosswalk = states[idx] if idx < len(states) else None
                box_color = (0, 0, 255) if in_crosswalk else (0, 255, 255)
                cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(
                    vis,
                    f"person {ped.confidence:.2f}",
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2,
                    cv2.LINE_AA,
                )
                if in_crosswalk is not None:
                    tag = "IN CROSSWALK" if in_crosswalk else "OUTSIDE"
                    cv2.putText(
                        vis,
                        tag,
                        (x1, min(vis.shape[0] - 8, y2 + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        box_color,
                        2,
                        cv2.LINE_AA,
                    )

        # Состояние светофора
        if result is not None and getattr(result, "traffic_light", None) is not None:
            tl = result.traffic_light
            signal = (getattr(tl, "signal_color", "unknown") or "unknown").upper()
            forbidden = (getattr(tl, "violation_color", "red") or "red").upper()
            mode = getattr(tl, "detection_mode", "brightness")
            if mode == "color":
                txt = f"light:{signal} forbid:{forbidden} R={tl.red_score:.3f} G={tl.green_score:.3f}"
            else:
                txt = f"light:{signal} forbid:{forbidden} Y={tl.red_score:.1f}"
            if signal == "UNKNOWN":
                color = (0, 255, 255)
            elif getattr(tl, "is_violation_signal", False):
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            cv2.putText(
                vis,
                txt,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )

        # Баннер нарушения
        if alert_active:
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 90), (0, 0, 255), thickness=-1)
            cv2.putText(
                vis,
                "VIOLATION ACTIVE",
                (12, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

        return vis
