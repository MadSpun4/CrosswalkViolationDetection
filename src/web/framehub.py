from __future__ import annotations

import os
import time
from threading import Thread, Event, Lock
from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np

from ..config import Settings
from ..pipeline import Pipeline
from .config_store import CalibrationStore


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
    """Background worker.

    Key change in v5:
    - We always run per-frame logic (traffic light + rule evaluation) to keep UI responsive.
    - Heavy pedestrian detection is throttled *inside Pipeline* (PED_DETECT_STRIDE),
      so `process_fps` stays close to `output_fps` even when detection is sparse.
    """

    def __init__(self, settings: Settings, store: CalibrationStore) -> None:
        self.settings = settings
        self.store = store
        self.pipeline = Pipeline(settings)

        # Stream performance knobs
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

        # Status
        self._frame_w = 0
        self._frame_h = 0
        self._fps_src = 0.0
        self._fps_out = 0.0
        self._fps_proc = 0.0           # per-frame logic fps (pipeline called each frame)
        self._fps_ped_det = 0.0        # how often heavy pedestrian detector was invoked
        self._paused_flag = False
        self._last_err: Optional[str] = None

        self._last_alert_ts: Optional[float] = None
        self._last_alert_msg: str = ""

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
        with self._lock:
            return {
                "frame_w": self._frame_w,
                "frame_h": self._frame_h,
                "source_fps": self._fps_src,
                "output_fps": self._fps_out,
                "process_fps": self._fps_proc,
                "ped_detect_fps": self._fps_ped_det,
                "ped_detect_stride": self.settings.ped_detect_stride,
                "ped_hold_frames": self.settings.ped_hold_frames,
                "output_max_fps": self.output_max_fps,
                "jpeg_quality": self.jpeg_quality,
                "stream_max_width": self.stream_max_width,
                "stream_max_height": self.stream_max_height,
                "paused": self._paused_flag,
                "last_error": self._last_err,
                "video_source": self.settings.video_source,
                "last_alert_ts": self._last_alert_ts,
                "last_alert_msg": self._last_alert_msg,
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

        next_ts = time.time()      # pacing for file sources
        out_next_ts = time.time()  # output limiter

        # FPS estimation windows
        t0_out = time.time()
        t0_proc = time.time()
        t0_det = time.time()
        c_out = 0
        c_proc = 0
        c_det = 0

        last_result = None

        while not self._stop.is_set():
            if self._paused.is_set():
                time.sleep(0.05)
                continue

            if self._restart.is_set() and src_is_file:
                self._restart.clear()
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                last_result = None

            ok, frame = cap.read()
            if not ok:
                if src_is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                cap.release()
                time.sleep(0.5)
                cap = self._open_capture()
                if cap is None:
                    time.sleep(0.5)
                    continue
                continue

            h, w = frame.shape[:2]
            with self._lock:
                self._frame_w = int(w)
                self._frame_h = int(h)

            # File pacing only (RTSP/camera are real-time by nature)
            if src_is_file:
                now = time.time()
                if now < next_ts:
                    time.sleep(max(0.0, next_ts - now))
                next_ts = max(next_ts + frame_dt, time.time())

            cal = self.store.get()

            # Manual red simulation before processing
            frame_for_pipeline = frame
            if cal.manual_red.enabled:
                frame_for_pipeline = frame.copy()
                cv2.circle(
                    frame_for_pipeline,
                    (int(cal.manual_red.x), int(cal.manual_red.y)),
                    int(cal.manual_red.radius),
                    (0, 0, 255),
                    thickness=-1,
                )

            # Run per-frame logic (pipeline decides internally whether to invoke heavy detector)
            try:
                before_i = self.pipeline._frame_i  # internal; ok for scaffold status
                last_result = self.pipeline.process_frame(
                    frame_for_pipeline,
                    traffic_light_roi=cal.traffic_light_roi,
                    crosswalk_polygon=cal.crosswalk,
                )
                after_i = self.pipeline._frame_i

                # Detect calls are made when pipeline internal counter hits stride.
                # We approximate ped-det count by checking age reset (implementation detail).
                # For UI, this is sufficient. (c_det is used for fps.)
                # We increment when pipeline's internal frame index % stride == 0.
                if (after_i % max(1, self.settings.ped_detect_stride)) == 0:
                    c_det += 1

                if last_result is not None and getattr(last_result, "alert", False):
                    self.pipeline.alerter.notify(last_result.message)
                    with self._lock:
                        self._last_alert_ts = time.time()
                        self._last_alert_msg = last_result.message
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

            # Output limiter
            if self.output_max_fps and self.output_max_fps > 0:
                now = time.time()
                if now < out_next_ts:
                    continue
                out_next_ts = now + (1.0 / self.output_max_fps)

            # Draw overlays and encode
            frame_vis = self._draw_overlays(frame_for_pipeline, cal, last_result)
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

    def _draw_overlays(self, frame: np.ndarray, cal, result) -> np.ndarray:
        vis = frame.copy()

        # Crosswalk polygon
        if cal.crosswalk and len(cal.crosswalk) >= 3:
            pts = np.array(cal.crosswalk, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            for (x, y) in cal.crosswalk:
                cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)

        # Traffic light ROI
        if cal.traffic_light_roi:
            x1, y1, x2, y2 = cal.traffic_light_roi
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Manual red indicator
        if cal.manual_red.enabled:
            cv2.circle(
                vis,
                (int(cal.manual_red.x), int(cal.manual_red.y)),
                int(cal.manual_red.radius),
                (0, 0, 255),
                2,
            )
            cv2.putText(
                vis,
                "MANUAL RED (sim)",
                (12, 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # Pedestrian bbox
        if result is not None and getattr(result, "pedestrian", None) is not None:
            x1, y1, x2, y2 = result.pedestrian.bbox_xyxy
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                vis,
                f"person {result.pedestrian.confidence:.2f}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            if getattr(result, "pedestrian_in_crosswalk", None) is not None:
                tag = "IN CROSSWALK" if result.pedestrian_in_crosswalk else "OUTSIDE"
                cv2.putText(
                    vis,
                    tag,
                    (x1, min(vis.shape[0] - 8, y2 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255) if result.pedestrian_in_crosswalk else (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        # Traffic light state
        if result is not None and getattr(result, "traffic_light", None) is not None:
            tl = result.traffic_light
            txt = (
                f"ped light: {'RED' if tl.is_red else 'GREEN'} "
                f"(Y={tl.red_score:.1f}, red={tl.red_fraction*100:.1f}%)"
            )
            cv2.putText(
                vis,
                txt,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255) if tl.is_red else (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # Violation banner
        if result is not None and getattr(result, "alert", False):
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 90), (0, 0, 255), thickness=-1)
            cv2.putText(
                vis,
                "VIOLATION DETECTED",
                (12, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

        return vis
