from __future__ import annotations

import time
import cv2

from .config import Settings
from .pipeline import Pipeline


def main() -> None:
    settings = Settings.from_env()
    pipeline = Pipeline(settings)

    cap = cv2.VideoCapture(settings.video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {settings.video_source!r}")

    last_alert_ts = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = pipeline.process_frame(frame)

        # Simple cooldown to avoid spamming alerts
        now = time.time()
        if result.alert and (now - last_alert_ts) >= settings.alert_cooldown_sec:
            pipeline.alerter.notify(result.message)
            last_alert_ts = now

    cap.release()


if __name__ == "__main__":
    main()
