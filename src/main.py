from __future__ import annotations

import cv2

from .config import Settings
from .pipeline import Pipeline


def main() -> None:
    settings = Settings.from_env()
    pipeline = Pipeline(settings)

    cap = cv2.VideoCapture(settings.video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {settings.video_source!r}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        pipeline.process_frame(frame)

    cap.release()


if __name__ == "__main__":
    main()
