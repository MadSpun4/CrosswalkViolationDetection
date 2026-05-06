# Automated Pedestrian Violation Monitoring – Project Template (Docker Compose)

This repository is a **starter template** for the diploma project: an automated system that processes a video stream,
detects a pedestrian, checks the pedestrian traffic light state, and issues a **real-time warning** (no data storage).

The template includes:
- A clean Python package layout (`src/`)
- A configurable runtime via `.env` and the web UI
- A reproducible environment via **Dockerfile + docker-compose**
- Pipeline steps aligned with the thesis structure: traffic-light ROI check, preprocessing, Viola-Jones ROI selection, YOLO person detection, and rule classification

## Prerequisites (host machine)
- Docker + Docker Compose v2
- Git

## Quick start
1. Copy environment file:
   ```bash
   cp .env.example .env
   ```

2. Put a test video into `input_data/` (the folder is gitignored):
   - Example: `input_data/test.mp4`
   - Or set `VIDEO_SOURCE` to an RTSP URL in `.env`

3. Build and run:
   ```bash
   docker compose build
   docker compose up
   ```

Stop:
```bash
docker compose down
```

## Configuration
All runtime parameters are controlled via `.env`:
- `VIDEO_SOURCE` – path in container or RTSP URL
- `CROSSWALK_POLYGON` – polygon for crosswalk region (optional)
- `TRAFFIC_LIGHT_ROI` – rectangular ROI for traffic light analysis (optional)
- `YOLO_PERSON_CLASS` – class id used for `person` detections (default `0` for COCO)
- `VIOLA_*` – Viola-Jones candidate ROI parameters before YOLO

## YOLO (PyTorch)
This template uses **Ultralytics** (PyTorch-based) as an optional pedestrian detector:
- If `YOLO_MODEL` is set (e.g., `yolov8n.pt`), the container will download/cached weights inside the image layer during first run,
  unless you mount a local `models/` directory and set a path.

Suggested approach:
- Put weights into `models/` and set `YOLO_MODEL=/app/models/yolov8n.pt`

## Development with PyCharm
PyCharm can use Docker Compose as an interpreter:
- Settings → Python Interpreter → Add → Docker Compose
- Choose `docker-compose.yml`, service `app`

## Project layout
```
src/
  main.py            # entry point
  config.py          # environment config
  pipeline.py        # orchestration
  preprocessing/     # image enhancement
  detectors/         # pedestrian + traffic light
```

## Notes
- Background subtraction and combined processing modes are visible in the UI as reserved controls; they are not bound to an algorithm yet.
- If `YOLO_MODEL` is empty, the pedestrian detector returns no detections.


## Web UI
Run with Docker Compose and open in browser:
- http://localhost:8000

The UI allows selecting:
- Crosswalk zone as a polygon
- Pedestrian traffic light ROI as a rectangle
- Preprocessing toggles and main analysis parameters

Calibration is stored in `runtime/calibration.json`.


## UI additions (v2)
- Crosswalk can be any polygon (>=3 points). Add points by clicking, then press 'Save polygon'.
- Traffic-light inversion can be used to treat green as the forbidden signal for demo videos.
- Playback controls: pause/resume; restart for file sources.


## Performance tuning
- PROCESS_STRIDE: run preprocessing + Viola-Jones + YOLO only on every N-th source frame.
  Example: `PROCESS_STRIDE=6` gives about 5 processed FPS from a 30 FPS input stream.
- DISPLAY_PREPROCESSED: show the preprocessed selected frame in the single MJPEG stream.
- OUTPUT_MAX_FPS: limit MJPEG streaming FPS if CPU-bound on JPEG.
- JPEG_QUALITY: quality for MJPEG stream.

If UI controls seem unresponsive after updates, do a hard refresh (Ctrl+F5) to bypass cached JS.

## Traffic light detection (ROI)
`TL_DETECTION_MODE=color` is the practical default. It searches for saturated red and green
pixels inside the traffic-light ROI and treats the signal as `UNKNOWN` when neither color is
dominant enough. This prevents a random bright ROI from immediately becoming "red".

For the thesis brightness mode, set `TL_DETECTION_MODE=brightness`; then the pedestrian light
state follows the documented formula:

```text
Y(x,y) = 0.299R(x,y) + 0.587G(x,y) + 0.114B(x,y)
Y* = mean(Y over ROI)
red if Y* >= TL_BRIGHTNESS_T
```

Green demo inversion is controlled by the UI button "Инвертировать цвет светофора" or
`TRAFFIC_LIGHT_INVERTED=1`; in that mode crossing on green is classified as a violation.
Set ROI in the UI.

## MJPEG stream performance
If UI FPS is much lower than source FPS, set `STREAM_MAX_WIDTH` (e.g., 480) to downscale frames before JPEG encoding.

## FPS semantics (v7)
- output_fps: how fast the browser receives processed MJPEG frames.
- process_fps: how many selected source frames reach preprocessing + rule evaluation.
- ped_detect_fps: how often YOLO actually runs; by default it matches the selected-frame analysis rate.

## Crosswalk membership
A pedestrian is considered inside the crosswalk if the *bottom edge* of their bbox lies inside the polygon (approximated by bottom-left/center/right points).
