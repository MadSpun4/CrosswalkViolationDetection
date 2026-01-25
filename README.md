# Automated Pedestrian Violation Monitoring — Project Template (Docker Compose)

This repository is a **starter template** for the diploma project: an automated system that processes a video stream,
detects a pedestrian, checks the pedestrian traffic light state, and issues a **real-time warning** (no data storage).

The template includes:
- A clean Python package layout (`src/`)
- A configurable runtime via `.env`
- A reproducible environment via **Dockerfile + docker-compose**
- Placeholder pipeline steps aligned with the thesis structure (preprocessing + detection + alerting)

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
- `VIDEO_SOURCE` — path in container or RTSP URL
- `CROSSWALK_POLYGON` — polygon for crosswalk region (optional)
- `TRAFFIC_LIGHT_ROI` — rectangular ROI for traffic light analysis (optional)
- `ALERT_MODE` — `console` (default) or `beep` (stub)

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
  alerting/          # warnings (real-time)
```

## Notes
- The implementation in this template is a **scaffold**. Core algorithms will be implemented incrementally.
- The pipeline currently runs and prints basic diagnostics; detection modules are stubs unless configured.


## Web UI
Run with Docker Compose and open in browser:
- http://localhost:8000

The UI allows selecting:
- Crosswalk zone as 4 points (quadrilateral)
- Pedestrian traffic light ROI as a rectangle

Calibration is stored in `runtime/calibration.json`.


## UI additions (v2)
- Crosswalk can be any polygon (>=3 points). Add points by clicking, then press 'Save polygon'.
- Manual red simulation: click to place a red circle; pipeline sees it as part of the frame.
- Playback controls: pause/resume; restart for file sources.


## Performance tuning
- PROCESS_STRIDE: process every N-th frame (e.g., 4 or 6) to reduce compute for real-time camera streams.
- OUTPUT_MAX_FPS: limit MJPEG streaming FPS if CPU-bound on JPEG.
- JPEG_QUALITY: quality for MJPEG stream.

If UI controls seem unresponsive after updates, do a hard refresh (Ctrl+F5) to bypass cached JS.

## Traffic light detection (ROI)
The pedestrian light state is computed by mean grayscale brightness inside ROI and compared with `TL_BRIGHTNESS_T` (default 60). Set ROI in the UI.

## MJPEG stream performance
If UI FPS is much lower than source FPS, set `STREAM_MAX_WIDTH` (e.g., 480) to downscale frames before JPEG encoding.

## FPS semantics (v5)
- output_fps: how fast the browser receives MJPEG frames.
- process_fps: per-frame logic rate (traffic light + rule evaluation). In v5 it should be close to output_fps.
- ped_detect_fps: how often the heavy pedestrian detector runs; controlled by PED_DETECT_STRIDE.

## Crosswalk membership
A pedestrian is considered inside the crosswalk if the *bottom edge* of their bbox lies inside the polygon (approximated by bottom-left/center/right points).
