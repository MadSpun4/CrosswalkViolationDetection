from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Tuple
import os
import time

from ..config import Settings
from .config_store import get_store_from_env, CalibrationStore
from .framehub import FrameHub

app = FastAPI(title="Pedestrian Monitoring UI")

# Static
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

settings = Settings.from_env()
store: CalibrationStore = get_store_from_env()
hub = FrameHub(settings, store)
hub.start()


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = os.path.join(static_dir, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


def _mjpeg(gen_func):
    boundary = "frame"

    def gen():
        while True:
            jpg = gen_func(timeout=1.0)
            if jpg is None:
                time.sleep(0.05)
                continue
            yield (f"--{boundary}\r\n"
                   f"Content-Type: image/jpeg\r\n"
                   f"Content-Length: {len(jpg)}\r\n\r\n").encode("utf-8") + jpg + b"\r\n"

    return StreamingResponse(gen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")


@app.get("/stream.mjpg")
def stream_raw():
    return _mjpeg(hub.get_latest_raw)


@app.get("/stream_processed.mjpg")
def stream_processed():
    return _mjpeg(hub.get_latest_processed)


class CrosswalkPayload(BaseModel):
    points: List[Tuple[int, int]] = Field(..., description="Polygon points: [[x,y], ...], length >= 3")


class RoiPayload(BaseModel):
    roi: Tuple[int, int, int, int] = Field(..., description="(x1,y1,x2,y2)")


class ManualRedPayload(BaseModel):
    enabled: bool = Field(True, description="Enable/disable manual red simulation")
    x: int = Field(..., description="Center x in frame coords")
    y: int = Field(..., description="Center y in frame coords")
    radius: int = Field(14, description="Circle radius in pixels (frame coords)")


@app.get("/api/status")
def status():
    return hub.get_status()


@app.get("/api/config")
def get_config():
    cal = store.get()
    return {
        "crosswalk": cal.crosswalk,
        "traffic_light_roi": cal.traffic_light_roi,
        "manual_red": {
            "enabled": cal.manual_red.enabled,
            "x": cal.manual_red.x,
            "y": cal.manual_red.y,
            "radius": cal.manual_red.radius,
        },
    }


@app.post("/api/crosswalk")
def set_crosswalk(payload: CrosswalkPayload):
    store.set_crosswalk(payload.points)
    return JSONResponse({"ok": True, "crosswalk": store.get().crosswalk})


@app.post("/api/traffic_light_roi")
def set_roi(payload: RoiPayload):
    store.set_traffic_light_roi(payload.roi)
    return JSONResponse({"ok": True, "traffic_light_roi": store.get().traffic_light_roi})


@app.post("/api/manual_red")
def set_manual_red(payload: ManualRedPayload):
    store.set_manual_red(payload.enabled, payload.x, payload.y, payload.radius)
    return JSONResponse({"ok": True, "manual_red": get_config()["manual_red"]})


@app.post("/api/manual_red/disable")
def disable_manual_red():
    store.disable_manual_red()
    return JSONResponse({"ok": True, "manual_red": get_config()["manual_red"]})


@app.post("/api/control/pause")
def pause():
    hub.pause()
    return JSONResponse({"ok": True, "paused": True})


@app.post("/api/control/resume")
def resume():
    hub.resume()
    return JSONResponse({"ok": True, "paused": False})


@app.post("/api/control/restart")
def restart():
    hub.restart()
    return JSONResponse({"ok": True})


@app.post("/api/reset")
def reset():
    store.reset()
    return JSONResponse({"ok": True})
