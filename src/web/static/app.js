const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const $ = (id) => document.getElementById(id);

const btnCrosswalk = $("mode-crosswalk");
const btnRoi = $("mode-roi");
const btnManualRed = $("mode-manualred");

const crosswalkTools = $("crosswalk-tools");
const manualredTools = $("manualred-tools");

const btnCrosswalkSave = $("crosswalk-save");
const btnCrosswalkUndo = $("crosswalk-undo");
const btnCrosswalkClear = $("crosswalk-clear");

const btnManualRedDisable = $("manualred-disable");
const radiusSlider = $("manualred-radius");
const radiusVal = $("manualred-radius-val");

const btnReset = $("reset");
const btnPause = $("pause");
const btnResume = $("resume");
const btnRestart = $("restart");

const cfgPre = $("config");
const statusDiv = $("status");

let mode = "crosswalk"; // crosswalk | roi | manualred
let crosswalkDraft = []; // [[x,y],...]
let roiDraft = null; // {x1,y1,x2,y2}
let isDragging = false;
let dragStart = null;

// Frame dimensions (frame coordinates) provided by backend
let frameW = 0;
let frameH = 0;

function setMode(m) {
  mode = m;

  btnCrosswalk?.classList.toggle("active", mode === "crosswalk");
  btnRoi?.classList.toggle("active", mode === "roi");
  btnManualRed?.classList.toggle("active", mode === "manualred");

  crosswalkTools?.classList.toggle("active", mode === "crosswalk");
  manualredTools?.classList.toggle("active", mode === "manualred");

  if (mode !== "crosswalk") crosswalkDraft = [];
  if (mode !== "roi") roiDraft = null;

  redraw();
}

function resizeCanvasToVideo() {
  const rect = video.getBoundingClientRect();
  canvas.width = Math.round(rect.width);
  canvas.height = Math.round(rect.height);
  redraw();
}

function clientToFrameXY(clientX, clientY) {
  const rect = video.getBoundingClientRect();
  const xOnView = clientX - rect.left;
  const yOnView = clientY - rect.top;

  // Prefer backend status dims; fallback to naturalWidth/Height
  const w = frameW || video.naturalWidth || 0;
  const h = frameH || video.naturalHeight || 0;
  if (!w || !h) return null;

  const sx = w / rect.width;
  const sy = h / rect.height;

  const x = Math.round(xOnView * sx);
  const y = Math.round(yOnView * sy);

  return { x, y };
}

function frameToViewXY(x, y) {
  const rect = video.getBoundingClientRect();
  const w = frameW || video.naturalWidth || 1;
  const h = frameH || video.naturalHeight || 1;
  const sx = rect.width / w;
  const sy = rect.height / h;
  return { x: x * sx, y: y * sy };
}

async function apiJson(url, options = {}) {
  const r = await fetch(url, options);
  if (!r.ok) {
    const t = await r.text().catch(() => "");
    throw new Error(`HTTP ${r.status} for ${url}: ${t}`);
  }
  return await r.json();
}

async function apiGetConfig() { return apiJson("/api/config"); }
async function apiGetStatus() { return apiJson("/api/status"); }

async function apiSetCrosswalk(points) {
  return apiJson("/api/crosswalk", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ points })
  });
}

async function apiSetRoi(roi) {
  return apiJson("/api/traffic_light_roi", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ roi })
  });
}

async function apiSetManualRed(payload) {
  return apiJson("/api/manual_red", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

async function apiDisableManualRed() { return apiJson("/api/manual_red/disable", { method: "POST" }); }
async function apiReset() { return apiJson("/api/reset", { method: "POST" }); }

async function apiPause() { await apiJson("/api/control/pause", { method: "POST" }); }
async function apiResume() { await apiJson("/api/control/resume", { method: "POST" }); }
async function apiRestart() { await apiJson("/api/control/restart", { method: "POST" }); }

function drawPolygon(points) {
  if (!points || points.length === 0) return;
  ctx.strokeStyle = "#ff3b3b";
  ctx.fillStyle = "#ff3b3b";
  ctx.lineWidth = 2;

  const pv = points.map(([x,y]) => frameToViewXY(x,y));

  ctx.beginPath();
  ctx.moveTo(pv[0].x, pv[0].y);
  for (let i = 1; i < pv.length; i++) ctx.lineTo(pv[i].x, pv[i].y);
  ctx.stroke();

  pv.forEach(p => {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawRoi(roi) {
  if (!roi) return;
  const a = frameToViewXY(roi.x1, roi.y1);
  const b = frameToViewXY(roi.x2, roi.y2);
  const x = Math.min(a.x, b.x);
  const y = Math.min(a.y, b.y);
  const w = Math.abs(a.x - b.x);
  const h = Math.abs(a.y - b.y);

  ctx.strokeStyle = "#ff3b3b";
  ctx.lineWidth = 2;
  ctx.strokeRect(x, y, w, h);
}

function redraw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (mode === "crosswalk") drawPolygon(crosswalkDraft);
  if (mode === "roi") drawRoi(roiDraft);
}

canvas.addEventListener("click", async (e) => {
  const pt = clientToFrameXY(e.clientX, e.clientY);
  if (!pt) return;

  try {
    if (mode === "crosswalk") {
      crosswalkDraft.push([pt.x, pt.y]);
      redraw();
      return;
    }

    if (mode === "manualred") {
      const radius = parseInt(radiusSlider?.value || "14", 10);
      await apiSetManualRed({ enabled: true, x: pt.x, y: pt.y, radius });
      await refreshAll();
      return;
    }
  } catch (err) {
    console.error(err);
    statusDiv.textContent = `Ошибка: ${err.message || err}`;
  }
});

canvas.addEventListener("mousedown", (e) => {
  if (mode !== "roi") return;
  const pt = clientToFrameXY(e.clientX, e.clientY);
  if (!pt) return;

  isDragging = true;
  dragStart = { x: pt.x, y: pt.y };
  roiDraft = { x1: pt.x, y1: pt.y, x2: pt.x, y2: pt.y };
  redraw();
});

canvas.addEventListener("mousemove", (e) => {
  if (mode !== "roi") return;
  if (!isDragging || !dragStart) return;

  const pt = clientToFrameXY(e.clientX, e.clientY);
  if (!pt) return;

  roiDraft = { x1: dragStart.x, y1: dragStart.y, x2: pt.x, y2: pt.y };
  redraw();
});

canvas.addEventListener("mouseup", async () => {
  if (mode !== "roi") return;
  if (!isDragging || !roiDraft) return;

  isDragging = false;

  const x1 = Math.min(roiDraft.x1, roiDraft.x2);
  const y1 = Math.min(roiDraft.y1, roiDraft.y2);
  const x2 = Math.max(roiDraft.x1, roiDraft.x2);
  const y2 = Math.max(roiDraft.y1, roiDraft.y2);

  try {
    await apiSetRoi([x1, y1, x2, y2]);
    roiDraft = null;
    await refreshAll();
    redraw();
  } catch (err) {
    console.error(err);
    statusDiv.textContent = `Ошибка: ${err.message || err}`;
  }
});

// Crosswalk tools
btnCrosswalkSave?.addEventListener("click", async () => {
  if (crosswalkDraft.length < 3) {
    alert("Нужно минимум 3 точки для полигона.");
    return;
  }
  try {
    await apiSetCrosswalk(crosswalkDraft);
    crosswalkDraft = [];
    await refreshAll();
    redraw();
  } catch (err) {
    console.error(err);
    statusDiv.textContent = `Ошибка: ${err.message || err}`;
  }
});

btnCrosswalkUndo?.addEventListener("click", () => {
  crosswalkDraft.pop();
  redraw();
});

btnCrosswalkClear?.addEventListener("click", () => {
  crosswalkDraft = [];
  redraw();
});

// Manual red tools
btnManualRedDisable?.addEventListener("click", async () => {
  try {
    await apiDisableManualRed();
    await refreshAll();
  } catch (err) {
    console.error(err);
    statusDiv.textContent = `Ошибка: ${err.message || err}`;
  }
});

radiusSlider?.addEventListener("input", () => {
  if (radiusVal) radiusVal.textContent = radiusSlider.value;
});

// Global controls
btnReset?.addEventListener("click", async () => {
  try {
    await apiReset();
    crosswalkDraft = [];
    roiDraft = null;
    redraw();
    await refreshAll();
  } catch (err) {
    console.error(err);
    statusDiv.textContent = `Ошибка: ${err.message || err}`;
  }
});

btnPause?.addEventListener("click", async () => { try { await apiPause(); await refreshAll(); } catch(e){ console.error(e); } });
btnResume?.addEventListener("click", async () => { try { await apiResume(); await refreshAll(); } catch(e){ console.error(e); } });
btnRestart?.addEventListener("click", async () => { try { await apiRestart(); await refreshAll(); } catch(e){ console.error(e); } });

// Mode buttons
btnCrosswalk?.addEventListener("click", () => setMode("crosswalk"));
btnRoi?.addEventListener("click", () => setMode("roi"));
btnManualRed?.addEventListener("click", () => setMode("manualred"));

async function refreshAll() {
  try {
    const [cfg, st] = await Promise.all([apiGetConfig(), apiGetStatus()]);
    cfgPre.textContent = JSON.stringify(cfg, null, 2);

    frameW = st.frame_w || frameW;
    frameH = st.frame_h || frameH;

    statusDiv.textContent =
      `Источник: ${st.video_source}
` +
      `Размер: ${st.frame_w}×${st.frame_h}
` +
      `FPS(источник): ${Number(st.source_fps).toFixed(2)} | FPS(вывод): ${Number(st.output_fps).toFixed(2)} | FPS(логика): ${Number(st.process_fps).toFixed(2)} | FPS(детекция): ${Number(st.ped_detect_fps).toFixed(2)}
` +
      `PED_DETECT_STRIDE=${st.ped_detect_stride}, HOLD=${st.ped_hold_frames} frames, OUTPUT_MAX_FPS=${st.output_max_fps}, JPEG_QUALITY=${st.jpeg_quality}
` +
      `STREAM_MAX_WIDTH=${st.stream_max_width}, STREAM_MAX_HEIGHT=${st.stream_max_height}
` +
      `Пауза: ${st.paused ? "да" : "нет"}
` +
      (st.last_alert_ts ? `Последнее нарушение: ${new Date(st.last_alert_ts * 1000).toLocaleTimeString()}
${st.last_alert_msg}
` : "") +
      (st.last_error ? `Ошибка backend: ${st.last_error}` : "");

  } catch (err) {
    console.error(err);
    statusDiv.textContent = `Ошибка: ${err.message || err}`;
  }
}

window.addEventListener("resize", resizeCanvasToVideo);
video.addEventListener("load", () => resizeCanvasToVideo());

// Start
setMode("crosswalk");
refreshAll();
setInterval(() => { resizeCanvasToVideo(); refreshAll(); }, 1500);
