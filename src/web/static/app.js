const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const $ = (id) => document.getElementById(id);

const btnCrosswalk = $("mode-crosswalk");
const btnRoi = $("mode-roi");

const crosswalkTools = $("crosswalk-tools");

const btnCrosswalkSave = $("crosswalk-save");

const btnReset = $("reset");
const btnPause = $("pause");
const btnResume = $("resume");
const btnRestart = $("restart");
const btnProcessingSave = $("processing-save");
const btnAudioEnable = $("audio-enable");
const btnDisplayPreprocessed = $("display-preprocessed-toggle");
const btnTrafficLightInvert = $("traffic-light-invert");

const procModeNeural = $("proc-mode-neural");
const procModeBg = $("proc-mode-bg");
const procModeCombined = $("proc-mode-combined");
const preprocessingEnabled = $("preprocessing-enabled");
const enableHomomorphic = $("enable-homomorphic");
const enableHistEq = $("enable-hist-eq");
const enableGaussianBlur = $("enable-gaussian-blur");
const gaussianKernel = $("gaussian-kernel");
const yoloConf = $("yolo-conf");
const processStride = $("process-stride");

const cfgPre = $("config");
const statusDiv = $("status");

let mode = "crosswalk"; // режим: crosswalk или roi
let crosswalkDraft = []; // точки полигона
let roiDraft = null; // прямоугольник ROI
let isDragging = false;
let dragStart = null;

// Размер кадра с сервера
let frameW = 0;
let frameH = 0;
let audioCtx = null;
let audioEnabled = false;
let violationSoundTimer = null;
let violationSoundStopTimer = null;
let violationSoundRequested = false;
let currentProcessing = {};
const VIOLATION_SOUND_REPEAT_MS = 850;

function setMode(m) {
  mode = m;

  btnCrosswalk?.classList.toggle("active", mode === "crosswalk");
  btnRoi?.classList.toggle("active", mode === "roi");

  crosswalkTools?.classList.toggle("active", mode === "crosswalk");

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

function renderedVideoRect() {
  const rect = video.getBoundingClientRect();
  const w = frameW || video.naturalWidth || 0;
  const h = frameH || video.naturalHeight || 0;
  if (!w || !h || !rect.width || !rect.height) return rect;

  const frameAspect = w / h;
  const boxAspect = rect.width / rect.height;
  if (boxAspect > frameAspect) {
    const width = rect.height * frameAspect;
    return {
      left: rect.left + (rect.width - width) / 2,
      top: rect.top,
      width,
      height: rect.height
    };
  }

  const height = rect.width / frameAspect;
  return {
    left: rect.left,
    top: rect.top + (rect.height - height) / 2,
    width: rect.width,
    height
  };
}

function clientToFrameXY(clientX, clientY) {
  const rect = renderedVideoRect();
  const xOnView = clientX - rect.left;
  const yOnView = clientY - rect.top;

  // Сначала берём размеры сервера
  const w = frameW || video.naturalWidth || 0;
  const h = frameH || video.naturalHeight || 0;
  if (!w || !h) return null;
  if (xOnView < 0 || yOnView < 0 || xOnView > rect.width || yOnView > rect.height) return null;

  const sx = w / rect.width;
  const sy = h / rect.height;

  const x = Math.round(xOnView * sx);
  const y = Math.round(yOnView * sy);

  return { x, y };
}

function frameToViewXY(x, y) {
  const videoRect = video.getBoundingClientRect();
  const rect = renderedVideoRect();
  const w = frameW || video.naturalWidth || 1;
  const h = frameH || video.naturalHeight || 1;
  const sx = rect.width / w;
  const sy = rect.height / h;
  return {
    x: (rect.left - videoRect.left) + x * sx,
    y: (rect.top - videoRect.top) + y * sy
  };
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

async function apiSetProcessing(payload) {
  return apiJson("/api/processing", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

async function apiReset() { return apiJson("/api/reset", { method: "POST" }); }

async function apiPause() { await apiJson("/api/control/pause", { method: "POST" }); }
async function apiResume() { await apiJson("/api/control/resume", { method: "POST" }); }
async function apiRestart() { await apiJson("/api/control/restart", { method: "POST" }); }

async function enableAudio() {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) return;
  if (!audioCtx) audioCtx = new AudioContextClass();
  if (audioCtx.state === "suspended") await audioCtx.resume();
  audioEnabled = true;
  if (btnAudioEnable) btnAudioEnable.textContent = "Звук включен";
  if (violationSoundRequested) startViolationSoundLoop();
}

function playViolationSound() {
  if (!audioEnabled || !audioCtx) return;
  if (audioCtx.state === "suspended") {
    audioCtx.resume().catch(console.error);
    return;
  }

  const now = audioCtx.currentTime;
  const master = audioCtx.createGain();
  master.gain.setValueAtTime(0.0001, now);
  master.gain.exponentialRampToValueAtTime(0.18, now + 0.02);
  master.gain.exponentialRampToValueAtTime(0.0001, now + 0.75);
  master.connect(audioCtx.destination);

  [0.0, 0.24, 0.48].forEach((offset) => {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = "sine";
    osc.frequency.setValueAtTime(880, now + offset);
    gain.gain.setValueAtTime(0.0001, now + offset);
    gain.gain.exponentialRampToValueAtTime(1.0, now + offset + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + offset + 0.16);
    osc.connect(gain);
    gain.connect(master);
    osc.start(now + offset);
    osc.stop(now + offset + 0.18);
  });

  window.setTimeout(() => {
    try {
      master.disconnect();
    } catch (err) {
      // Просто отлавливает ошибку отключения уже отключенного узла
    }
  }, 900);
}

function startViolationSoundLoop() {
  if (!audioEnabled || !audioCtx || violationSoundTimer !== null) return;
  playViolationSound();
  violationSoundTimer = window.setInterval(playViolationSound, VIOLATION_SOUND_REPEAT_MS);
}

function stopViolationSoundLoop() {
  if (violationSoundTimer !== null) {
    window.clearInterval(violationSoundTimer);
    violationSoundTimer = null;
  }
  if (violationSoundStopTimer !== null) {
    window.clearTimeout(violationSoundStopTimer);
    violationSoundStopTimer = null;
  }
}

function syncViolationSound(alertActive, remainingSec = 0) {
  violationSoundRequested = alertActive;

  if (!alertActive) {
    stopViolationSoundLoop();
    return;
  }

  startViolationSoundLoop();

  if (violationSoundStopTimer !== null) {
    window.clearTimeout(violationSoundStopTimer);
  }

  const remainingMs = Number.isFinite(remainingSec) ? Math.max(0, remainingSec * 1000) : 0;
  if (remainingMs <= 0) return;

  violationSoundStopTimer = window.setTimeout(() => {
    violationSoundStopTimer = null;
    if (violationSoundRequested) {
      violationSoundRequested = false;
      stopViolationSoundLoop();
    }
  }, remainingMs + 150);
}

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

// Инструменты перехода
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

// Общие кнопки
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
btnAudioEnable?.addEventListener("click", async () => { try { await enableAudio(); } catch(e){ console.error(e); } });
document.addEventListener("click", () => { if (!audioEnabled) enableAudio().catch(console.error); }, { once: true });

btnDisplayPreprocessed?.addEventListener("click", async () => {
  try {
    await apiSetProcessing({ display_preprocessed: !Boolean(currentProcessing.display_preprocessed) });
    await refreshAll();
  } catch (err) {
    console.error(err);
    statusDiv.textContent = `Ошибка: ${err.message || err}`;
  }
});

btnTrafficLightInvert?.addEventListener("click", async () => {
  try {
    await apiSetProcessing({ traffic_light_inverted: !Boolean(currentProcessing.traffic_light_inverted) });
    await refreshAll();
  } catch (err) {
    console.error(err);
    statusDiv.textContent = `Ошибка: ${err.message || err}`;
  }
});

btnProcessingSave?.addEventListener("click", async () => {
  try {
    await apiSetProcessing({
      processing_mode: "neural",
      preprocessing_enabled: Boolean(preprocessingEnabled?.checked),
      enable_homomorphic: Boolean(enableHomomorphic?.checked),
      enable_hist_eq: Boolean(enableHistEq?.checked),
      enable_gaussian_blur: Boolean(enableGaussianBlur?.checked),
      gaussian_kernel: parseInt(gaussianKernel?.value || "5", 10),
      yolo_conf: parseFloat(yoloConf?.value || "0.35"),
      process_stride: parseInt(processStride?.value || "1", 10),
      display_preprocessed: Boolean(currentProcessing.display_preprocessed),
      traffic_light_inverted: Boolean(currentProcessing.traffic_light_inverted)
    });
    await refreshAll();
  } catch (err) {
    console.error(err);
    statusDiv.textContent = `Ошибка: ${err.message || err}`;
  }
});

// Переключение режима
btnCrosswalk?.addEventListener("click", () => setMode("crosswalk"));
btnRoi?.addEventListener("click", () => setMode("roi"));

function populateProcessingControls(processing) {
  if (!processing) return;
  currentProcessing = { ...processing };
  const processingInputs = [
    preprocessingEnabled,
    enableHomomorphic,
    enableHistEq,
    enableGaussianBlur,
    gaussianKernel,
    yoloConf,
    processStride
  ];
  if (processingInputs.includes(document.activeElement)) return;

  if (preprocessingEnabled) preprocessingEnabled.checked = Boolean(processing.preprocessing_enabled);
  if (enableHomomorphic) enableHomomorphic.checked = Boolean(processing.enable_homomorphic);
  if (enableHistEq) enableHistEq.checked = Boolean(processing.enable_hist_eq);
  if (enableGaussianBlur) enableGaussianBlur.checked = Boolean(processing.enable_gaussian_blur);
  if (gaussianKernel) gaussianKernel.value = processing.gaussian_kernel ?? 5;
  if (yoloConf) yoloConf.value = processing.yolo_conf ?? 0.35;
  if (processStride) processStride.value = processing.process_stride ?? 1;

  const modeName = processing.processing_mode || "neural";
  procModeNeural?.classList.toggle("active", modeName === "neural");
  procModeBg?.classList.toggle("active", modeName === "background");
  procModeCombined?.classList.toggle("active", modeName === "combined");
  btnDisplayPreprocessed?.classList.toggle("active", Boolean(processing.display_preprocessed));
  btnTrafficLightInvert?.classList.toggle("active", Boolean(processing.traffic_light_inverted));
}

async function refreshAll() {
  try {
    const [cfg, st] = await Promise.all([apiGetConfig(), apiGetStatus()]);
    cfgPre.textContent = JSON.stringify(cfg, null, 2);
    populateProcessingControls(cfg.processing);

    frameW = st.frame_w || frameW;
    frameH = st.frame_h || frameH;

    syncViolationSound(Boolean(st.alert_active), Number(st.alert_remaining_sec) || 0);

    statusDiv.textContent =
      `Размер: ${st.frame_w}×${st.frame_h}
` +
      `FPS(источник): ${Number(st.source_fps).toFixed(2)} | FPS(вывод): ${Number(st.output_fps).toFixed(2)} | FPS(детекция): ${Number(st.ped_detect_fps).toFixed(2)}
` +
      `Пропуск кадров=${st.process_stride}, Сигнал=${st.alert_active ? `${Number(st.alert_remaining_sec).toFixed(1)}с` : "выкл"}
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

// Запуск
setMode("crosswalk");
refreshAll();
setInterval(() => { resizeCanvasToVideo(); refreshAll(); }, 1500);
