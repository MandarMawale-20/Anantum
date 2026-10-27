const WIDGET_WIDTH = 430;
const PILL_HEIGHT = 60;
const TOP_PAD = 4;
const GAP = 8;
const BOTTOM_PAD = 4;

const widget = document.getElementById("assistantWidget");
const orbShell = document.getElementById("orbShell");
const micBtn = document.getElementById("micBtn");
const pillText = document.getElementById("assistantText");
const bubble = document.getElementById("bubble");
const bubbleText = document.getElementById("bubbleText");
const bubbleNotice = document.getElementById("bubbleNotice");
const changeModelBtn = document.getElementById("changeModelBtn");
const typeBtn = document.getElementById("typeBtn");
const composer = document.getElementById("composer");
const composerInput = document.getElementById("composerInput");
const composerSend = document.getElementById("composerSend");
const menu = document.getElementById("menu");
const below = document.getElementById("below");

let state = "idle";
let curTranscript = "";
let bridgeWatchdogTimer = null;
let bubbleClearTimer = null;
let noticeClearTimer = null;
let stageRafTimer = null;
let deltaRaf = null;
let pendingDelta = "";
let hintTimer = null;
let hintIndex = 0;
let lastNoticeText = "";
let lastNoticeAt = 0;
let siriWave = null;

const IDLE_HINTS = [
  "How can I help?",
  "Try “what's the weather?”",
  "“Set a timer for 10 minutes”",
  "Use the keyboard icon to type",
];

const tauriGlobal = window.__TAURI__;
const tauriInvoke = tauriGlobal?.core?.invoke || null;
const tauriWindow = tauriGlobal?.window || null;

const stateLabels = {
  idle: "How can I help?",
  listening: "Listening…",
  thinking: "Thinking…",
  speaking: "Speaking…",
  stopped: "Disconnected — click to retry",
};

const stateConfig = {
  idle: { amplitude: 0.1, speed: 0.05 },
  listening: { amplitude: 1.5, speed: 0.2 },
  thinking: { amplitude: 0.9, speed: 0.11 },
  speaking: { amplitude: 2.0, speed: 0.15 },
  stopped: { amplitude: 0.05, speed: 0.02 },
};

const widgetStateClasses = ["is-idle", "is-listening", "is-thinking", "is-speaking", "is-stopped"];

/* ------------------------------------------------------------
   Dynamic window sizing. The pill keeps the top slot, the answer
   card / composer open BELOW it, and the window's bottom edge is
   pinned to the corner — so the stage slides up as it grows and
   down as it collapses. The slide itself is animated in Rust.
   ------------------------------------------------------------ */
let lastStageHeight = 0;

function scheduleStageHeight() {
  if (stageRafTimer) return;
  stageRafTimer = setTimeout(async () => {
    stageRafTimer = null;
    await applyStageHeight();
  }, 130);
}

/* Height the window needs for the current layout. Pass a panel to
   measure as if it were already hidden (used while closing it). */
function measureStage(skipPanel) {
  const wasHidden = skipPanel ? skipPanel.hidden : null;
  if (skipPanel) skipPanel.hidden = true;
  const belowHeight = Math.ceil(below.getBoundingClientRect().height);
  if (skipPanel) skipPanel.hidden = wasHidden;

  return belowHeight > 0
    ? TOP_PAD + PILL_HEIGHT + GAP + belowHeight + BOTTOM_PAD
    : TOP_PAD + PILL_HEIGHT + BOTTOM_PAD;
}

async function runStage(height) {
  if (!tauriInvoke || height === lastStageHeight) return false;
  lastStageHeight = height;

  try {
    const result = await tauriInvoke("apply_stage", { width: WIDGET_WIDTH, height });
    if (result && result.ok === false) throw new Error(result.error || "stage resize failed");
    return true;
  } catch {
    document.body.classList.add("no-resize");
    return false;
  }
}

/* Queue-free variant for deliberate open/close toggles: waiting out the
   debounce would leave the new panel clipped until the window caught up. */
function applyStageHeightNow() {
  if (stageRafTimer) {
    clearTimeout(stageRafTimer);
    stageRafTimer = null;
  }
  return runStage(measureStage());
}

/* After a panel is hidden for real, make the cached height honest again
   without moving the window a second time. */
function reconcileStageHeight() {
  const height = measureStage();
  if (height !== lastStageHeight) runStage(height);
}

async function applyStageHeight() {
  await runStage(measureStage());
}

/* Panels below the pill stay painted while the widget slides back down,
   so the collapse reads as one continuous motion instead of a snap. */
const panelTokens = new WeakMap();

function showPanel(panel) {
  panelTokens.set(panel, (panelTokens.get(panel) || 0) + 1);
  panel.hidden = false;
  return applyStageHeightNow();
}

async function hidePanel(panel) {
  if (panel.hidden) return;
  const token = (panelTokens.get(panel) || 0) + 1;
  panelTokens.set(panel, token);

  // Slide back down with the panel still on screen, then remove it once
  // it has dropped out of the visible area.
  await runStage(measureStage(panel));
  if (panelTokens.get(panel) !== token) return;

  panel.hidden = true;
  reconcileStageHeight();
}

/* ------------------------------------------------------------
   Visuals
   ------------------------------------------------------------ */
function initSiriWave() {
  const container = document.getElementById("siri-container");
  if (!window.SiriWave || !container) return;

  // The canvas is only 64x24; rendering it at 3x DPR is wasted
  // pixels. Cap the ratio while the library reads it once.
  const originalDpr = window.devicePixelRatio;
  try {
    Object.defineProperty(window, "devicePixelRatio", { value: 1, configurable: true });
  } catch {
    /* non-fatal: fall back to the native ratio */
  }

  try {
    siriWave = new window.SiriWave({
      container,
      width: 64,
      height: 24,
      speed: 0.08,
      style: "ios9",
      amplitude: stateConfig.idle.amplitude,
      autostart: false,
    });
  } finally {
    try {
      Object.defineProperty(window, "devicePixelRatio", { value: originalDpr, configurable: true });
    } catch {
      /* ignore */
    }
  }
}

function isWaveState(value) {
  return value === "listening" || value === "thinking" || value === "speaking";
}

function startIdleHints() {
  stopIdleHints();
  hintIndex = 0;
  hintTimer = setInterval(() => {
    if (state !== "idle") return;
    hintIndex = (hintIndex + 1) % IDLE_HINTS.length;
    pillText.textContent = IDLE_HINTS[hintIndex];
  }, 6500);
}

function stopIdleHints() {
  if (hintTimer) {
    clearInterval(hintTimer);
    hintTimer = null;
  }
}

function setState(nextState) {
  if (!stateConfig[nextState]) return;
  state = nextState;
  widget.classList.remove(...widgetStateClasses);
  widget.classList.add(`is-${nextState}`);

  const waveActive = isWaveState(nextState);
  if (waveActive && !siriWave) {
    initSiriWave(); // lazy: skip canvas work entirely until first use
  }

  if (siriWave) {
    const cfg = stateConfig[nextState] || stateConfig.idle;
    siriWave.setAmplitude(cfg.amplitude);
    siriWave.setSpeed(cfg.speed);

    // Only run the canvas rAF loop while the wave is visible; an
    // always-on render loop keeps the GPU/compositor busy for nothing.
    try {
      if (waveActive && !document.hidden) {
        siriWave.start?.();
      } else {
        siriWave.stop?.();
      }
    } catch {
      /* SiriWave tolerates repeat calls; ignore transient errors */
    }
  }

  if (nextState === "thinking") {
    pillText.textContent = curTranscript || "Thinking…";
  } else {
    pillText.textContent = stateLabels[nextState] || stateLabels.idle;
  }

  if (nextState === "idle") {
    startIdleHints();
  } else {
    stopIdleHints();
  }
}

function setBubbleText(text) {
  flushPendingDelta(true);
  const clean = (text || "").trim();
  bubbleText.textContent = clean;
  bubble.hidden = clean === "" && bubbleNotice.hidden;
  scheduleStageHeight();
}

/** Cancel any queued streaming write (used when text is replaced). */
function flushPendingDelta(cancel = false) {
  pendingDelta = "";
  if (deltaRaf) {
    cancelAnimationFrame(deltaRaf);
    deltaRaf = null;
  }
  if (cancel) return;
}

/** Queue a streamed token; DOM is written once per animation frame. */
function queueDelta(text) {
  pendingDelta += text || "";
  if (deltaRaf) return;
  deltaRaf = requestAnimationFrame(() => {
    deltaRaf = null;
    const chunk = pendingDelta;
    pendingDelta = "";
    if (!chunk) return;
    bubbleText.textContent = (bubbleText.textContent + chunk).slice(0, 4000);
    bubble.hidden = false;
    scheduleStageHeight();
  });
}

/** One quiet status line under the answer — no chip clutter. */
function pushNotice(text, tone = "note", ttl = 8000) {
  const value = (text || "").trim();
  if (!value) return;

  // Collapse duplicate notifications (e.g. Python + Rust both
  // reporting the backend exit) into a single line.
  const now = Date.now();
  if (value === lastNoticeText && now - lastNoticeAt < 2000) return;
  lastNoticeText = value;
  lastNoticeAt = now;

  clearTimeout(noticeClearTimer);
  bubbleNotice.className = `bubble-notice ${tone}`;
  bubbleNotice.textContent = value;
  bubbleNotice.hidden = false;
  bubble.hidden = false;
  scheduleStageHeight();

  noticeClearTimer = setTimeout(() => {
    bubbleNotice.classList.add("leaving");
    setTimeout(() => {
      bubbleNotice.hidden = true;
      bubbleNotice.classList.remove("leaving");
      bubbleNotice.textContent = "";
      bubble.hidden = !bubbleText.textContent;
      scheduleStageHeight();
    }, 220);
  }, ttl);
}

// Backwards-compatible alias used across the event handlers.
const pushChip = pushNotice;

/* ------------------------------------------------------------
   Event handling
   ------------------------------------------------------------ */
function handleAssistantEvent(msg) {
  if (!msg || typeof msg !== "object") return;

  switch (msg.type) {
    case "status": {
      const label = (msg.label || "").trim();
      if (msg.state === "stopped") {
        setState("stopped");
        break;
      }
      if (label && label !== "Bridge ready") pushChip(label, "note", 4000);
      break;
    }

    case "transcript": {
      curTranscript = msg.text || "";
      setState("thinking");
      setBubbleText("");
      break;
    }

    case "assistant_delta": {
      setState("speaking");
      queueDelta(msg.text || "");
      break;
    }

    case "assistant_final": {
      setBubbleText(msg.text || "");
      clearTimeout(bubbleClearTimer);
      bubbleClearTimer = setTimeout(() => {
        setBubbleText("");
      }, 15000);
      break;
    }

    case "tool_result":
    case "reminder":
    case "timer_update": {
      pushChip(msg.display || msg.text || "Update", msg.type === "reminder" || msg.tool === "set_timer" ? "timer" : "note");
      break;
    }

    case "error": {
      pushChip(msg.message || "Backend error", "error", 12000);
      break;
    }
  }
}

/* ------------------------------------------------------------
   Bridge startup / watchdog / reconnect
   ------------------------------------------------------------ */
let reconnecting = false;
let bridgeIsHealthy = false;

async function selectAndPersistModelPath(invoke, { startup = false } = {}) {
  try {
    const picked = await invoke("pick_model_file");
    if (!picked?.path) {
      if (startup && !picked?.cancelled) {
        pushChip("Please select a GGUF model to continue.", "error", 9000);
      }
      return false;
    }

    const saved = await invoke("set_model_path", { path: picked.path });
    if (!saved?.ok) throw new Error(saved?.error || "Unable to save model path");

    pushChip("Model saved — restarting…", "note", 5000);
    return true;
  } catch (error) {
    pushChip(String(error), "error", 9000);
    return false;
  }
}

function startWatchdog() {
  if (bridgeWatchdogTimer) clearInterval(bridgeWatchdogTimer);
  bridgeWatchdogTimer = setInterval(async () => {
    if (!bridgeIsHealthy) return;
    try {
      const health = await tauriInvoke("health");
      if (!health?.ok) throw new Error("no reply");
    } catch {
      bridgeIsHealthy = false;
      setState("stopped");
      pushChip("Bridge lost — click to reconnect", "error", 9000);
    }
  }, 30000);
}

async function attemptReconnect() {
  if (!tauriInvoke || reconnecting) return;
  reconnecting = true;
  pushChip("Reconnecting…", "note", 5000);
  try {
    let healthy = false;
    try {
      const health = await tauriInvoke("health");
      healthy = Boolean(health?.ok);
    } catch {
      healthy = false;
    }

    if (!healthy) {
      const restart = await tauriInvoke("restart_backend");
      if (!restart?.ok) throw new Error(restart?.error || "Could not restart backend");
    }

    const session = await tauriInvoke("start_session", { mode: "voice" });
    if (!session?.ok) throw new Error(session?.error || "Could not start assistant session");

    bridgeIsHealthy = true;
    setState("idle");
    pushChip("Connected", "note", 4000);
    startWatchdog();
  } catch (error) {
    bridgeIsHealthy = false;
    setState("stopped");
    pushChip(String(error?.message || error), "error", 9000);
  } finally {
    reconnecting = false;
  }
}

async function initTauriBridge() {
  if (!tauriInvoke || !tauriGlobal?.event?.listen) {
    setState("idle");
    pushChip("Preview mode — run inside Tauri to connect.", "note", 8000);
    return;
  }

  await tauriGlobal.event.listen("assistant-event", (event) => {
    handleAssistantEvent(event.payload);
  });

  try {
    const settingsResult = await tauriInvoke("get_settings");
    const hasModelPath = Boolean(settingsResult?.has_model_path);
    const modelExists = Boolean(settingsResult?.model_exists);

    if (!hasModelPath || !modelExists) {
      const reason = hasModelPath ? "Configured model file not found" : "Select your GGUF model";
      pushChip(reason, "note", 9000);

      const selected = await selectAndPersistModelPath(tauriInvoke, { startup: true });
      if (!selected) return;
    }

    const info = await tauriInvoke("health");
    if (!info?.running) {
      const sessionResult = await tauriInvoke("start_session", { mode: "voice" });
      if (sessionResult?.error === "no_model") {
        pushChip("No model file found. Select a .gguf model.", "error", 9000);
        const selected = await selectAndPersistModelPath(tauriInvoke, { startup: true });
        if (!selected) return;
        await new Promise((resolve) => setTimeout(resolve, 500));
        const retry = await tauriInvoke("start_session", { mode: "voice" });
        if (!retry?.running) {
          pushChip("Assistant failed to start after model selection.", "error", 9000);
          return;
        }
      }
    }

    bridgeIsHealthy = true;
    setState("idle");
    pushChip("Connected", "note", 4000);
    startWatchdog();
  } catch (error) {
    setState("stopped");
    pushChip(String(error), "error", 9000);
  }
}

/* ------------------------------------------------------------
   Wiring
   ------------------------------------------------------------ */

// The window is bottom-anchored, so opening/closing the composer
// (or menu) shifts the pill under the cursor mid-click. A plain
// `click` then lands on a different element and the action is
// lost — so actions that change the window height fire on press.
function onPress(element, handler) {
  if (!element) return;
  element.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    handler(event);
  });
}

function onTalkTrigger() {
  if (state === "stopped") {
    attemptReconnect();
    return;
  }
  setComposerOpen(false);
  if (state === "listening") {
    setState("idle");
    return;
  }
  setState("listening");
}

onPress(orbShell, onTalkTrigger);
onPress(micBtn, onTalkTrigger);

orbShell.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    onTalkTrigger();
  }
});

onPress(bubble, () => {
  clearTimeout(bubbleClearTimer);
  setBubbleText("");
});

if (changeModelBtn) {
  changeModelBtn.addEventListener("click", async () => {
    if (!tauriInvoke) {
      pushChip("Model picker is only available in the desktop app.", "note", 6000);
      return;
    }
    const changed = await selectAndPersistModelPath(tauriInvoke, { startup: false });
    if (!changed) return;
    try {
      await tauriInvoke("stop_session");
      await tauriInvoke("start_session", { mode: "voice" });
      pushChip("Model switched", "note", 4000);
    } catch (error) {
      pushChip(String(error), "error", 9000);
    }
  });
}

setState("idle");
initTauriBridge();
scheduleStageHeight();

/* ------------------------------------------------------------
   Type-to-Anantum composer
   ------------------------------------------------------------ */
/* Logical open state. A panel can still be on screen (sliding away)
   after being closed, so DOM `hidden` is not a reliable toggle source. */
let composerOpen = false;
let menuOpen = false;

function setComposerOpen(open) {
  if (open === composerOpen) return;
  composerOpen = open;
  if (typeBtn) {
    typeBtn.setAttribute("aria-expanded", String(open));
    typeBtn.title = open ? "Minimize typing" : "Type a request";
  }
  if (open) {
    composerInput.value = "";
    showPanel(composer);
    setTimeout(() => composerInput.focus(), 40);
  } else {
    hidePanel(composer);
  }
}

async function submitComposer() {
  const text = composerInput.value.trim();
  if (!text) return;
  if (!tauriInvoke) {
    pushChip("Typing needs the desktop app.", "note", 6000);
    return;
  }

  composerInput.value = "";
  clearTimeout(bubbleClearTimer);
  setComposerOpen(false);
  try {
    const result = await tauriInvoke("input_text", { text });
    if (result && result.ok === false) {
      pushChip(result.error || "Request failed", "error", 9000);
    }
  } catch (error) {
    pushChip(String(error), "error", 9000);
  }
}

onPress(typeBtn, () => setComposerOpen(!composerOpen));
onPress(composerSend, submitComposer);

composerInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    submitComposer();
  } else if (event.key === "Escape") {
    event.preventDefault();
    setComposerOpen(false);
  }
});

// Clicking anywhere else in the widget minimizes the type tab.
document.addEventListener("pointerdown", (event) => {
  if (!composerOpen) return;
  if (composer.contains(event.target) || typeBtn?.contains(event.target)) return;
  setComposerOpen(false);
});

/* ------------------------------------------------------------
   Right-click icon controls
   ------------------------------------------------------------ */
function setMenuOpen(open) {
  if (open === menuOpen) return;
  menuOpen = open;
  if (open) showPanel(menu);
  else hidePanel(menu);
}

widget.addEventListener("contextmenu", (event) => {
  event.preventDefault();
  setMenuOpen(!menuOpen);
});

document.addEventListener("click", (event) => {
  if (menuOpen && !menu.contains(event.target)) setMenuOpen(false);
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    setMenuOpen(false);
    setComposerOpen(false);
  }
});

onPress(document.getElementById("menuModel"), () => {
  setMenuOpen(false);
  changeModelBtn.click();
});

onPress(document.getElementById("menuRestart"), async () => {
  setMenuOpen(false);
  if (!tauriInvoke) return;
  pushChip("Restarting backend…", "note", 5000);
  try {
    await tauriInvoke("restart_backend");
    await tauriInvoke("start_session", { mode: "voice" });
    bridgeIsHealthy = true;
    setState("idle");
    startWatchdog();
    pushChip("Backend restarted", "note", 4000);
  } catch (error) {
    pushChip(String(error), "error", 9000);
  }
});

onPress(document.getElementById("menuQuit"), async () => {
  setMenuOpen(false);
  try {
    await tauriInvoke?.("stop_session");
  } catch {
    /* best effort */
  }
  tauriWindow?.getCurrentWindow?.()?.close?.();
});

/* ------------------------------------------------------------
   Pause work while the window is hidden
   ------------------------------------------------------------ */
document.addEventListener("visibilitychange", () => {
  document.body.classList.toggle("is-hidden", document.hidden);
  if (!siriWave) return;
  try {
    if (document.hidden) {
      siriWave.stop?.();
    } else if (isWaveState(state)) {
      siriWave.start?.();
    }
  } catch {
    /* ignore */
  }
});

window.AssistantWidget = {
  setState,
  setBubbleText,
  pushChip,
  handleAssistantEvent,
};
