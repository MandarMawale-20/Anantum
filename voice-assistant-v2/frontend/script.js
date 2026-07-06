const widget = document.getElementById("assistantWidget");
const orbShell = document.getElementById("orbShell");
const siriContainer = document.getElementById("siri-container");
const assistantText = document.getElementById("assistantText");
const assistantMeta = document.getElementById("assistantMeta");
const eventChip = document.getElementById("eventChip");
const reminderStack = document.getElementById("reminderStack");
const changeModelBtn = document.getElementById("changeModelBtn");

let state = "idle";
let streamBuffer = "";
let chipTimer = null;
let bridgeWatchdogTimer = null;
const REMINDER_LIMIT = 4;
let siriWave = null;
let bridgeIsHealthy = false;

const stateLabels = {
  idle: "Idle",
  listening: "Listening",
  thinking: "Thinking",
  speaking: "Speaking",
  stopped: "Disconnected",
};

const stateConfig = {
  idle: { amplitude: 0.1, speed: 0.05 },
  listening: { amplitude: 1.5, speed: 0.2 },
  thinking: { amplitude: 0.9, speed: 0.11 },
  speaking: { amplitude: 2.0, speed: 0.15 },
  stopped: { amplitude: 0.05, speed: 0.02 },
};

const widgetStateClasses = ["is-idle", "is-listening", "is-thinking", "is-speaking", "is-stopped"];

function initSiriWave() {
  if (!window.SiriWave || !siriContainer) {
    return;
  }

  siriWave = new window.SiriWave({
    container: siriContainer,
    width: 100,
    height: 50,
    style: "ios9",
    amplitude: stateConfig.idle.amplitude,
    speed: stateConfig.idle.speed,
    autostart: true,
  });
}

function setState(nextState) {
  if (!stateConfig[nextState]) {
    return;
  }

  state = nextState;
  widget.classList.remove(...widgetStateClasses);
  widget.classList.add(`is-${nextState}`);
  assistantMeta.textContent = stateLabels[nextState] || "Idle";

  if (siriWave) {
    const cfg = stateConfig[nextState] || stateConfig.idle;
    siriWave.setAmplitude(cfg.amplitude);
    siriWave.setSpeed(cfg.speed);
  }

  if (nextState === "stopped") {
    assistantText.textContent = "Bridge disconnected";
    showEventChip("Click to reconnect");
  }
}

function setText(text) {
  const clean = (text || "").trim();
  assistantText.textContent = clean || "How can I help?";
}

function appendText(delta) {
  streamBuffer += delta || "";
  setText(streamBuffer);
}

function resetStream() {
  streamBuffer = "";
}

function showEventChip(text) {
  if (!text || !text.trim()) {
    eventChip.hidden = true;
    return;
  }

  eventChip.textContent = text;
  eventChip.hidden = false;

  if (chipTimer) {
    clearTimeout(chipTimer);
  }

  chipTimer = setTimeout(() => {
    eventChip.hidden = true;
  }, 3800);
}

function pushReminderCard(text, tone = "note") {
  const value = (text || "").trim();
  if (!value) {
    return;
  }

  const card = document.createElement("div");
  card.className = `reminder-card reminder-${tone}`;
  card.innerHTML = `<span class="dot" aria-hidden="true"></span><span>${value}</span>`;
  reminderStack.prepend(card);

  while (reminderStack.children.length > REMINDER_LIMIT) {
    reminderStack.removeChild(reminderStack.lastElementChild);
  }

  setTimeout(() => {
    card.style.opacity = "0";
    card.style.transform = "translateY(-3px) scale(0.98)";
    setTimeout(() => card.remove(), 240);
  }, 7600);
}

function handleAssistantEvent(msg) {
  if (!msg || typeof msg !== "object") {
    return;
  }

  if (msg.type === "status") {
    if (msg.state === "stopped") {
      bridgeIsHealthy = false;
    }
    setState(msg.state || "idle");
    if (msg.label) {
      showEventChip(msg.label);
    }
    return;
  }

  if (msg.type === "transcript") {
    setState("thinking");
    setText(msg.text || "");
    return;
  }

  if (msg.type === "assistant_delta") {
    setState("speaking");
    appendText(msg.text || "");
    return;
  }

  if (msg.type === "assistant_final") {
    setText(msg.text || "");
    resetStream();
    setState("idle");
    return;
  }

  if (msg.type === "tool_result" || msg.type === "reminder" || msg.type === "timer_update") {
    showEventChip(msg.display || msg.text || "Update");
    if (msg.display) {
      setText(msg.display);
    }
    const tone = msg.type === "reminder" || msg.tool === "set_timer" ? "timer" : "note";
    pushReminderCard(msg.display || msg.text || "Update", tone);
    return;
  }

  if (msg.type === "error") {
    showEventChip("Backend error");
    if (msg.message) {
      pushReminderCard(msg.message, "error");
    }
    return;
  }
}

async function selectAndPersistModelPath(invoke, { startup = false } = {}) {
  try {
    const picked = await invoke("pick_model_file");
    if (!picked?.path) {
      if (startup && !picked?.cancelled) {
        showEventChip("Model required");
        pushReminderCard("Please select a GGUF model to continue.", "error");
      }
      return false;
    }

    const saved = await invoke("set_model_path", { path: picked.path });
    if (!saved?.ok) {
      throw new Error(saved?.error || "Unable to save model path");
    }

    showEventChip("Model saved");
    pushReminderCard("Model path updated. Restarting assistant...", "note");
    return true;
  } catch (error) {
    showEventChip("Model update failed");
    pushReminderCard(String(error), "error");
    return false;
  }
}

async function initTauriBridge() {
  const tauriCore = window.__TAURI__?.core;
  const tauriEvent = window.__TAURI__?.event;

  if (!tauriCore?.invoke || !tauriEvent?.listen) {
    setState("idle");
    showEventChip("Preview mode");
    pushReminderCard("Run inside Tauri to connect assistant runtime.", "note");
    return;
  }

  const invoke = tauriCore.invoke;
  const listen = tauriEvent.listen;

  await listen("assistant-event", (event) => {
    handleAssistantEvent(event.payload);
  });

  try {
    const settingsResult = await invoke("get_settings");
    const hasModelPath = Boolean(settingsResult?.has_model_path);
    const modelExists = Boolean(settingsResult?.model_exists);

    if (!hasModelPath || !modelExists) {
      const reason = hasModelPath ? "Configured model file not found" : "Select your GGUF model";
      showEventChip(reason);
      pushReminderCard(reason, "note");

      const selected = await selectAndPersistModelPath(invoke, { startup: true });
      if (!selected) {
        return;
      }
    }

    const info = await invoke("health");
    if (!info?.running) {
      const sessionResult = await invoke("start_session", { mode: "voice" });
      if (sessionResult?.error === "no_model") {
        showEventChip("Model required");
        pushReminderCard("No model file found. Please select a .gguf model.", "error");
        const selected = await selectAndPersistModelPath(invoke, { startup: true });
        if (!selected) return;
        await new Promise((resolve) => setTimeout(resolve, 500));
        const retry = await invoke("start_session", { mode: "voice" });
        if (!retry?.running) return;
      }
    }
    bridgeIsHealthy = true;
    showEventChip("Connected");

    if (bridgeWatchdogTimer) clearInterval(bridgeWatchdogTimer);
    bridgeWatchdogTimer = setInterval(async () => {
      if (!bridgeIsHealthy) return;
      try {
        const health = await invoke("health");
        if (!health?.ok) throw new Error("no reply");
      } catch {
        bridgeIsHealthy = false;
        setState("stopped");
        pushReminderCard("Bridge lost — click to reconnect", "error");
      }
    }, 60000);
  } catch (error) {
    setState("idle");
    showEventChip("Bridge error");
    pushReminderCard(String(error), "error");
  }
}

if (changeModelBtn) {
  changeModelBtn.addEventListener("click", async () => {
    const tauriCore = window.__TAURI__?.core;
    if (!tauriCore?.invoke) {
      pushReminderCard("Model picker is only available in the desktop app.", "note");
      return;
    }

    const invoke = tauriCore.invoke;
    const changed = await selectAndPersistModelPath(invoke, { startup: false });
    if (!changed) {
      return;
    }

    try {
      await invoke("stop_session");
      await invoke("start_session", { mode: "voice" });
      showEventChip("Model switched");
    } catch (error) {
      showEventChip("Restart failed");
      pushReminderCard(String(error), "error");
    }
  });
}

orbShell.addEventListener("click", () => {
  if (state === "listening") {
    setState("idle");
    setText("How can I help?");
    return;
  }

  setState("listening");
  setText("I am listening...");
});

orbShell.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    orbShell.click();
  }
});

setState("idle");
setText("How can I help?");
initSiriWave();
initTauriBridge();

window.AssistantWidget = {
  setState,
  setText,
  appendText,
  resetStream,
  showEventChip,
  handleAssistantEvent,
};

