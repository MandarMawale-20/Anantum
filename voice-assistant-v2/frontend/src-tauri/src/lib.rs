use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{mpsc, Arc, Mutex, atomic::{AtomicBool, AtomicU64, Ordering}};
use std::thread;
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager, PhysicalPosition, Runtime, State};
use log::warn;

struct BridgeProcess {
    child: Arc<Mutex<Child>>,
    stdin: Arc<Mutex<ChildStdin>>,
    pending: Arc<Mutex<HashMap<u64, mpsc::Sender<Value>>>>,
    next_id: Arc<Mutex<u64>>,
    alive: Arc<AtomicBool>,
}

impl BridgeProcess {
    fn resolve_backend_launch<R: Runtime>(
        app: &AppHandle<R>,
        project_root: &Path,
    ) -> Result<Command, String> {
        if cfg!(debug_assertions) {
            let script_path = project_root.join("bridge").join("stdio_bridge.py");
            let mut cmd = Command::new(Self::resolve_python(project_root));
            cmd.arg(script_path).current_dir(project_root);
            return Ok(cmd);
        }

        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Ok(resource_dir) = app.path().resource_dir() {
            candidates.push(resource_dir.join("backend").join("assistant-backend.exe"));
            candidates.push(resource_dir.join("backend").join("assistant-backend"));
            candidates.push(
                resource_dir
                    .join("backend")
                    .join("assistant-backend")
                    .join("assistant-backend.exe"),
            );
        }

        if let Ok(env_path) = std::env::var("ANANTUM_BACKEND_EXE") {
            let path = PathBuf::from(env_path);
            if path.exists() {
                let mut cmd = Command::new(path);
                cmd.current_dir(project_root);
                return Ok(cmd);
            }
        }

        for candidate in &candidates {
            if candidate.exists() {
                let mut cmd = Command::new(candidate);
                cmd.current_dir(project_root);
                return Ok(cmd);
            }
        }

        let searched = candidates
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join("; ");
        Err(format!("Backend executable not found. Looked in: {searched}"))
    }

    /// Prefer an explicit interpreter, then a project virtualenv, then
    /// whatever `python` resolves to on PATH. Using the system Python
    /// instead of the venv was why the backend died on startup with
    /// "ModuleNotFoundError: No module named 'sounddevice'".
    fn resolve_python(project_root: &Path) -> String {
        if let Ok(explicit) = std::env::var("ANANTUM_PYTHON") {
            if !explicit.trim().is_empty() {
                return explicit;
            }
        }

        let candidates = [
            project_root.join(".venv").join("Scripts").join("python.exe"),
            project_root
                .join("..")
                .join(".venv")
                .join("Scripts")
                .join("python.exe"),
            project_root.join(".venv").join("bin").join("python"),
            project_root.join("..").join(".venv").join("bin").join("python"),
        ];

        for candidate in candidates {
            if candidate.exists() {
                return candidate.to_string_lossy().to_string();
            }
        }

        "python".to_string()
    }

    fn spawn<R: Runtime>(app: AppHandle<R>) -> Result<Self, String> {
        let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .ok_or_else(|| "Failed to resolve project root".to_string())?
            .to_path_buf();

        let mut command = Self::resolve_backend_launch(&app, &project_root)?;
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to start backend bridge: {e}"))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "Failed to open bridge stdin".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "Failed to open bridge stdout".to_string())?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| "Failed to open bridge stderr".to_string())?;

        let child = Arc::new(Mutex::new(child));
        let stdin = Arc::new(Mutex::new(stdin));
        let pending: Arc<Mutex<HashMap<u64, mpsc::Sender<Value>>>> = Arc::new(Mutex::new(HashMap::new()));
        let next_id = Arc::new(Mutex::new(1_u64));
        let alive = Arc::new(AtomicBool::new(true));
        let stderr_tail: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

        {
            let tail = Arc::clone(&stderr_tail);
            thread::spawn(move || {
                let reader = BufReader::new(stderr);
                for line in reader.lines() {
                    match line {
                        Ok(v) => {
                            let trimmed = v.trim();
                            if trimmed.is_empty() {
                                continue;
                            }
                            warn!("[backend stderr] {}", trimmed);
                            if let Ok(mut buffer) = tail.lock() {
                                buffer.push(trimmed.to_string());
                                let len = buffer.len();
                                if len > 30 {
                                    buffer.drain(0..len - 30);
                                }
                            }
                        }
                        Err(_) => break,
                    }
                }
            });
        }

        let pending_for_thread = Arc::clone(&pending);
        let app_for_thread = app.clone();
        let alive_for_thread = Arc::clone(&alive);
        let stderr_for_thread = Arc::clone(&stderr_tail);
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line_result in reader.lines() {
                let line = match line_result {
                    Ok(v) => v,
                    Err(_) => break,
                };

                let parsed: Value = match serde_json::from_str(&line) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                match parsed.get("type").and_then(Value::as_str) {
                    Some("event") => {
                        if let Some(payload) = parsed.get("event") {
                            let _ = app_for_thread.emit("assistant-event", payload.clone());
                        }
                    }
                    Some("response") => {
                        if let Some(id) = parsed.get("id").and_then(Value::as_u64) {
                            let maybe_sender = {
                                let mut guard = pending_for_thread.lock().ok();
                                guard.as_mut().and_then(|m| m.remove(&id))
                            };
                            if let Some(sender) = maybe_sender {
                                let payload = parsed.get("result").cloned().unwrap_or_else(|| json!({"ok": false, "error": "empty response"}));
                                let _ = sender.send(payload);
                            }
                        }
                    }
                    _ => {}
                }
            }
            alive_for_thread.store(false, Ordering::SeqCst);
            // Surface the backend's own error output so failures like
            // "ModuleNotFoundError: No module named 'sounddevice'" are
            // visible in the widget instead of a silent "Disconnected".
            let reason = stderr_for_thread
                .lock()
                .ok()
                .map(|buffer| {
                    let is_fatal = |line: &String| {
                        line.contains("Traceback")
                            || line.contains("Error")
                            || line.contains("error:")
                    };
                    buffer
                        .iter()
                        .rev()
                        .find(|line| is_fatal(line))
                        .or_else(|| buffer.last())
                        .cloned()
                })
                .flatten();

            if let Some(reason) = reason {
                let _ = app_for_thread.emit("assistant-event", json!({
                    "type": "error",
                    "message": reason
                }));
            }
            let _ = app_for_thread.emit("assistant-event", json!({
                "type": "status",
                "state": "stopped",
                "label": "Backend exited"
            }));
        });

        Ok(Self {
            child,
            stdin,
            pending,
            next_id,
            alive,
        })
    }

    fn request_with_timeout(&self, command: &str, args: Value, timeout: Duration) -> Result<Value, String> {
        if !self.alive.load(Ordering::SeqCst) {
            return Err("Backend process is not running".to_string());
        }

        let (tx, rx) = mpsc::channel::<Value>();

        let id = {
            let mut id_guard = self
                .next_id
                .lock()
                .map_err(|_| "Bridge id lock poisoned".to_string())?;
            let id = *id_guard;
            *id_guard += 1;
            id
        };

        {
            let mut pending = self
                .pending
                .lock()
                .map_err(|_| "Bridge pending lock poisoned".to_string())?;
            pending.insert(id, tx);
        }

        let packet = json!({
            "id": id,
            "command": command,
            "args": args,
        });
        let packet_line = format!("{}\n", packet);

        {
            let mut input = self
                .stdin
                .lock()
                .map_err(|_| "Bridge stdin lock poisoned".to_string())?;
            input
                .write_all(packet_line.as_bytes())
                .map_err(|e| format!("Failed to write bridge command: {e}"))?;
            input
                .flush()
                .map_err(|e| format!("Failed to flush bridge command: {e}"))?;
        }

        rx.recv_timeout(timeout)
            .map_err(|_| format!("Bridge command timed out: {command}"))
    }

    fn shutdown(&self) {
        if !self.alive.load(Ordering::SeqCst) {
            return;
        }
        let _ = self.request_with_timeout("shutdown", json!({}), Duration::from_secs(3));
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Runs a bridge request on a blocking thread so the Tauri
/// main/UI thread stays responsive ("Not Responding" freeze fix).
async fn bridge_call(
    app: AppHandle<tauri::Wry>,
    command: &'static str,
    args: Value,
    timeout: Duration,
) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let state: State<BridgeState> = app.state();
        let process = state.process()?;
        process.request_with_timeout(command, args, timeout)
    })
    .await
    .map_err(|e| format!("Bridge task failed: {e}"))?
}

struct BridgeState {
    process: Mutex<Option<Arc<BridgeProcess>>>,
}

impl BridgeState {
    fn process(&self) -> Result<Arc<BridgeProcess>, String> {
        self.process
            .lock()
            .map_err(|_| "Bridge state lock poisoned".to_string())?
            .as_ref()
            .cloned()
            .ok_or_else(|| "Backend process is not running".to_string())
    }

    /// Shuts down the current backend (if any) and spawns a fresh one.
    fn respawn<R: Runtime>(&self, app: &AppHandle<R>) -> Result<(), String> {
        let mut guard = self
            .process
            .lock()
            .map_err(|_| "Bridge state lock poisoned".to_string())?;

        if let Some(old) = guard.as_ref() {
            old.shutdown();
            app.emit(
                "assistant-event",
                json!({
                    "type": "status",
                    "state": "idle",
                    "label": "Backend restarting"
                }),
            )
            .ok();
        }
        *guard = None;

        let fresh = BridgeProcess::spawn(app.clone())?;
        *guard = Some(Arc::new(fresh));
        Ok(())
    }
}

#[tauri::command]
async fn health(app: AppHandle<tauri::Wry>) -> Result<Value, String> {
    bridge_call(app, "health", json!({}), Duration::from_secs(4)).await
}

#[tauri::command]
async fn get_settings(app: AppHandle<tauri::Wry>) -> Result<Value, String> {
    bridge_call(app, "get_settings", json!({}), Duration::from_secs(4)).await
}

#[tauri::command]
async fn start_session(mode: String, app: AppHandle<tauri::Wry>) -> Result<Value, String> {
    // Model initialization can be slow; run well clear of any UI freeze.
    bridge_call(app, "start_session", json!({ "mode": mode }), Duration::from_secs(60)).await
}

#[tauri::command]
async fn stop_session(app: AppHandle<tauri::Wry>) -> Result<Value, String> {
    bridge_call(app, "stop_session", json!({}), Duration::from_secs(10)).await
}

#[tauri::command]
async fn input_text(text: String, app: AppHandle<tauri::Wry>) -> Result<Value, String> {
    bridge_call(app, "input_text", json!({ "text": text }), Duration::from_secs(35)).await
}

#[tauri::command]
async fn set_model_path(path: String, app: AppHandle<tauri::Wry>) -> Result<Value, String> {
    bridge_call(app, "set_model_path", json!({ "path": path }), Duration::from_secs(4)).await
}

/// Sliding-window state for the widget.
///
/// The window's BOTTOM edge is pinned to the corner of the screen and its
/// top edge is what moves, so the stage can grow/shrink without the pill
/// ever being snapped around: the pill lives at the window's top, and only
/// the animated slide changes its position on screen.
#[derive(Default)]
struct StageState {
    /// Bumped for every requested transition so an in-flight slide can
    /// abandon itself as soon as a newer one arrives.
    token: AtomicU64,
    /// True while a slide is running. While sliding, the last y we
    /// commanded is fresher than the position Windows reports.
    sliding: AtomicBool,
    /// Last vertical position (physical px) we commanded.
    y: Mutex<Option<i32>>,
}

#[tauri::command]
async fn apply_stage(app: AppHandle<tauri::Wry>, width: f64, height: f64) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || slide_stage(&app, width, height))
        .await
        .map_err(|e| format!("Stage task failed: {e}"))?
}

fn slide_stage(app: &AppHandle<tauri::Wry>, width: f64, height: f64) -> Result<Value, String> {
    let window = app
        .get_webview_window("assistant")
        .ok_or_else(|| "assistant window not found".to_string())?;
    let stage: State<StageState> = app.state();

    let scale: f64 = window.scale_factor().unwrap_or(1.0);
    let w = (width * scale).round().max(1.0) as u32;
    let target_h = (height * scale).round().max(1.0) as u32;

    let current = window.outer_position().ok().zip(window.outer_size().ok());

    let (x, current_h, bottom) = match current {
        Some((pos, size)) => (pos.x, size.height, pos.y + size.height as i32),
        None => {
            let monitor = window
                .current_monitor()
                .ok()
                .flatten()
                .ok_or_else(|| "no monitor available".to_string())?;
            let monitor_size = monitor.size();
            let margin = (18.0 * scale).round() as i32;
            (
                monitor_size.width as i32 - w as i32 - margin,
                target_h,
                monitor_size.height as i32 - margin,
            )
        }
    };

    let target_y = (bottom - target_h as i32).max(0);

    // While a slide is running, trust our own last commanded y: the OS may
    // still be reporting the position from before the previous step. Once
    // idle, the reported position wins so a user drag is never undone.
    let start_y = if stage.sliding.load(Ordering::SeqCst) {
        stage
            .y
            .lock()
            .unwrap()
            .unwrap_or_else(|| current.map(|(pos, _)| pos.y).unwrap_or(target_y))
    } else {
        current.map(|(pos, _)| pos.y).unwrap_or(target_y)
    };

    let token = stage.token.fetch_add(1, Ordering::SeqCst) + 1;
    let growing = target_h > current_h;

    // Growing: resize first. The extra height opens BELOW the pill (the
    // pill is glued to the window's top) and is then pulled on screen by
    // the slide, so nothing ever snaps.
    // Shrinking: slide down first while everything is still painted, and
    // only resize once the panel has dropped out of sight.
    if growing {
        window
            .set_size(tauri::PhysicalSize::new(w, target_h))
            .map_err(|e| format!("Failed to resize stage: {e}"))?;
    }

    let distance = (target_y - start_y).abs();
    let duration_ms: u64 = if distance < 2 { 0 } else { 190 };
    let steps = (duration_ms / 12).max(1) as u32;
    let step_delay = if duration_ms > 0 {
        duration_ms / steps as u64
    } else {
        0
    };

    stage.sliding.store(duration_ms > 0, Ordering::SeqCst);

    let mut last_y = start_y;
    for step in 1..=steps {
        if stage.token.load(Ordering::SeqCst) != token {
            // A newer transition owns the window now; leave it alone.
            return Ok(json!({ "ok": true, "superseded": true }));
        }

        let progress = step as f64 / steps as f64;
        let eased = 1.0 - (1.0 - progress).powi(3); // ease-out cubic
        let y = start_y + ((target_y - start_y) as f64 * eased).round() as i32;

        if y != last_y {
            window
                .set_position(tauri::PhysicalPosition::new(x, y.max(0)))
                .map_err(|e| format!("Failed to slide stage: {e}"))?;
            last_y = y;
            *stage.y.lock().unwrap() = Some(y);
        }

        if step_delay > 0 {
            thread::sleep(Duration::from_millis(step_delay));
        }
    }

    if !growing {
        window
            .set_size(tauri::PhysicalSize::new(w, target_h))
            .map_err(|e| format!("Failed to resize stage: {e}"))?;
    }

    *stage.y.lock().unwrap() = Some(target_y);
    stage.sliding.store(false, Ordering::SeqCst);

    Ok(json!({ "ok": true }))
}

#[tauri::command]
async fn restart_backend(app: AppHandle<tauri::Wry>) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let state: State<BridgeState> = app.state();
        state.respawn(&app)
    })
    .await
    .map_err(|e| format!("Backend restart task failed: {e}"))??;
    Ok(json!({ "ok": true }))
}

#[tauri::command]
async fn pick_model_file() -> Result<Value, String> {
    // Native dialogs are blocking; keep them off the UI thread too.
    tauri::async_runtime::spawn_blocking(|| {
        let picked = rfd::FileDialog::new()
            .add_filter("GGUF model", &["gguf"])
            .set_title("Select GGUF model")
            .pick_file();

        Ok(match picked {
            Some(path) => json!({ "ok": true, "path": path.to_string_lossy() }),
            None => json!({ "ok": false, "cancelled": true }),
        })
    })
    .await
    .map_err(|e| format!("File dialog task failed: {e}"))?
}

fn position_bottom_right<R: Runtime>(app: &AppHandle<R>) {
    if let Some(window) = app.get_webview_window("assistant") {
        if let Ok(Some(monitor)) = window.current_monitor() {
            let scale = monitor.scale_factor();
            let monitor_size = monitor.size();
            let margin = (18.0 * scale) as i32;
            let win_w = (430.0 * scale) as i32;
            // 4px stage pad + 60px pill + 4px pad, all scaled
            let base_h = (68.0 * scale) as i32;
            let x = monitor_size.width as i32 - win_w - margin;
            let y = monitor_size.height as i32 - base_h - margin;
            let _ = window.set_position(PhysicalPosition::new(x, y));
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(StageState::default())
        .setup(|app| {
            position_bottom_right(app.handle());

            let process = BridgeProcess::spawn(app.handle().clone())
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
            app.manage(BridgeState {
                process: Mutex::new(Some(Arc::new(process))),
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            health,
            get_settings,
            start_session,
            stop_session,
            input_text,
            set_model_path,
            pick_model_file,
            restart_backend,
            apply_stage,
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                // Never block the UI thread while shutting the
                // backend down (it can take seconds).
                if let Ok(process) = window.state::<BridgeState>().process() {
                    std::thread::spawn(move || process.shutdown());
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
