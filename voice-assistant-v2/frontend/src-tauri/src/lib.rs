use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{mpsc, Arc, Mutex, atomic::{AtomicBool, Ordering}};
use std::thread;
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager, PhysicalPosition, Runtime, State};
use log::{error, info, warn};

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
            let mut cmd = Command::new("python");
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

        thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                match line {
                    Ok(v) => warn!("[backend stderr] {}", v),
                    Err(_) => break,
                }
            }
        });

        let pending_for_thread = Arc::clone(&pending);
        let app_for_thread = app.clone();
        let alive_for_thread = Arc::clone(&alive);
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

    fn request(&self, command: &str, args: Value) -> Result<Value, String> {
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

        rx.recv_timeout(Duration::from_secs(20))
            .map_err(|_| format!("Bridge command timed out: {command}"))
    }

    fn shutdown(&self) {
        if !self.alive.load(Ordering::SeqCst) {
            return;
        }
        let _ = self.request("shutdown", json!({}));
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

struct BridgeState {
    process: Arc<BridgeProcess>,
}

#[tauri::command]
fn health(state: State<'_, BridgeState>) -> Result<Value, String> {
    state.process.request("health", json!({}))
}

#[tauri::command]
fn get_settings(state: State<'_, BridgeState>) -> Result<Value, String> {
    state.process.request("get_settings", json!({}))
}

#[tauri::command]
fn start_session(mode: String, state: State<'_, BridgeState>) -> Result<Value, String> {
    state.process.request("start_session", json!({ "mode": mode }))
}

#[tauri::command]
fn stop_session(state: State<'_, BridgeState>) -> Result<Value, String> {
    state.process.request("stop_session", json!({}))
}

#[tauri::command]
fn input_text(text: String, state: State<'_, BridgeState>) -> Result<Value, String> {
    state.process.request("input_text", json!({ "text": text }))
}

#[tauri::command]
fn set_model_path(path: String, state: State<'_, BridgeState>) -> Result<Value, String> {
    state.process
        .request("set_model_path", json!({ "path": path }))
}

#[tauri::command]
fn pick_model_file() -> Result<Value, String> {
    let picked = rfd::FileDialog::new()
        .add_filter("GGUF model", &["gguf"])
        .set_title("Select GGUF model")
        .pick_file();

    Ok(match picked {
        Some(path) => json!({ "ok": true, "path": path.to_string_lossy() }),
        None => json!({ "ok": false, "cancelled": true }),
    })
}

fn position_bottom_right<R: Runtime>(app: &AppHandle<R>) {
    if let Some(window) = app.get_webview_window("assistant") {
        if let Ok(Some(monitor)) = window.current_monitor() {
            let monitor_size = monitor.size();
            let margin = 24;
            let win_w = 366_u32;
            let win_h = 76_u32;
            let x = monitor_size.width.saturating_sub(win_w).saturating_sub(margin) as i32;
            let y = monitor_size.height.saturating_sub(win_h).saturating_sub(margin) as i32;
            let _ = window.set_position(PhysicalPosition::new(x, y));
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            position_bottom_right(app.handle());

            let process = BridgeProcess::spawn(app.handle().clone())
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
            app.manage(BridgeState {
                process: Arc::new(process),
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
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                let state = window.state::<BridgeState>();
                state.process.shutdown();
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
