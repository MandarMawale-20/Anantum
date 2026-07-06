import logging
import os
import threading
from typing import Any, Optional
from pathlib import Path

from core.assistant import Anantum
from config.settings import CONFIG
from skills.timers import TimerManager

logger = logging.getLogger(__name__)


class AssistantRuntime:
    def __init__(self, event_sink):
        self._event_sink = event_sink
        self._assistant: Optional[Anantum] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._mode: str = "voice"
        self._lock_fd: Optional[int] = None

    def _acquire_session_lock(self) -> bool:
        """Prevent double-launch by creating a lock file with exclusive access."""
        lock_path = CONFIG.data_dir / "anantum.lock"
        try:
            if os.name == "nt":
                import msvcrt
                fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                except OSError:
                    os.close(fd)
                    return False
                self._lock_fd = fd
            else:
                import fcntl
                fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    os.close(fd)
                    return False
                self._lock_fd = fd
            return True
        except Exception as e:
            logger.debug("Session lock acquire failed: %s", e)
            return False

    def _release_session_lock(self) -> None:
        if self._lock_fd is not None:
            try:
                os.close(self._lock_fd)
            except Exception:
                pass
            self._lock_fd = None
        try:
            (CONFIG.data_dir / "anantum.lock").unlink(missing_ok=True)
        except Exception:
            pass

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def mode(self) -> str:
        return self._mode

    def _emit(self, event: dict[str, Any]) -> None:
        try:
            self._event_sink(event)
        except Exception:
            logger.debug("Failed to emit bridge event", exc_info=True)

    def start(self, mode: str = "voice") -> None:
        with self._lock:
            if self.running:
                return
            if not self._acquire_session_lock():
                logger.warning("Another instance may already be running")
            self._mode = mode
            self._assistant = Anantum(event_sink=self._emit)
            TimerManager.set_event_callback(self._emit)
            self._thread = threading.Thread(
                target=self._run_assistant,
                name="assistant-runtime",
                daemon=True,
            )
            self._thread.start()

    def _run_assistant(self) -> None:
        assert self._assistant is not None
        try:
            if self._mode == "text":
                self._assistant.run_text()
            else:
                self._assistant.run_voice()
        except Exception as exc:
            logger.exception("Assistant runtime crashed: %s", exc)
            self._emit({"type": "error", "message": f"Assistant runtime crashed: {exc}"})
        finally:
            self._emit({"type": "status", "state": "stopped", "label": "Stopped"})

    def stop(self) -> None:
        with self._lock:
            if self._assistant is not None:
                self._assistant.stop()
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=2.0)
            self._release_session_lock()

    def process_text(self, text: str) -> str:
        with self._lock:
            if not self.running or self._assistant is None:
                raise RuntimeError("Assistant is not running")
            if self._mode != "text":
                raise RuntimeError("Text input endpoint is only available in text runtime mode")
            return self._assistant.process_text_once(text)

def execute_command(runtime: AssistantRuntime, command: str, args: dict[str, Any]) -> dict[str, Any]:
    if command == "health":
        model_path = Path(CONFIG.llm_model)
        return {
            "ok": True,
            "running": runtime.running,
            "mode": runtime.mode,
            "model_path": str(model_path),
            "model_exists": model_path.exists(),
        }

    if command == "get_settings":
        model_path = Path(CONFIG.llm_model)
        settings = CONFIG.to_user_settings_dict()
        return {
            "ok": True,
            "settings": settings,
            "settings_path": str(CONFIG.user_settings_file),
            "has_model_path": bool(str(settings.get("model_path", "")).strip()),
            "model_exists": model_path.exists(),
        }

    if command == "set_model_path":
        path_value = str(args.get("path", "")).strip()
        if not path_value:
            return {"ok": False, "error": "'path' is required"}

        model_path = Path(path_value).expanduser()
        if model_path.suffix.lower() != ".gguf":
            return {"ok": False, "error": "Model file must be a .gguf file"}
        if not model_path.exists():
            return {"ok": False, "error": "Selected model file does not exist"}

        CONFIG.llm_model = str(model_path)
        ok, error = CONFIG.save_user_settings()
        if not ok:
            return {"ok": False, "error": f"Failed to save settings: {error}"}

        return {
            "ok": True,
            "settings": {
                "model_path": str(model_path),
                "model_exists": True,
            },
            "requires_restart": runtime.running,
        }

    if command == "start_session":
        mode = str(args.get("mode", "voice")).lower().strip()
        if mode not in {"voice", "text"}:
            return {"ok": False, "error": "mode must be 'voice' or 'text'"}
        model_path = Path(CONFIG.llm_model)
        if not model_path.exists():
            return {
                "ok": False,
                "error": "no_model",
                "message": "No model file found. Please select a .gguf model to continue."
            }
        runtime.start(mode=mode)
        return {"ok": True, "running": runtime.running, "mode": mode}

    if command == "stop_session":
        runtime.stop()
        return {"ok": True, "running": runtime.running}

    if command == "input_text":
        text = str(args.get("text", "")).strip()
        if not text:
            return {"ok": False, "error": "'text' is required"}
        try:
            reply = runtime.process_text(text)
            return {"ok": True, "reply": reply}
        except RuntimeError as exc:
            return {"ok": False, "error": str(exc)}

    if command == "shutdown":
        runtime.stop()
        return {"ok": True}

    return {"ok": False, "error": f"Unknown command '{command}'"}
