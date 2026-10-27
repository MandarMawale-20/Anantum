import atexit
import json
import logging
import sys
import threading
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bridge.server import AssistantRuntime, execute_command

logging.basicConfig(level=logging.INFO, format="%(asctime)s [bridge] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Response writes come from the reader thread plus command workers.
_write_lock = threading.Lock()

# Commands that can block for seconds (model init, generation). They run off
# the stdin reader thread so a heartbeat/health request is still answered
# while an answer is being generated.
_TEXT_COMMANDS = {"input_text"}
_SESSION_COMMANDS = {"start_session", "stop_session", "shutdown"}

# Serializes text turns and session lifecycle changes respectively.
_text_lock = threading.Lock()
_session_lock = threading.Lock()


def _write_packet(packet: dict) -> None:
    line = json.dumps(packet, ensure_ascii=True) + "\n"
    with _write_lock:
        sys.stdout.write(line)
        sys.stdout.flush()


def _emit_event(event: dict) -> None:
    _write_packet({"type": "event", "event": event})


def _on_exit():
    """Signal frontend on crash so it can show 'Restart' instead of frozen UI."""
    try:
        _write_packet({"type": "event", "event": {
            "type": "status", "state": "stopped", "label": "Backend exited"
        }})
    except Exception:
        pass


def _respond(runtime: AssistantRuntime, req_id, command: str, args: dict) -> None:
    try:
        result = execute_command(runtime, command, args)
    except Exception as exc:
        logger.exception("Bridge command '%s' failed", command)
        result = {"ok": False, "error": str(exc)}
    _write_packet({"type": "response", "id": req_id, "result": result})


def _respond_async(runtime: AssistantRuntime, req_id, command: str, args: dict, lock: threading.Lock) -> None:
    def _work():
        with lock:
            _respond(runtime, req_id, command, args)

    threading.Thread(target=_work, name=f"bridge-{command}", daemon=True).start()


def run_stdio_bridge() -> None:
    atexit.register(_on_exit)
    runtime = AssistantRuntime(event_sink=_emit_event)
    _emit_event({"type": "status", "state": "idle", "label": "Bridge ready"})

    try:
        for line in sys.stdin:
            raw = line.strip()
            if not raw:
                continue

            req_id = None
            try:
                payload = json.loads(raw)
                req_id = payload.get("id")
                command = str(payload.get("command", "")).strip()
                args = payload.get("args") or {}

                if command in _SESSION_COMMANDS:
                    _respond_async(runtime, req_id, command, args, _session_lock)
                elif command in _TEXT_COMMANDS:
                    _respond_async(runtime, req_id, command, args, _text_lock)
                else:
                    # health / get_settings / set_model_path are cheap and
                    # must never queue behind a running generation.
                    _respond(runtime, req_id, command, args)
            except Exception as exc:
                logger.exception("Bridge command error")
                _write_packet({
                    "type": "response",
                    "id": req_id,
                    "result": {"ok": False, "error": str(exc)},
                })
    finally:
        try:
            runtime.stop()
        except Exception:
            pass


if __name__ == "__main__":
    run_stdio_bridge()
