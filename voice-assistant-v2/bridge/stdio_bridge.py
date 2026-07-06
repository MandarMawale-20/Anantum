import atexit
import json
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bridge.server import AssistantRuntime, execute_command

logging.basicConfig(level=logging.INFO, format="%(asctime)s [bridge] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _write_packet(packet: dict) -> None:
    sys.stdout.write(json.dumps(packet, ensure_ascii=True) + "\n")
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
                result = execute_command(runtime, command, args)
            except Exception as exc:
                logger.exception("Bridge command error")
                result = {"ok": False, "error": str(exc)}

            _write_packet({
                "type": "response",
                "id": req_id,
                "result": result,
            })
    finally:
        try:
            runtime.stop()
        except Exception:
            pass


if __name__ == "__main__":
    run_stdio_bridge()
