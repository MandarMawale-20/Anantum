# In-process countdown timers with desktop notifications.

import logging
import platform
import subprocess
import time
import uuid
import threading

logger = logging.getLogger(__name__)

from skills.base import ToolRegistry


class TimerManager:
    """Thread-safe in-process timer manager with desktop notifications."""
    _timers: dict = {}
    _lock = threading.Lock()
    _event_callback = None

    @classmethod
    def set_event_callback(cls, callback):
        cls._event_callback = callback

    @classmethod
    def set(cls, seconds: int, label: str = "Timer") -> dict:
        # Start a background timer thread.
        timer_id = str(uuid.uuid4())
        end_time = time.time() + seconds

        def _fire():
            time.sleep(seconds)
            cls._notify(label, seconds)
            callback = cls._event_callback
            if callable(callback):
                try:
                    callback({
                        "type": "reminder",
                        "label": label,
                        "seconds": seconds,
                        "display": f"{label}: Time's up!",
                    })
                except Exception:
                    logger.debug("Failed to emit timer reminder event", exc_info=True)
            with cls._lock:
                cls._timers.pop(timer_id, None)

        t = threading.Thread(target=_fire, daemon=True)
        t.start()

        with cls._lock:
            cls._timers[timer_id] = {
                "id": timer_id,
                "label": label,
                "ends_at": end_time,
                "seconds": seconds,
                "thread": t
            }

        duration_str = cls._format_duration(seconds)
        return {
            "timer_id": timer_id,
            "label": label,
            "duration": duration_str,
            "display": f"Timer set! I'll remind you in {duration_str}."
        }

    @classmethod
    def list(cls) -> dict:
        now = time.time()
        with cls._lock:
            active = []
            for tid, info in cls._timers.items():
                remaining = max(0, int(info["ends_at"] - now))
                active.append({
                    "id": tid,
                    "label": info["label"],
                    "remaining": cls._format_duration(remaining),
                    "remaining_seconds": remaining
                })
        return {
            "count": len(active),
            "timers": active,
            "display": cls._format_list(active)
        }

    @classmethod
    def cancel(cls, timer_id: str = None) -> dict:
        # Cancel one timer by ID or clear all timers.
        with cls._lock:
            if timer_id:
                if timer_id in cls._timers:
                    del cls._timers[timer_id]
                    return {"display": f"Timer {timer_id} cancelled."}
                return {"display": f"No timer found with ID {timer_id}."}
            else:
                count = len(cls._timers)
                cls._timers.clear()
                return {"display": f"Cancelled {count} timer(s)."}

    @staticmethod
    def _notify(label: str, duration: int):
        """Cross-platform desktop notification (safe against shell injection)."""
        msg = f"[Timer] {label} - Time's up!"
        try:
            system = platform.system()
            if system == "Darwin":
                subprocess.run(
                    ["osascript", "-e",
                     f'display notification "{msg}" with title "Anantum"'],
                    timeout=5, check=False,
                )
            elif system == "Linux":
                subprocess.run(
                    ["notify-send", "Anantum", msg],
                    timeout=5, check=False,
                )
            elif system == "Windows":
                # msg.exe target: * means all sessions
                subprocess.run(
                    ["msg", "*", msg],
                    timeout=5, check=False,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            else:
                logger.info("Timer done: %s", msg)
        except Exception:
            logger.info("Timer done: %s", msg)

    @staticmethod
    def _format_duration(seconds: int) -> str:
        if seconds < 60:
            return f"{seconds} second{'s' if seconds != 1 else ''}"
        elif seconds < 3600:
            m = seconds // 60
            s = seconds % 60
            base = f"{m} minute{'s' if m != 1 else ''}"
            return base + (f" {s}s" if s > 0 else "")
        else:
            h = seconds // 3600
            m = (seconds % 3600) // 60
            return f"{h}h {m}m"

    @staticmethod
    def _format_list(timers: list) -> str:
        if not timers:
            return "No active timers."
        lines = ["Active timers:"]
        for t in timers:
            lines.append(f"  [{t['id']}] {t['label']} - {t['remaining']} remaining")
        return "\n".join(lines)


@ToolRegistry.register("set_timer", "Set a countdown timer", mode="both")
def set_timer(seconds: int = 60, label: str = "Timer") -> dict:
    return TimerManager.set(seconds, label)


@ToolRegistry.register("list_timers", "List active timers", mode="both")
def list_timers() -> dict:
    return TimerManager.list()


@ToolRegistry.register("cancel_timer", "Cancel a timer", mode="both")
def cancel_timer(timer_id: str = None) -> dict:
    return TimerManager.cancel(timer_id)
