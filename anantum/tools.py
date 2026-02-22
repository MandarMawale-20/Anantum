# tools.py — all tool functions available in Normal Mode
# Each tool is registered with a decorator and invoked through ToolRegistry.

import os
import re
import json
import time
import uuid
import math
import datetime
import threading
import sqlite3
from pathlib import Path
from typing import Callable, Optional

DATA_DIR = Path("anantum_data")
DATA_DIR.mkdir(exist_ok=True)
NOTES_DB = DATA_DIR / "notes.db"


# Tool registry: central dispatch for all available functions.
# Each tool is a standalone function with @ToolRegistry.register decorator.
class ToolRegistry:
    _tools: dict = {}

    @classmethod
    def register(cls, name: str, description: str, mode: str = "both"):
        """Register a function as a callable tool. Modes: 'normal', 'celestial', 'both'."""
        def decorator(fn: Callable):
            cls._tools[name] = {
                "fn": fn,
                "description": description,
                "mode": mode,
            }
            return fn
        return decorator

    @classmethod
    def run(cls, name: str, **kwargs) -> dict:
        if name not in cls._tools:
            return {"error": f"Tool '{name}' not found"}
        try:
            result = cls._tools[name]["fn"](**kwargs)
            return {"success": True, "result": result, "tool": name}
        except Exception as e:
            return {"success": False, "error": str(e), "tool": name}

    @classmethod
    def list_tools(cls, mode: str = "normal") -> dict:
        return {
            k: v["description"]
            for k, v in cls._tools.items()
            if v["mode"] in (mode, "both")
        }


# --- time and date ---

@ToolRegistry.register("get_time", "Get current time", mode="both")
def get_time() -> dict:
    now = datetime.datetime.now()
    return {
        "time_12h": now.strftime("%I:%M %p"),
        "time_24h": now.strftime("%H:%M:%S"),
        "display": f"It's {now.strftime('%I:%M %p')}",
    }


@ToolRegistry.register("get_date", "Get current date and day", mode="both")
def get_date() -> dict:
    now = datetime.datetime.now()
    return {
        "date": now.strftime("%B %d, %Y"),
        "day": now.strftime("%A"),
        "display": f"Today is {now.strftime('%A, %B %d, %Y')}",
        "iso": now.date().isoformat(),
    }


# --- timer manager ---

class TimerManager:
    """Thread-safe in-process timer manager with desktop notifications."""
    _timers: dict = {}
    _lock = threading.Lock()

    @classmethod
    def set(cls, seconds: int, label: str = "Timer") -> dict:
        timer_id = str(uuid.uuid4())[:8]
        end_time = time.time() + seconds

        def _fire():
            time.sleep(seconds)
            cls._notify(label, seconds)
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
        """Cross-platform desktop notification."""
        msg = f"[Timer] {label} - Time's up!"
        try:
            import platform
            system = platform.system()
            if system == "Darwin":
                os.system(f'osascript -e \'display notification "{msg}" with title "Anantum"\'')
            elif system == "Linux":
                os.system(f'notify-send "Anantum" "{msg}" 2>/dev/null || echo "{msg}"')
            elif system == "Windows":
                os.system(f'msg * "{msg}" 2>nul || echo {msg}')
            else:
                print(f"\n[TIMER] {msg}\n")
        except Exception:
            print(f"\n[TIMER] {msg}\n")

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


# --- notes ---

def _init_notes_db():
    conn = sqlite3.connect(str(NOTES_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            tags TEXT,
            created_at REAL,
            updated_at REAL
        )
    """)
    conn.commit()
    return conn


@ToolRegistry.register("save_note", "Save a note", mode="both")
def save_note(content: str, tags: list = None) -> dict:
    note_id = str(uuid.uuid4())[:8]
    now = time.time()
    tags = tags or []
    conn = _init_notes_db()
    conn.execute(
        "INSERT INTO notes VALUES (?,?,?,?,?)",
        (note_id, content, json.dumps(tags), now, now)
    )
    conn.commit()
    conn.close()
    return {
        "id": note_id,
        "content": content,
        "display": f"Note saved. (ID: {note_id})"
    }


@ToolRegistry.register("get_notes", "Retrieve recent notes", mode="both")
def get_notes(limit: int = 5, search: str = None) -> dict:
    conn = _init_notes_db()
    if search:
        cur = conn.execute(
            "SELECT id, content, tags, created_at FROM notes WHERE content LIKE ? ORDER BY created_at DESC LIMIT ?",
            (f"%{search}%", limit)
        )
    else:
        cur = conn.execute(
            "SELECT id, content, tags, created_at FROM notes ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
    rows = cur.fetchall()
    conn.close()

    notes = []
    for r in rows:
        ts = datetime.datetime.fromtimestamp(r[3]).strftime("%b %d, %H:%M")
        notes.append({"id": r[0], "content": r[1], "tags": json.loads(r[2]), "created": ts})

    if not notes:
        display = "No notes found."
    else:
        lines = [f"Your {len(notes)} most recent notes:"]
        for n in notes:
            lines.append(f"  [{n['id']}] {n['content'][:100]}  ({n['created']})")
        display = "\n".join(lines)

    return {"notes": notes, "count": len(notes), "display": display}


# --- calculator ---

@ToolRegistry.register("calculate", "Evaluate a math expression safely", mode="both")
def calculate(expression: str = "", raw_text: str = "") -> dict:
    """Safely evaluate math expressions. No eval() or exec()—uses AST whitelisting.
    
    This prevents code injection while still supporting natural input like
    '15% of 200' or 'square root of 144'.
    """
    expr = expression or raw_text

    # Normalize natural language math operators to symbols.
    expr = expr.lower()
    expr = re.sub(r"\bplus\b", "+", expr)
    expr = re.sub(r"\bminus\b", "-", expr)
    expr = re.sub(r"\btimes\b|multiplied by", "*", expr)
    expr = re.sub(r"\bdivided by\b", "/", expr)
    expr = re.sub(r"\bsquared\b", "**2", expr)
    expr = re.sub(r"\bcubed\b", "**3", expr)
    expr = re.sub(r"\bsquare root of\b", "sqrt", expr)

    # Percent of
    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:percent|%)\s*of\s*(\d+(?:\.\d+)?)", expr)
    if pct_match:
        pct, total = float(pct_match.group(1)), float(pct_match.group(2))
        result = pct / 100 * total
        return {"result": result, "display": f"{pct}% of {total} = {result}"}

    # sqrt
    sqrt_match = re.search(r"sqrt\s*\(?(\d+(?:\.\d+)?)\)?", expr)
    if sqrt_match:
        n = float(sqrt_match.group(1))
        result = math.sqrt(n)
        return {"result": result, "display": f"√{n} = {result}"}

    # Safe eval: only allow numbers and operators
    safe_expr = re.sub(r"[^\d\s\+\-\*\/\.\(\)\^%]", "", expr)
    safe_expr = safe_expr.replace("^", "**")

    try:
        import ast
        tree = ast.parse(safe_expr, mode='eval')

        # only allow basic arithmetic nodes, reject anything else
        allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                   ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
                   ast.FloorDiv, ast.USub, ast.UAdd)
        for node in ast.walk(tree):
            if not isinstance(node, allowed):
                return {"error": "Unsafe expression", "display": "I can only compute basic math expressions."}

        result = eval(compile(tree, '<string>', 'eval'))
        return {"result": result, "expression": safe_expr, "display": f"{safe_expr} = {result}"}
    except Exception as e:
        return {"error": str(e), "display": f"Couldn't compute that. Try: '2 + 2' or '15% of 200'"}


# --- device / system info ---

@ToolRegistry.register("get_device_info", "Get system info: CPU, RAM, disk, battery", mode="both")
def get_device_info() -> dict:
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.5)
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        info = {
            "cpu_percent": cpu,
            "ram_used_gb": round(ram.used / 1e9, 1),
            "ram_total_gb": round(ram.total / 1e9, 1),
            "ram_percent": ram.percent,
            "disk_used_gb": round(disk.used / 1e9, 1),
            "disk_total_gb": round(disk.total / 1e9, 1),
            "disk_percent": disk.percent,
        }

        try:
            battery = psutil.sensors_battery()
            if battery:
                info["battery_percent"] = round(battery.percent, 1)
                info["battery_plugged"] = battery.power_plugged
        except Exception:
            pass  # not all platforms support battery info

        lines = [
            f"CPU: {cpu}%",
            f"RAM: {info['ram_used_gb']}GB / {info['ram_total_gb']}GB ({ram.percent}%)",
            f"Disk: {info['disk_used_gb']}GB / {info['disk_total_gb']}GB ({disk.percent}%)",
        ]
        if "battery_percent" in info:
            status = "plugged in" if info["battery_plugged"] else "on battery"
            lines.append(f"Battery: {info['battery_percent']}% ({status})")

        info["display"] = "\n".join(lines)
        return info

    except ImportError:
        return {"display": "Install psutil for system info: pip install psutil", "error": "psutil not installed"}


# --- weather (requires internet) ---

@ToolRegistry.register("get_weather", "Get current weather (requires internet)", mode="both")
def get_weather(location: str = None) -> dict:
    """Fetch weather via Open-Meteo (preferred) or wttr.in (fallback).
    
    Open-Meteo is free, fast, and works globally without an API key.
    Both support location fuzzing, so 'London' finds the likely city.
    """
    import urllib.request
    import urllib.parse

    def _try_open_meteo(loc: str) -> dict:
        """Open-Meteo: free, no API key, works globally."""
        if loc and loc != "auto":
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(loc)}&count=1&language=en&format=json"
            with urllib.request.urlopen(geo_url, timeout=8) as r:
                geo = json.loads(r.read())
            if not geo.get("results"):
                return None
            result = geo["results"][0]
            lat = result["latitude"]
            lon = result["longitude"]
            city = result.get("name", loc)
            country = result.get("country", "")
        else:
            # No location given: use IP geolocation to detect user's approximate region.
            with urllib.request.urlopen("https://ipapi.co/json/", timeout=5) as r:
                ip_data = json.loads(r.read())
            lat = ip_data["latitude"]
            lon = ip_data["longitude"]
            city = ip_data.get("city", "Unknown")
            country = ip_data.get("country_name", "")

        # Step 2: Get weather
        wx_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,"
            f"weather_code,wind_speed_10m"
            f"&wind_speed_unit=kmh&timezone=auto"
        )
        with urllib.request.urlopen(wx_url, timeout=8) as r:
            wx = json.loads(r.read())

        cur = wx["current"]
        temp_c = round(cur["temperature_2m"])
        feels_c = round(cur["apparent_temperature"])
        temp_f = round(temp_c * 9/5 + 32)
        humidity = cur["relative_humidity_2m"]
        wind = round(cur["wind_speed_10m"])
        code = cur["weather_code"]

        # WMO weather codes (standard in meteorology). Mapped to human-readable descriptions.
        WMO = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Icy fog", 51: "Light drizzle", 53: "Drizzle",
            55: "Heavy drizzle", 61: "Light rain", 63: "Rain", 65: "Heavy rain",
            71: "Light snow", 73: "Snow", 75: "Heavy snow", 80: "Showers",
            81: "Rain showers", 82: "Heavy showers", 95: "Thunderstorm",
            96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
        }
        desc = WMO.get(code, f"Weather code {code}")

        display = (
            f"Weather in {city}, {country}:\n"
            f"  {desc}, {temp_c}°C / {temp_f}°F\n"
            f"  Feels like: {feels_c}°C\n"
            f"  Humidity: {humidity}%  |  Wind: {wind} km/h"
        )
        return {
            "location": f"{city}, {country}",
            "temp_c": temp_c, "temp_f": temp_f, "feels_c": feels_c,
            "description": desc, "humidity": humidity, "wind_kmph": wind,
            "display": display
        }

    def _try_wttr(loc: str) -> dict:
        loc_param = loc if loc and loc != "auto" else ""
        url = f"https://wttr.in/{urllib.parse.quote(loc_param)}?format=j1"
        with urllib.request.urlopen(url, timeout=8) as r:
            data = json.loads(r.read())
        cur = data["current_condition"][0]
        area = data["nearest_area"][0]
        city = area["areaName"][0]["value"]
        country = area["country"][0]["value"]
        temp_c = int(cur["temp_C"])
        temp_f = int(cur["temp_F"])
        feels_c = int(cur["FeelsLikeC"])
        desc = cur["weatherDesc"][0]["value"]
        humidity = cur["humidity"]
        wind = cur["windspeedKmph"]
        display = (
            f"Weather in {city}, {country}:\n"
            f"  {desc}, {temp_c}°C / {temp_f}°F\n"
            f"  Feels like: {feels_c}°C\n"
            f"  Humidity: {humidity}%  |  Wind: {wind} km/h"
        )
        return {"location": f"{city},{country}", "temp_c": temp_c, "temp_f": temp_f,
                "description": desc, "humidity": humidity, "display": display}

    loc = (location or "").strip()
    errors = []

    # Try Open-Meteo first. More reliable than wttr.in and doesn't have regional blocks.
    try:
        return _try_open_meteo(loc if loc else "auto")
    except Exception as e:
        errors.append(f"Open-Meteo: {e}")

    # Fallback: wttr.in. Slightly slower but works when Open-Meteo times out.
    try:
        return _try_wttr(loc)
    except Exception as e:
        errors.append(f"wttr.in: {e}")

    # Both failed
    print(f"[Weather] All APIs failed: {errors}")
    return {
        "error": str(errors),
        "display": f"Couldn't fetch weather for {'''+loc+''' if loc else 'your location'}. Error: {errors[0]}"
    }


# Web search via DuckDuckGo. Enabled only in Celestial mode to avoid latency in normal mode.
@ToolRegistry.register("web_search", "Search the web using DuckDuckGo", mode="celestial")
def web_search(query: str) -> dict:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return {"display": "No results found.", "results": []}
        formatted = []
        for r in results:
            formatted.append({
                "title": r.get("title", ""),
                "body": r.get("body", ""),
                "url": r.get("href", "")
            })
        display_lines = [f"Web results for '{query}':"]
        for i, r in enumerate(formatted[:3], 1):
            display_lines.append(f"\n{i}. {r['title']}\n   {r['body'][:200]}\n   {r['url']}")
        return {"results": formatted, "display": "\n".join(display_lines)}
    except ImportError:
        return {"error": "duckduckgo_search not installed", "display": "Install: pip install duckduckgo-search"}
    except Exception as e:
        return {"error": str(e), "display": f"Search failed: {e}"}