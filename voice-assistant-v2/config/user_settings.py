"""Persistent user settings stored under AppData for packaged installs."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

APP_FOLDER_NAME = "Anantum"
SETTINGS_FILENAME = "settings.json"

PERSISTED_KEYS = (
    "model_path",
    "voice",
    "gpu_layers",
    "tts_device",
    "stt_device",
)


def _settings_root() -> Path:
    override = os.getenv("ANANTUM_SETTINGS_DIR", "").strip()
    if override:
        root = Path(override)
    else:
        base = os.getenv("APPDATA") or os.getenv("LOCALAPPDATA")
        root = Path(base) / APP_FOLDER_NAME if base else Path.home() / ".anantum"
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_settings_file() -> Path:
    return _settings_root() / SETTINGS_FILENAME


def load_user_settings() -> dict[str, Any]:
    path = get_settings_file()
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load user settings from %s: %s", path, exc)
        return {}

    if not isinstance(payload, dict):
        logger.warning("Ignoring invalid user settings payload in %s", path)
        return {}

    return payload


def save_user_settings(settings: dict[str, Any]) -> tuple[bool, str | None]:
    path = get_settings_file()
    payload = {k: settings.get(k) for k in PERSISTED_KEYS if k in settings}

    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return True, None
    except Exception as exc:
        logger.error("Failed to save user settings to %s: %s", path, exc)
        return False, str(exc)


def apply_user_settings(config: Any) -> dict[str, Any]:
    """Compatibility helper to apply persisted values to an AppConfig-like object."""
    loaded = load_user_settings()
    if not loaded:
        return {}

    mapping = {
        "model_path": "llm_model",
        "voice": "kokoro_voice",
        "gpu_layers": "llm_n_gpu_layers",
        "tts_device": "tts_device",
        "stt_device": "stt_device",
    }

    applied: dict[str, Any] = {}
    for key, attr in mapping.items():
        if key not in loaded:
            continue
        try:
            setattr(config, attr, loaded[key])
            applied[key] = loaded[key]
        except Exception as exc:
            logger.warning("Failed to apply persisted key '%s': %s", key, exc)

    return applied


def snapshot_user_settings(config: Any) -> dict[str, Any]:
    return {
        "model_path": getattr(config, "llm_model", ""),
        "voice": getattr(config, "kokoro_voice", ""),
        "gpu_layers": getattr(config, "llm_n_gpu_layers", 0),
        "tts_device": getattr(config, "tts_device", ""),
        "stt_device": getattr(config, "stt_device", ""),
    }
