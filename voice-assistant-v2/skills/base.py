# Central tool registry and dispatch.

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ToolRegistry:
    _tools: dict[str, dict[str, Any]] = {}

    # Register a callable tool.
    @classmethod
    def register(cls, name: str, description: str, mode: str = "both"):
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
