"""Packaged backend entrypoint for desktop builds.

This script is compiled to an executable and launched by the Tauri shell.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bridge.stdio_bridge import run_stdio_bridge


if __name__ == "__main__":
    run_stdio_bridge()
