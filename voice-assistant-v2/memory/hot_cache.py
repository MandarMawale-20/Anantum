# In-memory ring buffer for recent conversation turns.

import time
from collections import deque
from typing import Any


class HotCache:
    """In-memory ring buffer for conversation turns with auto-expiration."""

    def __init__(self, max_turns: int = 20):
        # Fixed-size deque drops oldest items automatically.
        self.turns: deque[dict[str, Any]] = deque(maxlen=max_turns)

    def add(self, role: str, content: str):
        self.turns.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })

    def get_recent(self, n: int = 6) -> list:
        return list(self.turns)[-n:]

    def get_all(self) -> list:
        return list(self.turns)

    def clear(self):
        self.turns.clear()

    def __len__(self):
        return len(self.turns)
