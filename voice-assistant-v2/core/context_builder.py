# Build prompt context from memory for each user turn.

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ContextBuilder:
    # Cache the last built context for handlers.

    def __init__(self, memory):
        self.memory   = memory
        self._last_ctx = None

    def build(self, user_message: str) -> dict:
        ctx = self.memory.get_context_for_prompt(user_message)
        self._last_ctx = ctx
        return ctx

    @property
    def memory_context(self) -> dict:
        if self._last_ctx is None:
            logger.warning("No context built yet, returning empty")
            return {}
        return self._last_ctx
