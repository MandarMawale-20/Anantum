# Background conversation summarizer.

import logging
import threading
from typing import Optional

from memory.hot_cache import HotCache
from memory.cold_archive import ColdArchive

logger = logging.getLogger(__name__)


class Summarizer:

    def __init__(self, hot: HotCache, cold: ColdArchive, threshold: int = 20):
        self.hot = hot
        self.cold = cold
        self._bg_lock = threading.Lock()
        self._summary_pending_turns = 0
        self._summary_threshold = threshold
        self._llm: Optional[object] = None  # set via set_llm() after LLM loads

    def set_llm(self, llm) -> None:
        """Inject the LLM after it finishes loading so summaries become real summaries."""
        self._llm = llm

    def on_turn(self):
        self._summary_pending_turns += 1
        if self._summary_pending_turns >= self._summary_threshold:
            self._summary_pending_turns = 0
            self.trigger(force=False)

    def trigger(self, force: bool = False):
        def _summarize():
            with self._bg_lock:
                turns = self.hot.get_all()
                if len(turns) < 4:
                    return

                lines = []
                for t in turns:
                    role = "User" if t["role"] == "user" else "Anantum"
                    lines.append(f"{role}: {t['content'][:200]}")

                raw_text = " | ".join(lines[-10:])
                turn_range = f"{len(turns)} turns"
                tags = self._extract_topics(raw_text)

                # Fall back to raw text if LLM summary is unavailable.
                summary_text = self._llm_summarize(lines) or raw_text

                self.cold.store_summary(
                    summary=summary_text,
                    turn_range=turn_range,
                    tags=tags
                )
                logger.info("Saved conversation summary (%s)", turn_range)

        thread = threading.Thread(target=_summarize, daemon=True)
        thread.start()

    def _llm_summarize(self, lines: list[str]) -> Optional[str]:
        """Use the LLM to compress the conversation into 2-3 sentences.

        Returns None when the LLM is unavailable or produces garbage output.
        """
        if self._llm is None or not getattr(self._llm, "is_loaded", False):
            return None
        try:
            # Local import avoids circular dependency.
            from core.llm_manager import format_turn, format_generation_start
            conversation = "\n".join(lines[-10:])
            prompt = (
                format_turn("user",
                    f"Summarize the following conversation in 2-3 concise sentences. "
                    f"Focus on key facts, decisions, and topics. No filler phrases.\n\n"
                    f"{conversation}")
                + format_generation_start()
            )
            result = self._llm.generate(prompt, max_tokens=120, temperature=0.3)
            result = result.strip()
            return result if result and len(result) > 20 else None
        except Exception as e:
            logger.debug("LLM summarization failed: %s", e)
            return None

    def _extract_topics(self, text: str) -> list[str]:
        # Build simple tags from frequent content words.
        stopwords = {"i", "a", "the", "is", "it", "to", "you", "me", "and",
                     "or", "of", "in", "on", "my", "do", "did", "was", "are"}
        words = text.lower().split()
        words = [w.strip(".,?!\"'") for w in words if len(w) > 3 and w not in stopwords]
        freq: dict[str, int] = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        top = sorted(freq, key=freq.get, reverse=True)
        return top[:5]
