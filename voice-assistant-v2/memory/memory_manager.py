"""Central memory orchestration for hot, warm, and cold memory tiers."""

import logging
import re

from config.settings import CONFIG
from memory.hot_cache import HotCache
from memory.faiss_store import WarmStore
from memory.cold_archive import ColdArchive
from memory.summarizer import Summarizer

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Unified three-tier memory orchestration: hot cache + warm FAISS store + cold SQLite archive.

    Automatically extracts and stores user-disclosed facts, maintaining precision over recall
    to prevent false memories corrupting the knowledge base. Auto-prunes warm store when size
    exceeds configured limits.

    Fact extraction uses strict regex patterns with sentence boundaries (no "I'm going to...").
    """

    # Regex patterns for extracting user facts.
    FACT_PATTERNS: list[tuple[str, str, float]] = [
        # Name
        (r"my name is ([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", "name", 1.0),
        (r"(?:call me|i(?:'m| am)) ([A-Z][a-z]+)(?:\s*[,\.]|$)", "name", 0.9),

        # Location
        (r"i(?:'m| am) from ([\w\s]+?)(?:\s*[,\.]|$)", "location", 0.7),
        (r"i live in ([\w\s]+?)(?:\s*[,\.]|$)", "location", 0.7),
        (r"i(?:'m| am) based in ([\w\s]+?)(?:\s*[,\.]|$)", "location", 0.7),

        # Role
        (r"i(?:'m| am) a(?:n)? ([\w\s]{3,30})(?:\s*[,\.]|$)", "role", 0.6),

        # Preferences
        (r"i (?:really )?(?:like|love|enjoy) ([\w\s]{3,40})(?:\s*[,\.]|$)", "preference", 0.6),
        (r"i (?:hate|dislike|don't like) ([\w\s]{3,40})(?:\s*[,\.]|$)", "aversion", 0.6),
        (r"i (?:use|prefer) ([\w\s]{2,30}) (?:for|over|instead)", "tool_preference", 0.6),

        # Projects
        (r"i(?:'m| am) working on ([A-Za-z][\w\s]{2,40})(?:\s*[,\.]|$)", "project", 0.8),
        (r"i(?:'m| am) building ([A-Za-z][\w\s]{2,40})(?:\s*[,\.]|$)", "project", 0.8),
        (r"i(?:'m| am) developing ([A-Za-z][\w\s]{2,40})(?:\s*[,\.]|$)", "project", 0.8),

        # Goals
        (r"my (?:goal|aim|target) is (?:to )?([\w\s]{5,60})(?:\s*[,\.]|$)", "goal", 0.9),

        # Explicit remember/note requests
        (r"remember that (.{5,100})(?:\s*[,\.]|$)", "note", 0.85),
        (r"note that (.{5,100})(?:\s*[,\.]|$)", "note", 0.85),
    ]

    # Single-word values to ignore.
    FACT_EXCLUSIONS = {
        "going", "fine", "good", "great", "okay", "ok", "well", "here",
        "there", "sure", "ready", "done", "back", "up", "in", "out",
        "now", "just", "still", "also", "about", "trying", "planning",
        "not", "very", "too", "really", "actually", "literally",
    }

    def __init__(self):
        """Initialize memory tiers and summarizer."""
        logger.info("Initializing memory with embedding model '%s'...", CONFIG.embedding_model)

        self.hot = HotCache(max_turns=20)
        self.warm = WarmStore(embedding_model=CONFIG.embedding_model)
        self.cold = ColdArchive()
        self._summarizer = Summarizer(self.hot, self.cold,
                                      threshold=CONFIG.summary_threshold)

        logger.info("Memory ready — %d warm memories loaded", len(self.warm))

    def set_llm(self, llm) -> None:
        """Pass the loaded LLM to the summarizer for intelligent summarization."""
        self._summarizer.set_llm(llm)

    def process_user_message(self, text: str) -> None:
        """Record user turn, extract facts, trigger summarization, and auto-prune if needed."""
        self.hot.add("user", text)
        self._extract_and_store_facts(text)
        self._summarizer.on_turn()
        if len(self.warm) > CONFIG.warm_store_max_entries:
            self.prune_if_needed()

    def process_assistant_message(self, text: str) -> None:
        """Record assistant response in hot cache."""
        self.hot.add("assistant", text)

    def get_context_for_prompt(self, query: str) -> dict:
        """
        Pull relevant context from all three tiers for prompt assembly.

        Returns:
            recent_turns: Last 6 turns from hot cache
            relevant_facts: Top-5 semantically similar facts from warm store
            past_summaries: Recent compressed session summaries from cold archive
        """
        recent_turns = self.hot.get_recent(n=6)

        if len(query) < 20 or query.lower() in ["hello", "hi", "hey"]:
            relevant_facts = []
        else:
            warm_results = self.warm.search(query, top_k=5)
            relevant_facts = [
                {"text": e.text, "score": round(score, 3), "type": e.memory_type}
                for e, score in warm_results
                if score > 0.15
            ]

        past_summaries = self.cold.get_recent_summaries(limit=2)

        return {
            "recent_turns": recent_turns,
            "relevant_facts": relevant_facts,
            "past_summaries": past_summaries
        }

    def store_fact(self, text: str, importance: float = 0.5,
                   memory_type: str = "fact", metadata: dict = None) -> None:
        """Manually store a fact in the warm store with optional metadata."""
        self.warm.add(text, importance=importance,
                      memory_type=memory_type, metadata=metadata)

    def prune_if_needed(self) -> None:
        """Trim warm store to configured max size if grown too large."""
        self.warm.prune(max_entries=CONFIG.warm_store_max_entries)

    def clear_all(self) -> None:
        """Wipe all memory tiers (called when user says 'forget everything')."""
        with self.warm._save_lock:
            if self.warm._save_timer is not None:
                self.warm._save_timer.cancel()
                self.warm._save_timer = None
        self.warm.entries.clear()
        self.warm._rebuild_index()
        self.warm.flush_save()
        self.hot.clear()
        logger.info("All memory cleared")

    def on_session_end(self) -> None:
        """Flush warm store and trigger final summarization on app exit."""
        if len(self.hot) >= 4:
            self._summarizer.trigger(force=True)
        self.warm.flush_save()  # flush pending writes before exit

    def _extract_and_store_facts(self, text: str) -> None:
        """Extract and store user facts with precision-over-recall filtering."""
        for pattern, fact_type, importance in self.FACT_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                continue

            fact_text = match.group(1).strip()
            fact_lower = fact_text.lower()

            if len(fact_text) < 3:
                continue

            words = fact_lower.split()
            if len(words) == 1 and fact_lower in self.FACT_EXCLUSIONS:
                continue

            if len(words) <= 2 and words and words[0] in self.FACT_EXCLUSIONS:
                continue

            # Skip action phrases like "going to..."
            if re.match(r"^(going|trying|planning|about|wanting|looking|just|also)\s",
                       fact_lower):
                continue

            full_fact = f"[{fact_type}] {fact_text}"
            already_stored = any(
                e.memory_type == fact_type and e.text.lower() == full_fact.lower()
                for e in self.warm.entries
            )
            if already_stored:
                continue

            # Replace old name entries with the latest one.
            if fact_type == "name":
                self.warm.entries = [e for e in self.warm.entries if e.memory_type != "name"]
                self.warm._rebuild_index()

            self.warm.add(
                full_fact,
                importance=importance,
                memory_type=fact_type,
                metadata={"source": "auto_extract", "original": text}
            )
            logger.info("Extracted fact: %s (importance=%.2f)", full_fact, importance)
