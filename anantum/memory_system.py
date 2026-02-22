# memory_system.py — three-tier memory for personalization and context.
#
# Hot:  in-memory deque of last N turns. Used for immediate context. Zero latency.
# Warm: FAISS HNSW semantic index. Stores facts and key memories. ~5ms query.
# Cold: SQLite archive for conversation summaries. Searched less often. ~20ms.

import json
import math
import time
import threading
import sqlite3
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import deque
from typing import Optional
import faiss
from sentence_transformers import SentenceTransformer

EMBEDDING_DIM = 384
DATA_DIR = Path("anantum_data")
DATA_DIR.mkdir(exist_ok=True)


@dataclass
class MemoryEntry:
    text: str
    embedding: list           # 384-dim vector
    timestamp: float
    access_count: int = 0
    importance: float = 0.5   # 0.0 → 1.0
    decay_rate: float = 0.008 # per day — ~4 months half-life
    memory_type: str = "fact" # fact | preference | event | summary
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def effective_score(self, query_similarity: float) -> float:
        """Score = semantic relevance × time decay × importance boost + access frequency.
        
        Exponential decay means 4-month-old facts fade to 50%. Frequently-accessed
        memories (e.g., user name) stay fresh. Low-importance facts (e.g., "likes coffee")
        lose relevance faster unless queried, encouraging focus on salient context.
        """
        days_old = (time.time() - self.timestamp) / 86400
        temporal_decay = math.exp(-self.decay_rate * days_old)  # ~4mo half-life
        access_boost = min(self.access_count * 0.04, 0.25)
        importance_weight = 0.7 + (self.importance * 0.6)  # scales 0.7–1.3

        return (query_similarity * temporal_decay * importance_weight) + access_boost


# --- hot cache ---

class HotCache:
    """Last 20 conversation turns kept in a deque. No embedding overhead."""

    def __init__(self, max_turns: int = 20):
        self.turns: deque = deque(maxlen=max_turns)

    def add(self, role: str, content: str):
        self.turns.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })

    def get_recent(self, n: int = 6) -> list:
        turns = list(self.turns)
        return turns[-n:] if len(turns) >= n else turns

    def get_all(self) -> list:
        return list(self.turns)

    def clear(self):
        self.turns.clear()

    def __len__(self):
        return len(self.turns)


# --- warm store (FAISS) ---

class WarmStore:
    """
    FAISS HNSW index for semantic memory retrieval.
    HNSW is ~10x faster than a flat index at scale and stays accurate enough for this use case.
    """

    INDEX_FILE = DATA_DIR / "warm_index.faiss"
    META_FILE = DATA_DIR / "warm_meta.json"

    def __init__(self, embedding_model: SentenceTransformer):
        self.model = embedding_model
        self.entries: list[MemoryEntry] = []

        # HNSW: approximate nearest neighbor search. 32 connections per node balances
        # query latency (~5ms) vs recall (~99%). Faster than flat L2 at scale.
        self.index = faiss.IndexHNSWFlat(EMBEDDING_DIM, 32)
        self.index.hnsw.efConstruction = 200  # index construction cost
        self.index.hnsw.efSearch = 64          # query-time search effort

        self._load()

    def embed(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], convert_to_numpy=True)[0]
        return vec.astype(np.float32)

    def add(self, text: str, importance: float = 0.5,
            memory_type: str = "fact", metadata: dict = None) -> MemoryEntry:
        vec = self.embed(text)
        entry = MemoryEntry(
            text=text,
            embedding=vec.tolist(),
            timestamp=time.time(),
            importance=importance,
            memory_type=memory_type,
            metadata=metadata or {}
        )
        self.entries.append(entry)
        self.index.add(vec.reshape(1, -1))
        self._save()
        return entry

    def search(self, query: str, top_k: int = 5) -> list[tuple[MemoryEntry, float]]:
        if len(self.entries) == 0:
            return []

        q_vec = self.embed(query).reshape(1, -1)
        # Fetch 3x candidates; re-ranking by effective_score corrects for recency/importance.
        # This trades a small search cost for better relevance than pure semantic similarity.
        k = min(top_k * 3, len(self.entries))
        distances, indices = self.index.search(q_vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            # Convert L2 distance to 0-1 similarity for consistent scoring.
            similarity = 1.0 / (1.0 + dist)
            entry = self.entries[idx]
            final_score = entry.effective_score(similarity)
            results.append((entry, final_score))

        # re-rank and trim to top_k
        results.sort(key=lambda x: x[1], reverse=True)
        top_results = results[:top_k]

        for entry, _ in top_results:
            entry.access_count += 1

        return top_results

    def prune(self, max_entries: int = 5000, min_score_threshold: float = 0.05):
        """Drop lowest-scoring entries when the store grows too large."""
        if len(self.entries) <= max_entries:
            return

        dummy_query_similarity = 0.3
        scored = [(e, e.effective_score(dummy_query_similarity)) for e in self.entries]
        scored.sort(key=lambda x: x[1], reverse=True)

        self.entries = [e for e, _ in scored[:max_entries]]

        # HNSW doesn't support deletion, so we have to rebuild
        self._rebuild_index()
        self._save()
        print(f"[Memory] Pruned to {len(self.entries)} entries")

    def _rebuild_index(self):
        self.index = faiss.IndexHNSWFlat(EMBEDDING_DIM, 32)
        self.index.hnsw.efConstruction = 200
        self.index.hnsw.efSearch = 64
        if self.entries:
            vecs = np.array([e.embedding for e in self.entries], dtype=np.float32)
            self.index.add(vecs)

    def _save(self):
        faiss.write_index(self.index, str(self.INDEX_FILE))
        meta = [
            {
                "text": e.text,
                "embedding": e.embedding,
                "timestamp": e.timestamp,
                "access_count": e.access_count,
                "importance": e.importance,
                "decay_rate": e.decay_rate,
                "memory_type": e.memory_type,
                "metadata": e.metadata
            }
            for e in self.entries
        ]
        with open(self.META_FILE, "w") as f:
            json.dump(meta, f)

    def _load(self):
        if self.INDEX_FILE.exists() and self.META_FILE.exists():
            try:
                self.index = faiss.read_index(str(self.INDEX_FILE))
                with open(self.META_FILE) as f:
                    meta = json.load(f)
                self.entries = [MemoryEntry(**m) for m in meta]
                print(f"[Memory] Loaded {len(self.entries)} warm memories")
            except Exception as e:
                print(f"[Memory] Failed to load warm store: {e}, starting fresh")

    def __len__(self):
        return len(self.entries)


# --- cold archive (SQLite) ---

class ColdArchive:
    """
    SQLite-backed archive for conversation summaries and old memories.
    Written in a background thread; never fully loaded into RAM.
    """

    DB_FILE = DATA_DIR / "cold_archive.db"

    def __init__(self):
        self.conn = sqlite3.connect(str(self.DB_FILE), check_same_thread=False)
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self):
        with self.conn:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    summary TEXT NOT NULL,
                    turn_range TEXT,
                    created_at REAL,
                    tags TEXT
                );

                CREATE TABLE IF NOT EXISTS archived_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    memory_type TEXT,
                    importance REAL,
                    original_timestamp REAL,
                    archived_at REAL,
                    metadata TEXT
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts
                USING fts5(summary, tags, content=summaries, content_rowid=id);
            """)

    def store_summary(self, summary: str, turn_range: str = "", tags: list = None):
        with self._lock:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO summaries (summary, turn_range, created_at, tags) VALUES (?,?,?,?)",
                    (summary, turn_range, time.time(), json.dumps(tags or []))
                )

    def archive_memory(self, entry: MemoryEntry):
        with self._lock:
            with self.conn:
                self.conn.execute(
                    """INSERT INTO archived_memories
                       (text, memory_type, importance, original_timestamp, archived_at, metadata)
                       VALUES (?,?,?,?,?,?)""",
                    (entry.text, entry.memory_type, entry.importance,
                     entry.timestamp, time.time(), json.dumps(entry.metadata))
                )

    def search_summaries(self, keyword: str, limit: int = 5) -> list[dict]:
        with self._lock:
            cur = self.conn.execute(
                "SELECT summary, turn_range, created_at FROM summaries WHERE summary LIKE ? ORDER BY created_at DESC LIMIT ?",
                (f"%{keyword}%", limit)
            )
            return [{"summary": r[0], "turn_range": r[1], "created_at": r[2]} for r in cur.fetchall()]

    def get_recent_summaries(self, limit: int = 3) -> list[str]:
        with self._lock:
            cur = self.conn.execute(
                "SELECT summary FROM summaries ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )
            return [r[0] for r in cur.fetchall()]


# --- memory manager ---

class MemoryManager:
    """
    Unified interface over all three tiers.
    Handles fact extraction, background summarisation, and decay/pruning.
    """

    # Regex patterns for extracting facts from user messages.
    # Patterns are intentionally strict and require sentence boundaries
    # to avoid false positives like "I'm going to the store."
    FACT_PATTERNS = [
        # Name — very specific, requires proper noun-like word
        (r"my name is ([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", "name", 1.0),
        (r"(?:call me|i(?:'m| am)) ([A-Z][a-z]+)(?:\s*[,\.]|$)", "name", 0.9),

        # Location — requires preposition + location noun
        (r"i(?:'m| am) from ([\w\s]+?)(?:\s*[,\.]|$)", "location", 0.7),
        (r"i live in ([\w\s]+?)(?:\s*[,\.]|$)", "location", 0.7),
        (r"i(?:'m| am) based in ([\w\s]+?)(?:\s*[,\.]|$)", "location", 0.7),

        # Role — must end clearly, not "I'm going to..."
        (r"i(?:'m| am) a(?:n)? ([\w\s]{3,30})(?:\s*[,\.]|$)", "role", 0.6),

        # Preferences — require a meaningful object
        (r"i (?:really )?(?:like|love|enjoy) ([\w\s]{3,40})(?:\s*[,\.]|$)", "preference", 0.6),
        (r"i (?:hate|dislike|don't like) ([\w\s]{3,40})(?:\s*[,\.]|$)", "aversion", 0.6),
        (r"i (?:use|prefer) ([\w\s]{2,30}) (?:for|over|instead)", "tool_preference", 0.6),

        # Projects — must follow "on" or "building"
        (r"i(?:'m| am) working on ([A-Za-z][\w\s]{2,40})(?:\s*[,\.]|$)", "project", 0.8),
        (r"i(?:'m| am) building ([A-Za-z][\w\s]{2,40})(?:\s*[,\.]|$)", "project", 0.8),
        (r"i(?:'m| am) developing ([A-Za-z][\w\s]{2,40})(?:\s*[,\.]|$)", "project", 0.8),

        # Goals
        (r"my (?:goal|aim|target) is (?:to )?([\w\s]{5,60})(?:\s*[,\.]|$)", "goal", 0.9),

        # Explicit remember/note requests (user deliberately saying this)
        (r"remember that (.{5,100})(?:\s*[,\.]|$)", "note", 0.85),
        (r"note that (.{5,100})(?:\s*[,\.]|$)", "note", 0.85),
    ]

    # Values that should never be stored as a fact on their own
    FACT_BLACKLIST = {
        "going", "fine", "good", "great", "okay", "ok", "well", "here",
        "there", "sure", "ready", "done", "back", "up", "in", "out",
        "now", "just", "still", "also", "about", "trying", "planning",
        "not", "very", "too", "really", "actually", "literally",
    }

    def __init__(self):
        print("[Memory] Loading embedding model...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        self.hot = HotCache(max_turns=20)
        self.warm = WarmStore(self.embedder)
        self.cold = ColdArchive()

        self._summary_pending_turns = 0
        self._summary_threshold = 20  # summarize every 20 turns
        self._bg_lock = threading.Lock()

        print(f"[Memory] Ready — {len(self.warm)} warm memories loaded")

    def process_user_message(self, text: str):
        """Call on every user turn. Adds to hot cache and runs fact extraction."""
        self.hot.add("user", text)
        self._extract_and_store_facts(text)
        self._summary_pending_turns += 1

        if self._summary_pending_turns >= self._summary_threshold:
            self._trigger_background_summary()

    def process_assistant_message(self, text: str):
        """Call on every assistant turn. Adds to hot cache."""
        self.hot.add("assistant", text)

    def get_context_for_prompt(self, query: str) -> dict:
        """
        Returns the memory context dict needed to build a prompt:
          recent_turns   -> hot cache (last 6 turns)
          relevant_facts -> warm FAISS results
          past_summaries -> cold archive summaries
        """
        recent_turns = self.hot.get_recent(n=6)

        warm_results = self.warm.search(query, top_k=5)
        relevant_facts = [
            {"text": e.text, "score": round(score, 3), "type": e.memory_type}
            for e, score in warm_results
            if score > 0.15  # skip very low-relevance hits
        ]

        # tier 3: a couple of recent summaries for long-term context
        past_summaries = self.cold.get_recent_summaries(limit=2)

        return {
            "recent_turns": recent_turns,
            "relevant_facts": relevant_facts,
            "past_summaries": past_summaries
        }

    def store_fact(self, text: str, importance: float = 0.5,
                   memory_type: str = "fact", metadata: dict = None):
        self.warm.add(text, importance=importance,
                      memory_type=memory_type, metadata=metadata)

    def prune_if_needed(self):
        self.warm.prune(max_entries=5000)

    def on_session_end(self):
        """Trigger a final summary when the user closes the app or goes idle."""
        if len(self.hot) >= 4:
            self._trigger_background_summary(force=True)

    # --- fact extraction ---

    def _extract_and_store_facts(self, text: str):
        """Extract user facts from natural conversation.
        
        Patterns are strict (require sentence endings) to avoid false positives like
        "I'm going to the store" being parsed as "going" = location. False positives
        pollute memory more than false negatives, so we prefer high precision.
        """
        import re
        for pattern, fact_type, importance in self.FACT_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                continue
            fact_text = match.group(1).strip()
            fact_lower = fact_text.lower()
            if len(fact_text) < 3:
                continue
            # Skip if it's purely a blacklisted word (e.g., "fine" or "going").
            words = fact_lower.split()
            if len(words) == 1 and fact_lower in self.FACT_BLACKLIST:
                continue
            if len(words) <= 2 and words and words[0] in self.FACT_BLACKLIST:
                continue
            # Skip verb phrases that user is saying as actions, not defining themselves.
            # "I'm going to X" is not the same as "I am X".
            if re.match(r"^(going|trying|planning|about|wanting|looking|just|also)\s", fact_lower):
                continue
            full_fact = f"[{fact_type}] {fact_text}"
            already = any(
                e.memory_type == fact_type and e.text.lower() == full_fact.lower()
                for e in self.warm.entries
            )
            if already:
                continue
            # Names: replace old entry to avoid ambiguity ("one" vs "mandar").
            # User mistake or correction should overwrite stale fact.
            if fact_type == "name":
                self.warm.entries = [e for e in self.warm.entries if e.memory_type != "name"]
                self.warm._rebuild_index()
            self.warm.add(
                full_fact, importance=importance,
                memory_type=fact_type,
                metadata={"source": "auto_extract", "original": text}
            )
            print(f"[Memory] Extracted: {full_fact}")

    # --- background summarisation ---

    def _trigger_background_summary(self, force: bool = False):
        """Summarise recent conversation turns in a background thread."""
        def _summarize():
            with self._bg_lock:
                turns = self.hot.get_all()
                if len(turns) < 4:
                    return

                # Build summary text from turns
                lines = []
                for t in turns:
                    role = "User" if t["role"] == "user" else "Anantum"
                    lines.append(f"{role}: {t['content'][:200]}")

                summary_text = " | ".join(lines[-10:])  # condense last 10 turns
                turn_range = f"{len(turns)} turns"

                # Extract key topics for tags
                tags = self._extract_topics(summary_text)

                self.cold.store_summary(
                    summary=summary_text,
                    turn_range=turn_range,
                    tags=tags
                )
                self._summary_pending_turns = 0
                print(f"[Memory] Saved conversation summary ({turn_range})")

        thread = threading.Thread(target=_summarize, daemon=True)
        thread.start()

    def _extract_topics(self, text: str) -> list[str]:
        """Simple word-frequency keyword extraction for summary tags."""
        stopwords = {"i", "a", "the", "is", "it", "to", "you", "me", "and",
                     "or", "of", "in", "on", "my", "do", "did", "was", "are"}
        words = text.lower().split()
        words = [w.strip(".,?!\"'") for w in words if len(w) > 3 and w not in stopwords]
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        top = sorted(freq, key=freq.get, reverse=True)
        return top[:5]