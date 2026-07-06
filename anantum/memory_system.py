"""Three-tier memory stack: hot deque, warm FAISS store, cold SQLite archive.

Provides persistent, semantically-searchable memory across sessions
with automatic fact extraction and conversation summarization.
"""

import json
import math
import time
import re
import threading
import sqlite3
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from collections import deque
from typing import Optional
import faiss
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import torch

EMBEDDING_DIM = 384
DATA_DIR = Path("anantum_data")
DATA_DIR.mkdir(exist_ok=True)


class ONNXEmbeddingModel:
    """ONNX-optimized sentence embedding model running on GPU."""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model = ORTModelForFeatureExtraction.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            export=True,
            provider="CUDAExecutionProvider"
        )

    def encode(self, texts, convert_to_numpy=True):
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)

        if convert_to_numpy:
            return embeddings.cpu().numpy()
        return embeddings


@dataclass
class MemoryEntry:
    text: str
    embedding: list
    timestamp: float
    access_count: int = 0
    importance: float = 0.5
    decay_rate: float = 0.008
    memory_type: str = "fact"
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def effective_score(self, query_similarity: float) -> float:
        """Blend semantic relevance, freshness, importance, and access frequency."""
        days_old = (time.time() - self.timestamp) / 86400
        temporal_decay = math.exp(-self.decay_rate * days_old)
        access_boost = min(self.access_count * 0.04, 0.25)
        importance_weight = 0.7 + (self.importance * 0.6)

        return (query_similarity * temporal_decay * importance_weight) + access_boost


class HotCache:
    """In-memory window of recent conversation turns."""

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


class WarmStore:
    """FAISS HNSW semantic store for medium-term memory."""

    INDEX_FILE = DATA_DIR / "warm_index.faiss"
    META_FILE = DATA_DIR / "warm_meta.json"

    def __init__(self, embedding_model):
        self.model = embedding_model
        self.entries: list[MemoryEntry] = []

        cpu_index = faiss.IndexHNSWFlat(EMBEDDING_DIM, 32)
        cpu_index.hnsw.efConstruction = 200
        cpu_index.hnsw.efSearch = 32

        try:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        except Exception:
            self.index = cpu_index

        self._load()

    def embed(self, text: str) -> np.ndarray:
        vec = self.model.encode([text])[0]
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
        if len(self.entries) % 10 == 0:
            self._save()
        return entry

    def search(self, query: str, top_k: int = 3) -> list[tuple[MemoryEntry, float]]:
        if len(self.entries) == 0:
            return []

        q_vec = self.embed(query).reshape(1, -1)
        k = min(top_k * 3, len(self.entries))
        distances, indices = self.index.search(q_vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            similarity = 1.0 / (1.0 + dist)
            entry = self.entries[idx]
            final_score = entry.effective_score(similarity)
            results.append((entry, final_score))

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

        self._rebuild_index()
        self._save()

    def _rebuild_index(self):
        cpu_index = faiss.IndexHNSWFlat(EMBEDDING_DIM, 32)
        cpu_index.hnsw.efConstruction = 200
        cpu_index.hnsw.efSearch = 32
        try:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        except Exception:
            self.index = cpu_index
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
            except Exception as e:
                print(f"[Memory] Failed to load warm store: {e}, starting fresh")

    def __len__(self):
        return len(self.entries)


class ColdArchive:
    """SQLite archive for summaries and long-tail memory."""

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


class MemoryManager:
    """Unified coordinator over hot, warm, and cold memory tiers."""

    FACT_PATTERNS = [
        (r"my name is ([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", "name", 1.0),
        (r"(?:call me|i(?:'m| am)) ([A-Z][a-z]+)(?:\s*[,\.]|$)", "name", 0.9),
        (r"i(?:'m| am) from ([\w\s]+?)(?:\s*[,\.]|$)", "location", 0.7),
        (r"i live in ([\w\s]+?)(?:\s*[,\.]|$)", "location", 0.7),
        (r"i(?:'m| am) based in ([\w\s]+?)(?:\s*[,\.]|$)", "location", 0.7),
        (r"i(?:'m| am) a(?:n)? ([\w\s]{3,30})(?:\s*[,\.]|$)", "role", 0.6),
        (r"i (?:really )?(?:like|love|enjoy) ([\w\s]{3,40})(?:\s*[,\.]|$)", "preference", 0.6),
        (r"i (?:hate|dislike|don't like) ([\w\s]{3,40})(?:\s*[,\.]|$)", "aversion", 0.6),
        (r"i (?:use|prefer) ([\w\s]{2,30}) (?:for|over|instead)", "tool_preference", 0.6),
        (r"i(?:'m| am) working on ([A-Za-z][\w\s]{2,40})(?:\s*[,\.]|$)", "project", 0.8),
        (r"i(?:'m| am) building ([A-Za-z][\w\s]{2,40})(?:\s*[,\.]|$)", "project", 0.8),
        (r"i(?:'m| am) developing ([A-Za-z][\w\s]{2,40})(?:\s*[,\.]|$)", "project", 0.8),
        (r"my (?:goal|aim|target) is (?:to )?([\w\s]{5,60})(?:\s*[,\.]|$)", "goal", 0.9),
        (r"remember that (.{5,100})(?:\s*[,\.]|$)", "note", 0.85),
        (r"note that (.{5,100})(?:\s*[,\.]|$)", "note", 0.85),
    ]

    FACT_BLACKLIST = {
        "going", "fine", "good", "great", "okay", "ok", "well", "here",
        "there", "sure", "ready", "done", "back", "up", "in", "out",
        "now", "just", "still", "also", "about", "trying", "planning",
        "not", "very", "too", "really", "actually", "literally",
    }

    def __init__(self):
        print("[Memory] Loading ONNX embedding model on GPU...")
        self.embedder = ONNXEmbeddingModel()

        self.hot = HotCache(max_turns=20)
        self.warm = WarmStore(self.embedder)
        self.cold = ColdArchive()

        self._summary_pending_turns = 0
        self._summary_threshold = 20
        self._bg_lock = threading.Lock()

    def process_user_message(self, text: str):
        """Record user turn and run extraction/summarization checks."""
        self.hot.add("user", text)
        self._extract_and_store_facts(text)
        self._summary_pending_turns += 1

        if self._summary_pending_turns >= self._summary_threshold:
            self._trigger_background_summary()

    def process_assistant_message(self, text: str):
        """Record assistant turn."""
        self.hot.add("assistant", text)

    def get_context_for_prompt(self, query: str) -> dict:
        """Build memory context payload for prompt construction."""
        recent_turns = self.hot.get_recent(n=6)

        if len(query.strip()) < 20 or query.lower() in ("hi", "hello", "hey"):
            relevant_facts = []
        else:
            warm_results = self.warm.search(query, top_k=3)
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
                   memory_type: str = "fact", metadata: dict = None):
        self.warm.add(text, importance=importance,
                      memory_type=memory_type, metadata=metadata)

    def prune_if_needed(self):
        self.warm.prune(max_entries=5000)

    def on_session_end(self):
        """Trigger final summary before shutdown."""
        if len(self.hot) >= 4:
            self._trigger_background_summary(force=True)

    def _extract_and_store_facts(self, text: str):
        """Extract high-confidence user facts from free-form text."""
        for pattern, fact_type, importance in self.FACT_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                continue
            fact_text = match.group(1).strip()
            fact_lower = fact_text.lower()
            if len(fact_text) < 3:
                continue
            words = fact_lower.split()
            if len(words) == 1 and fact_lower in self.FACT_BLACKLIST:
                continue
            if len(words) <= 2 and words and words[0] in self.FACT_BLACKLIST:
                continue
            if re.match(r"^(going|trying|planning|about|wanting|looking|just|also)\s", fact_lower):
                continue
            full_fact = f"[{fact_type}] {fact_text}"
            already = any(
                e.memory_type == fact_type and e.text.lower() == full_fact.lower()
                for e in self.warm.entries
            )
            if already:
                continue
            if fact_type == "name":
                self.warm.entries = [e for e in self.warm.entries if e.memory_type != "name"]
                self.warm._rebuild_index()
            self.warm.add(
                full_fact, importance=importance,
                memory_type=fact_type,
                metadata={"source": "auto_extract", "original": text}
            )

    def _trigger_background_summary(self, force: bool = False):
        """Summarize recent turns on a background worker."""
        def _summarize():
            with self._bg_lock:
                turns = self.hot.get_all()
                if len(turns) < 4:
                    return

                lines = []
                for t in turns:
                    role = "User" if t["role"] == "user" else "Anantum"
                    lines.append(f"{role}: {t['content'][:200]}")

                summary_text = " | ".join(lines[-10:])
                turn_range = f"{len(turns)} turns"

                tags = self._extract_topics(summary_text)

                self.cold.store_summary(
                    summary=summary_text,
                    turn_range=turn_range,
                    tags=tags
                )
                self._summary_pending_turns = 0

        thread = threading.Thread(target=_summarize, daemon=True)
        thread.start()

    def _extract_topics(self, text: str) -> list[str]:
        """Simple frequency-based tag extraction for summaries."""
        stopwords = {"i", "a", "the", "is", "it", "to", "you", "me", "and",
                     "or", "of", "in", "on", "my", "do", "did", "was", "are"}
        words = text.lower().split()
        words = [w.strip(".,?!\"'") for w in words if len(w) > 3 and w not in stopwords]
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        top = sorted(freq, key=freq.get, reverse=True)
        return top[:5]