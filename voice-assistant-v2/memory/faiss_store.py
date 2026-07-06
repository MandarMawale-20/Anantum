# Warm memory store backed by FAISS and sentence embeddings.

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from config.settings import CONFIG

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


@dataclass
class MemoryEntry:
    text: str
    timestamp: float = field(default_factory=time.time)
    importance: float = 1.0
    memory_type: str = "general"
    metadata: dict = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)


class WarmStore:
    """FAISS HNSW index for semantic similarity search over recent memories."""

    def __init__(self,
                 index_path: Path = None,
                 meta_path: Path = None,
                 embedding_model: str = None):
        self._index_path = index_path or CONFIG.WARM_FAISS_INDEX
        self._meta_path  = meta_path  or CONFIG.WARM_METADATA_FILE
        self._model_name = embedding_model or CONFIG.embedding_model
        self._model = None
        self._dim = 384
        self.entries: list[MemoryEntry] = []
        self._index = None
        self._lock = threading.Lock()
        self._save_timer: Optional[threading.Timer] = None
        self._save_lock = threading.Lock()

        self._load_or_init()
        self._load_model()

    def _load_model(self):
        if SentenceTransformer is None:
            logger.warning("sentence-transformers not installed; memory embeddings disabled")
            return
        try:
            self._model = SentenceTransformer(self._model_name)
            logger.info("Embedding model loaded: %s", self._model_name)
        except Exception as e:
            logger.error("Failed to load embedding model: %s", e)

    def _load_or_init(self):
        if faiss is None:
            logger.warning("faiss not installed; warm store disabled")
            return
        if self._index_path.exists() and self._meta_path.exists():
            try:
                self._index = faiss.read_index(str(self._index_path))
                meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
                self.entries = [MemoryEntry(**e) for e in meta]
                logger.info("Warm store loaded: %d entries", len(self.entries))
                return
            except Exception as e:
                logger.warning("Failed to load warm store: %s — trying backup", e)
                # Try backup files before giving up
                bak_idx = self._index_path.with_suffix(".bak.faiss")
                bak_meta = self._meta_path.with_suffix(".bak.json")
                if bak_idx.exists() and bak_meta.exists():
                    try:
                        self._index = faiss.read_index(str(bak_idx))
                        meta = json.loads(bak_meta.read_text(encoding="utf-8"))
                        self.entries = [MemoryEntry(**e) for e in meta]
                        logger.info("Warm store restored from backup: %d entries", len(self.entries))
                        return
                    except Exception as e2:
                        logger.warning("Backup also failed: %s — starting fresh", e2)
        self._init_index()

    def _init_index(self):
        if faiss is None:
            return
        self._index = faiss.IndexHNSWFlat(self._dim, 32)
        self._index.hnsw.efConstruction = 64
        self._index.hnsw.efSearch = 32

    def _embed(self, text: str) -> Optional[np.ndarray]:
        if self._model is None:
            return None
        try:
            vec = self._model.encode(
                [text],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return vec[0].astype(np.float32)
        except Exception as e:
            logger.debug("Embedding failed: %s", e)
            return None

    def _rebuild_index(self):
        if faiss is None:
            return
        self._init_index()
        for entry in self.entries:
            if entry.embedding:
                vec = np.array([entry.embedding], dtype=np.float32)
                self._index.add(vec)

    def add(
        self,
        text: str,
        importance: float = 1.0,
        memory_type: str = "general",
        metadata: dict = None,
        embedding: list = None,
    ) -> None:
        if faiss is None:
            return
        vec = np.array([embedding], dtype=np.float32) if embedding else self._embed(text)
        if vec is None:
            return
        entry = MemoryEntry(
            text=text,
            timestamp=time.time(),
            importance=float(importance),
            memory_type=memory_type,
            metadata=metadata or {},
            embedding=vec[0].tolist() if isinstance(vec, np.ndarray) else embedding,
        )
        with self._lock:
            self.entries.append(entry)
            self._index.add(vec.reshape(1, -1))
        self._schedule_save()

    def _schedule_save(self, delay: float = 5.0) -> None:
        with self._save_lock:
            if self._save_timer is not None:
                self._save_timer.cancel()
            self._save_timer = threading.Timer(delay, self._save)
            self._save_timer.daemon = True
            self._save_timer.start()

    def flush_save(self) -> None:
        with self._save_lock:
            if self._save_timer is not None:
                self._save_timer.cancel()
                self._save_timer = None
        self._save()

    def _save(self) -> None:
        if faiss is None or self._index is None:
            return
        try:
            with self._lock:
                # Write to temp files first
                tmp_idx = self._index_path.with_suffix(".tmp.faiss")
                faiss.write_index(self._index, str(tmp_idx))
                tmp_meta = self._meta_path.with_suffix(".tmp.json")
                meta = [
                    {
                        "text": e.text,
                        "timestamp": e.timestamp,
                        "importance": e.importance,
                        "memory_type": e.memory_type,
                        "metadata": e.metadata,
                        "embedding": e.embedding,
                    }
                    for e in self.entries
                ]
                tmp_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

                # Rotate backup before replacing live files
                bak_idx = self._index_path.with_suffix(".bak.faiss")
                bak_meta = self._meta_path.with_suffix(".bak.json")
                if self._index_path.exists():
                    self._index_path.replace(bak_idx)
                if self._meta_path.exists():
                    self._meta_path.replace(bak_meta)

                tmp_idx.replace(self._index_path)
                tmp_meta.replace(self._meta_path)
            logger.debug("Warm store saved: %d entries", len(self.entries))
        except Exception as e:
            logger.error("Failed to save warm store: %s", e)

    def search(self, query: str, top_k: int = 5) -> list[tuple[MemoryEntry, float]]:
        if faiss is None or self._index is None or self._model is None:
            return []
        query_vec = self._embed(query)
        if query_vec is None or self._index.ntotal == 0:
            return []
        try:
            with self._lock:
                k = min(top_k, len(self.entries))
                distances, indices = self._index.search(query_vec.reshape(1, -1), k)

            results: list[tuple[MemoryEntry, float]] = []
            metric_type = getattr(self._index, "metric_type", None)
            for rank, i in enumerate(indices[0]):
                if not (0 <= i < len(self.entries)):
                    continue
                dist = float(distances[0][rank])
                # Convert L2 distance to a bounded similarity-like score.
                if metric_type == getattr(faiss, "METRIC_INNER_PRODUCT", -1):
                    score = dist
                else:
                    score = 1.0 / (1.0 + max(dist, 0.0))
                results.append((self.entries[i], score))
            return results
        except Exception as e:
            logger.debug("FAISS search failed: %s", e)
            return []

    def prune(self, max_entries: int) -> None:
        if len(self.entries) <= max_entries:
            return
        with self._lock:
            self.entries = sorted(
                self.entries,
                key=lambda e: (e.importance, e.timestamp),
                reverse=True
            )[:max_entries]
            self._rebuild_index()
        self.flush_save()
        logger.info("Pruned warm store to %d entries", len(self.entries))

    def __len__(self) -> int:
        return len(self.entries)
