# SQLite archive for conversation summaries.

import json
import logging
import time
import threading
import sqlite3

from config.settings import CONFIG
from memory.faiss_store import MemoryEntry

logger = logging.getLogger(__name__)


class ColdArchive:

    DB_FILE = CONFIG.COLD_ARCHIVE_DB

    def __init__(self):
        self.conn = sqlite3.connect(str(self.DB_FILE), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
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

                -- Keep FTS5 index in sync with the summaries table
                CREATE TRIGGER IF NOT EXISTS summaries_ai
                AFTER INSERT ON summaries BEGIN
                    INSERT INTO summaries_fts(rowid, summary, tags)
                    VALUES (new.id, new.summary, new.tags);
                END;

                CREATE TRIGGER IF NOT EXISTS summaries_ad
                AFTER DELETE ON summaries BEGIN
                    INSERT INTO summaries_fts(summaries_fts, rowid, summary, tags)
                    VALUES ('delete', old.id, old.summary, old.tags);
                END;

                CREATE TRIGGER IF NOT EXISTS summaries_au
                AFTER UPDATE ON summaries BEGIN
                    INSERT INTO summaries_fts(summaries_fts, rowid, summary, tags)
                    VALUES ('delete', old.id, old.summary, old.tags);
                    INSERT INTO summaries_fts(rowid, summary, tags)
                    VALUES (new.id, new.summary, new.tags);
                END;
            """)

    def store_summary(self, summary: str, turn_range: str = "", tags: list = None):
        """Persist a conversation summary; FTS5 triggers keep the index in sync."""
        safe_summary = summary.encode("utf-8", errors="replace").decode("utf-8")
        with self._lock:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO summaries (summary, turn_range, created_at, tags) VALUES (?,?,?,?)",
                    (safe_summary, turn_range, time.time(), json.dumps(tags or []))
                )

    def archive_memory(self, entry: MemoryEntry):
        # Move a warm memory entry into the archive.
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
        """Search summaries using FTS5 full-text index with a LIKE fallback."""
        with self._lock:
            # Escape quotes for FTS5.
            fts_query = keyword.replace('"', '""')
            try:
                cur = self.conn.execute(
                    """
                    SELECT s.summary, s.turn_range, s.created_at
                    FROM summaries_fts
                    JOIN summaries s ON summaries_fts.rowid = s.id
                    WHERE summaries_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, limit)
                )
            except sqlite3.OperationalError:
                # Fall back to LIKE when FTS query parsing fails.
                cur = self.conn.execute(
                    "SELECT summary, turn_range, created_at FROM summaries "
                    "WHERE summary LIKE ? ORDER BY created_at DESC LIMIT ?",
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
