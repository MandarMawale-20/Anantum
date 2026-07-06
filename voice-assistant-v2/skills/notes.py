# SQLite-backed notes store.

import json
import logging
import sqlite3
import threading
import time
import uuid
import datetime

from config.settings import CONFIG
from skills.base import ToolRegistry

logger = logging.getLogger(__name__)

_conn: sqlite3.Connection | None = None
_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    # Return or create the shared DB connection.
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(str(CONFIG.NOTES_DB_FILE), check_same_thread=False)
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                tags TEXT,
                created_at REAL,
                updated_at REAL
            )
        """)
        _conn.commit()
    return _conn


@ToolRegistry.register("save_note", "Save a note", mode="both")
def save_note(content: str, tags: list = None) -> dict:
    """Persist a new note and return its ID."""
    note_id = str(uuid.uuid4())[:8]
    now = time.time()
    tags = tags or []
    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO notes VALUES (?,?,?,?,?)",
            (note_id, content, json.dumps(tags), now, now)
        )
        conn.commit()
    return {
        "id": note_id,
        "content": content,
        "display": f"Note saved. (ID: {note_id})"
    }


@ToolRegistry.register("get_notes", "Retrieve recent notes", mode="both")
def get_notes(limit: int = 5, search: str = None) -> dict:
    """Return the most recent notes, optionally filtered by a search string."""
    with _lock:
        conn = _get_conn()
        if search:
            cur = conn.execute(
                "SELECT id, content, tags, created_at FROM notes WHERE content LIKE ? ORDER BY created_at DESC LIMIT ?",
                (f"%{search}%", limit)
            )
        else:
            cur = conn.execute(
                "SELECT id, content, tags, created_at FROM notes ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )
        rows = cur.fetchall()

    notes = []
    for r in rows:
        ts = datetime.datetime.fromtimestamp(r[3]).strftime("%b %d, %H:%M")
        notes.append({"id": r[0], "content": r[1], "tags": json.loads(r[2]), "created": ts})

    if not notes:
        display = "No notes found."
    else:
        lines = [f"Your {len(notes)} most recent notes:"]
        for n in notes:
            lines.append(f"  [{n['id']}] {n['content'][:100]}  ({n['created']})")
        display = "\n".join(lines)

    return {"notes": notes, "count": len(notes), "display": display}
