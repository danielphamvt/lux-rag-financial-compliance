"""SQLite-backed persistence for chat conversations (stdlib sqlite3 only).

Reads its env vars directly instead of importing config so tests can use this
module without pulling in the heavy model/Qdrant stack:
- CHAT_HISTORY_PATH: database file (default db/chat_history.db)
- MAX_STORED_CONVERSATIONS: retention limit, oldest conversations pruned (default 20)

Each message stores both `content` (the display HTML exactly as shown by the
Chatbot) and `plain` (clean text used to re-seed agent memory when a saved
conversation is continued after a restart).
"""
import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path

_DEFAULT_DB_PATH = str(Path(__file__).resolve().parent / "chat_history.db")
DB_PATH = os.getenv("CHAT_HISTORY_PATH", _DEFAULT_DB_PATH)

# Millisecond-resolution timestamps so recency ordering survives bursts
_NOW = "strftime('%Y-%m-%d %H:%M:%f', 'now')"

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT ({_NOW}),
    updated_at TEXT NOT NULL DEFAULT ({_NOW})
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    plain TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT ({_NOW})
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id);
"""

_COMMAND_PREFIX = re.compile(r"^/\w+\s+")
_INITIALIZED_DBS: set[str] = set()


def _keep_count() -> int:
    try:
        return max(1, int(os.getenv("MAX_STORED_CONVERSATIONS", "20")))
    except (TypeError, ValueError):
        return 20


def _conn() -> sqlite3.Connection:
    directory = os.path.dirname(DB_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)
    db_existed = os.path.exists(DB_PATH)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA foreign_keys = ON")
    if DB_PATH not in _INITIALIZED_DBS or not db_existed:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        _INITIALIZED_DBS.add(DB_PATH)
    return conn


def make_title(text: str, limit: int = 48) -> str:
    """Conversation title: drop any /command prefix, collapse whitespace, truncate."""
    title = " ".join(_COMMAND_PREFIX.sub("", (text or "").strip()).split())
    if len(title) > limit:
        head = title[:limit]
        cut = head.rsplit(" ", 1)[0] if " " in head else head
        title = cut.rstrip(" ,;:.-") + "…"
    return title or "New chat"


def _prune(conn: sqlite3.Connection) -> None:
    """Keep only the newest `_keep_count()` conversations (messages cascade)."""
    conn.execute(
        "DELETE FROM conversations WHERE id NOT IN ("
        "  SELECT id FROM conversations ORDER BY updated_at DESC, id DESC LIMIT ?"
        ")",
        (_keep_count(),),
    )


def create_conversation(title: str) -> int:
    with closing(_conn()) as conn, conn:
        cur = conn.execute("INSERT INTO conversations (title) VALUES (?)", (title,))
        _prune(conn)
        return cur.lastrowid


def add_message(conv_id: int, role: str, content: str, plain: str = "") -> None:
    """Append a message, bump the conversation's recency, and enforce retention."""
    with closing(_conn()) as conn, conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, plain) VALUES (?, ?, ?, ?)",
            (conv_id, role, content, plain),
        )
        conn.execute(
            f"UPDATE conversations SET updated_at = {_NOW} WHERE id = ?", (conv_id,)
        )
        _prune(conn)


def get_messages(conv_id: int) -> list:
    """Display messages (role + content) in conversation order."""
    with closing(_conn()) as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id",
            (conv_id,),
        ).fetchall()
    return [{"role": role, "content": content} for role, content in rows]


def plain_messages(conv_id: int) -> list:
    """(role, plain) pairs in conversation order, for agent-memory re-seeding."""
    with closing(_conn()) as conn:
        rows = conn.execute(
            "SELECT role, plain FROM messages WHERE conversation_id = ? ORDER BY id",
            (conv_id,),
        ).fetchall()
    return list(rows)


def delete_conversation(conv_id: int) -> None:
    """Remove a conversation and (via FK cascade) all of its messages."""
    with closing(_conn()) as conn, conn:
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))


def list_recent() -> list:
    """[(title, conversation_id)] newest first, capped at the retention limit."""
    with closing(_conn()) as conn:
        rows = conn.execute(
            "SELECT title, id FROM conversations ORDER BY updated_at DESC, id DESC LIMIT ?",
            (_keep_count(),),
        ).fetchall()
    return list(rows)
