from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from src.constants import PROJECT_ROOT


DEFAULT_FEEDBACK_DB_PATH = PROJECT_ROOT / "data" / "feedback.db"


def _get_db_path() -> Path:
    """
    Resolve the SQLite database path for feedback events.

    Uses FEEDBACK_DB_PATH if set; otherwise falls back to data/feedback.db.

    Returns:
        Path to the feedback database file.
    """
    env_path = Path(str(Path.cwd()))  # placeholder to avoid mypy complaints if os is missing
    try:
        import os

        value = os.getenv("FEEDBACK_DB_PATH")
        if value:
            return Path(value)
    except Exception:
        # Fall back to default if environment inspection fails for any reason.
        pass
    return DEFAULT_FEEDBACK_DB_PATH


def _ensure_parent_dir(path: Path) -> None:
    """Create parent directory of path if it does not exist.

    Args:
        path: File or directory path whose parent should exist.
    """
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def init_db() -> Path:
    """
    Initialize the feedback SQLite database if it does not already exist.

    Returns:
        Resolved Path to the database file.
    """
    db_path = _get_db_path().resolve()
    _ensure_parent_dir(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT,
                event_type TEXT NOT NULL,
                user_id TEXT,
                product_id TEXT NOT NULL,
                user_context_hash TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_request ON feedback_events(request_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_event_type ON feedback_events(event_type)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback_events(created_at)"
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


@dataclass
class FeedbackEventRecord:
    request_id: Optional[str]
    event_type: str
    product_id: str
    user_id: Optional[str] = None
    user_context_hash: Optional[str] = None
    metadata: Optional[Mapping[str, Any]] = None
    created_at: Optional[datetime] = None


def _serialize_metadata(metadata: Optional[Mapping[str, Any]]) -> Optional[str]:
    if metadata is None:
        return None
    try:
        return json.dumps(metadata, ensure_ascii=False)
    except TypeError:
        # Fallback: best-effort string conversion
        return json.dumps(str(metadata), ensure_ascii=False)


def record_event(event: FeedbackEventRecord) -> None:
    """
    Insert a single feedback event into the SQLite store.

    Args:
        event: FeedbackEventRecord to insert.
    """
    db_path = init_db()
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO feedback_events (
                request_id,
                event_type,
                user_id,
                product_id,
                user_context_hash,
                metadata,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))
            """,
            (
                event.request_id,
                event.event_type,
                event.user_id,
                event.product_id,
                event.user_context_hash,
                _serialize_metadata(event.metadata),
                event.created_at.isoformat() if event.created_at else None,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def record_events(events: Iterable[FeedbackEventRecord]) -> None:
    """
    Insert multiple feedback events in a single transaction.

    Args:
        events: Iterable of FeedbackEventRecord to insert.
    """
    events_list = list(events)
    if not events_list:
        return
    db_path = init_db()
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        rows = [
            (
                e.request_id,
                e.event_type,
                e.user_id,
                e.product_id,
                e.user_context_hash,
                _serialize_metadata(e.metadata),
                e.created_at.isoformat() if e.created_at else None,
            )
            for e in events_list
        ]
        cur.executemany(
            """
            INSERT INTO feedback_events (
                request_id,
                event_type,
                user_id,
                product_id,
                user_context_hash,
                metadata,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()

