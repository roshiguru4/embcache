"""SQLite backend: the zero-config default.

Stores serialized vectors in a single local file (or in-memory). One table,
``key TEXT PRIMARY KEY`` -> ``value BLOB``. WAL mode is enabled so reads and
writes don't block each other, which matters when the same cache file is shared
across processes (e.g. parallel test runs).
"""

from __future__ import annotations

import sqlite3
import threading

from .base import Backend


class SQLiteBackend(Backend):
    """A :class:`Backend` backed by a local SQLite database file."""

    def __init__(self, path: str = "./emb.db") -> None:
        """Open (or create) the SQLite database at ``path``.

        Args:
            path: Filesystem path to the database file. Use ``":memory:"`` for
                an ephemeral in-memory store (handy for tests).
        """
        self.path = path
        # check_same_thread=False + a lock lets the cache be used from worker
        # threads; the lock serializes access to the single connection.
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        if path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, value BLOB NOT NULL)"
        )
        self._conn.commit()

    def get(self, key: str) -> bytes | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM embeddings WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row is not None else None

    def set(self, key: str, value: bytes) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO embeddings (key, value) VALUES (?, ?)",
                (key, value),
            )
            self._conn.commit()

    def has(self, key: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM embeddings WHERE key = ?", (key,)
            ).fetchone()
        return row is not None

    def close(self) -> None:
        with self._lock:
            self._conn.close()
