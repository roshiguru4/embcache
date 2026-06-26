"""SQLite backend: the zero-config default.

Stores serialized vectors in a single local file (or in-memory). The main table
is ``embeddings`` (``key TEXT PRIMARY KEY`` -> ``value BLOB``). Two small side
tables hold cumulative savings (``stats``) and free-form metadata (``meta``) so
the "prove how much you saved" report survives across runs. WAL mode is enabled
so reads and writes don't block each other, which matters when the same cache
file is shared across processes (e.g. parallel test runs).
"""

from __future__ import annotations

import os
import sqlite3
import threading
from collections.abc import Mapping

from .base import STAT_KEYS, Backend


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
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS stats (key TEXT PRIMARY KEY, value REAL NOT NULL DEFAULT 0)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        self._conn.commit()

    # -- core ---------------------------------------------------------------

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

    # -- introspection ------------------------------------------------------

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        return int(row[0])

    def size_bytes(self) -> int | None:
        if self.path == ":memory:":
            with self._lock:
                page_count = self._conn.execute("PRAGMA page_count").fetchone()[0]
                page_size = self._conn.execute("PRAGMA page_size").fetchone()[0]
            return int(page_count) * int(page_size)
        # Include the WAL and shared-memory sidecar files if present.
        total = 0
        for suffix in ("", "-wal", "-shm"):
            try:
                total += os.path.getsize(self.path + suffix)
            except OSError:
                pass
        return total

    def clear(self) -> int:
        with self._lock:
            removed = int(
                self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            )
            self._conn.execute("DELETE FROM embeddings")
            self._conn.commit()
        return removed

    # -- persisted stats ----------------------------------------------------

    def increment_stats(self, deltas: Mapping[str, float]) -> None:
        items = [(k, float(v)) for k, v in deltas.items() if v != 0]
        if not items:
            return
        with self._lock:
            # UPSERT: atomic read-add-write per row within the transaction, so
            # concurrent processes accumulate instead of clobbering each other.
            self._conn.executemany(
                "INSERT INTO stats (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = value + excluded.value",
                items,
            )
            self._conn.commit()

    def read_stats(self) -> dict[str, float]:
        with self._lock:
            rows = self._conn.execute("SELECT key, value FROM stats").fetchall()
        stored = {key: float(value) for key, value in rows}
        return {key: stored.get(key, 0.0) for key in STAT_KEYS}

    def reset_stats(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM stats")
            self._conn.commit()

    # -- metadata -----------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row is not None else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (key, value),
            )
            self._conn.commit()
