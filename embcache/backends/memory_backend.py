"""In-memory backend: a process-local dict store.

Nothing is persisted — when the process exits, the cache is gone. That makes it
perfect for tests, ephemeral runs, and notebooks where you just want duplicate
calls within one session to be free without leaving a file behind. It's also the
smallest complete implementation of the :class:`Backend` contract, so it doubles
as the reference for writing your own.

Vectors, persisted stats, and metadata live in three separate dicts, so counting
entries never trips over the stats/meta bookkeeping.
"""

from __future__ import annotations

from collections.abc import Mapping

from .base import STAT_KEYS, Backend


class MemoryBackend(Backend):
    """A :class:`Backend` backed by in-process dictionaries."""

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}
        self._stats: dict[str, float] = {}
        self._meta: dict[str, str] = {}

    # -- core ---------------------------------------------------------------

    def get(self, key: str) -> bytes | None:
        return self._data.get(key)

    def set(self, key: str, value: bytes) -> None:
        self._data[key] = value

    def has(self, key: str) -> bool:
        return key in self._data

    # -- introspection ------------------------------------------------------

    def count(self) -> int:
        return len(self._data)

    def size_bytes(self) -> int:
        # Bytes held by the stored vectors; a fair proxy for the footprint.
        return sum(len(v) for v in self._data.values())

    def clear(self) -> int:
        removed = len(self._data)
        self._data.clear()
        return removed

    # -- persisted stats ----------------------------------------------------

    def increment_stats(self, deltas: Mapping[str, float]) -> None:
        for key, delta in deltas.items():
            if delta == 0:
                continue
            self._stats[key] = self._stats.get(key, 0.0) + float(delta)

    def read_stats(self) -> dict[str, float]:
        return {key: self._stats.get(key, 0.0) for key in STAT_KEYS}

    def reset_stats(self) -> None:
        self._stats.clear()

    # -- metadata -----------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        return self._meta.get(key)

    def set_meta(self, key: str, value: str) -> None:
        self._meta[key] = value
