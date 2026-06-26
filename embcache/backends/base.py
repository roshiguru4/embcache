"""Backend abstract base class.

A backend is a dumb key -> bytes store. It knows nothing about embeddings,
models, normalization, or stats; the :class:`~embcache.cache.EmbeddingCache`
orchestrates all of that. Keeping the *core* contract small means adding a new
store (Redis, in-memory, etc.) is a single class implementing three methods:
``get`` / ``set`` / ``has``.

On top of those three, the base class provides default implementations for
persistence and introspection (lifetime stats, metadata, counting, clearing).
The defaults are written purely in terms of ``get``/``set`` using a reserved key
namespace, so a custom backend that implements only the three core methods still
gets persistent savings tracking for free. The shipped SQLite and Redis backends
override these with native, atomic, more efficient versions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

# The five numeric counters that make up a persisted savings report. Kept here
# so backends and the cache agree on the field set.
STAT_KEYS: tuple[str, ...] = (
    "hits",
    "misses",
    "tokens_saved",
    "dollars_saved",
    "latency_saved_ms",
)

# Reserved key prefixes for the get/set-based default implementations. Cache
# keys are SHA-256 hex digests ([0-9a-f]), so a NUL byte can never collide with
# a real vector key.
_STAT_PREFIX = "\x00embcache\x00stat\x00"
_META_PREFIX = "\x00embcache\x00meta\x00"


class Backend(ABC):
    """A key -> bytes store for serialized vectors."""

    # -- core contract ------------------------------------------------------

    @abstractmethod
    def get(self, key: str) -> bytes | None:
        """Return the stored bytes for ``key``, or ``None`` if absent."""

    @abstractmethod
    def set(self, key: str, value: bytes) -> None:
        """Store ``value`` bytes under ``key``, overwriting any existing value."""

    @abstractmethod
    def has(self, key: str) -> bool:
        """Return whether ``key`` is present in the store."""

    def close(self) -> None:
        """Release any resources held by the backend.

        Default is a no-op; backends with open connections/handles override it.
        """

    # -- introspection (default impls; override for efficiency) -------------

    def count(self) -> int | None:
        """Return the number of cached vectors, or ``None`` if not countable.

        The base store has no efficient way to count its own keys, so it returns
        ``None`` ("unknown"). SQLite and Redis override this.
        """
        return None

    def size_bytes(self) -> int | None:
        """Return the storage footprint in bytes, or ``None`` if unknown."""
        return None

    def clear(self) -> int:
        """Delete all cached vectors and return how many were removed.

        Lifetime stats and metadata are intentionally preserved. The base class
        cannot enumerate its own keys, so this must be overridden.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support clear()"
        )

    # -- persisted stats (default impls via reserved keys) ------------------

    def increment_stats(self, deltas: Mapping[str, float]) -> None:
        """Add ``deltas`` (per :data:`STAT_KEYS`) to the persisted counters.

        Default implementation is a read-modify-write over reserved keys, which
        is correct for single-process use. Backends that can increment
        atomically (SQLite, Redis) override this so concurrent processes sharing
        one store don't lose updates.
        """
        for key, delta in deltas.items():
            if delta == 0:
                continue
            current = self._read_stat(key)
            self._write_stat(key, current + delta)

    def read_stats(self) -> dict[str, float]:
        """Return all persisted lifetime counters as a ``{name: value}`` dict."""
        return {key: self._read_stat(key) for key in STAT_KEYS}

    def reset_stats(self) -> None:
        """Zero all persisted lifetime counters."""
        for key in STAT_KEYS:
            self._write_stat(key, 0.0)

    # -- metadata (default impls via reserved keys) -------------------------

    def get_meta(self, key: str) -> str | None:
        """Return the metadata string stored under ``key``, or ``None``."""
        blob = self.get(_META_PREFIX + key)
        return blob.decode("utf-8") if blob is not None else None

    def set_meta(self, key: str, value: str) -> None:
        """Store the metadata string ``value`` under ``key``."""
        self.set(_META_PREFIX + key, value.encode("utf-8"))

    # -- helpers for the default stat impls ---------------------------------

    def _read_stat(self, name: str) -> float:
        blob = self.get(_STAT_PREFIX + name)
        return float(blob.decode("utf-8")) if blob is not None else 0.0

    def _write_stat(self, name: str, value: float) -> None:
        self.set(_STAT_PREFIX + name, repr(value).encode("utf-8"))
