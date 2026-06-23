"""Storage backends for embcache."""

from __future__ import annotations

from .base import Backend
from .sqlite_backend import SQLiteBackend

__all__ = ["Backend", "SQLiteBackend", "RedisBackend"]


def __getattr__(name: str) -> object:
    # Lazy import so the optional redis dependency is only required when used.
    if name == "RedisBackend":
        from .redis_backend import RedisBackend

        return RedisBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
