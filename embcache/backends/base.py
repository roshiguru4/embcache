"""Backend abstract base class.

A backend is a dumb key -> bytes store. It knows nothing about embeddings,
models, normalization, or stats; the :class:`~embcache.cache.EmbeddingCache`
orchestrates all of that. Keeping the contract this small means adding a new
store (Redis, in-memory, etc.) is a single class implementing three methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Backend(ABC):
    """A key -> bytes store for serialized vectors."""

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
