"""Redis backend: optional, for shared/multi-process caches.

``redis`` is an optional dependency. It is imported lazily so the package works
with no extra install for the SQLite default; importing this module without the
``redis`` package raises a clear error only when you actually try to use it.

Install with: ``pip install embcache[redis]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import Backend

if TYPE_CHECKING:
    import redis as _redis


class RedisBackend(Backend):
    """A :class:`Backend` backed by a Redis server.

    Keys are prefixed (default ``"embcache:"``) so cached vectors don't collide
    with other data in a shared Redis instance.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        prefix: str = "embcache:",
        client: Any = None,
    ) -> None:
        """Connect to Redis.

        Args:
            url: Redis connection URL. Ignored if ``client`` is given.
            prefix: Key namespace prefix for all stored vectors.
            client: An existing ``redis.Redis`` instance to use instead of
                creating one from ``url`` (useful for tests / connection reuse).

        Raises:
            ImportError: If the ``redis`` package is not installed and no
                ``client`` was provided.
        """
        self.prefix = prefix
        if client is not None:
            self._client = client
        else:
            try:
                import redis
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "RedisBackend requires the 'redis' package. "
                    "Install it with: pip install embcache[redis]"
                ) from exc
            self._client = redis.Redis.from_url(url)

    def _k(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def get(self, key: str) -> bytes | None:
        value = self._client.get(self._k(key))
        return value if value is not None else None

    def set(self, key: str, value: bytes) -> None:
        self._client.set(self._k(key), value)

    def has(self, key: str) -> bool:
        return bool(self._client.exists(self._k(key)))

    def close(self) -> None:
        self._client.close()
