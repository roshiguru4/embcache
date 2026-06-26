"""Redis backend: optional, for shared/multi-process caches.

``redis`` is an optional dependency. It is imported lazily so the package works
with no extra install for the SQLite default; importing this module without the
``redis`` package raises a clear error only when you actually try to use it.

Vectors are stored at ``{prefix}{key}``. Cumulative savings live in a single
hash at ``{prefix}\\x00stats`` (incremented atomically with ``HINCRBYFLOAT``) and
metadata in ``{prefix}\\x00meta``. Cache keys are SHA-256 hex digests, so the NUL
byte in the reserved hash names can never collide with a real vector key.

Install with: ``pip install embcache[redis]``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .base import STAT_KEYS, Backend

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
        self._stats_key = f"{prefix}\x00stats"
        self._meta_key = f"{prefix}\x00meta"
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

    # -- core ---------------------------------------------------------------

    def get(self, key: str) -> bytes | None:
        value = self._client.get(self._k(key))
        return value if value is not None else None

    def set(self, key: str, value: bytes) -> None:
        self._client.set(self._k(key), value)

    def has(self, key: str) -> bool:
        return bool(self._client.exists(self._k(key)))

    def close(self) -> None:
        self._client.close()

    # -- introspection ------------------------------------------------------

    def _vector_keys(self) -> list[bytes]:
        """All vector keys under the prefix, excluding the reserved hashes."""
        pattern = f"{self.prefix}*"
        return [k for k in self._client.scan_iter(match=pattern) if b"\x00" not in k]

    def count(self) -> int:
        return len(self._vector_keys())

    def clear(self) -> int:
        keys = self._vector_keys()
        if keys:
            self._client.delete(*keys)
        return len(keys)

    # size_bytes() inherits the base default (None): Redis has no cheap
    # per-namespace byte total.

    # -- persisted stats ----------------------------------------------------

    def increment_stats(self, deltas: Mapping[str, float]) -> None:
        pipe = self._client.pipeline()
        wrote = False
        for key, delta in deltas.items():
            if delta == 0:
                continue
            pipe.hincrbyfloat(self._stats_key, key, float(delta))
            wrote = True
        if wrote:
            pipe.execute()

    def read_stats(self) -> dict[str, float]:
        raw = self._client.hgetall(self._stats_key)
        stored = {
            (k.decode() if isinstance(k, bytes) else k): float(v)
            for k, v in raw.items()
        }
        return {key: stored.get(key, 0.0) for key in STAT_KEYS}

    def reset_stats(self) -> None:
        self._client.delete(self._stats_key)

    # -- metadata -----------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        value = self._client.hget(self._meta_key, key)
        if value is None:
            return None
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    def set_meta(self, key: str, value: str) -> None:
        self._client.hset(self._meta_key, key, value)
