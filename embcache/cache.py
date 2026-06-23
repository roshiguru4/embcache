"""The EmbeddingCache: orchestration and the get_or_compute logic.

This ties the pieces together: normalize -> key -> backend lookup. On a hit it
deserializes the stored vector and records the savings; on a miss it calls the
user's embed function, times it, stores the result, and counts it as a real
(paid) call. The cache never generates embeddings itself — the user always
supplies the embed function.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Sequence
from typing import Any

from . import pricing
from .backends import Backend, SQLiteBackend
from .keys import make_key
from .normalize import normalize
from .report import format_report
from .serialize import dumps, loads
from .stats import Stats

Vector = list[float]
EmbedFn = Callable[[str], Sequence[float]]
BatchEmbedFn = Callable[[list[str]], Sequence[Sequence[float]]]


class EmbeddingCache:
    """A drop-in exact-match cache for text embeddings.

    Example:
        >>> cache = EmbeddingCache(backend="sqlite", path="./emb.db",
        ...                        model="text-embedding-3-small")
        >>> vec = cache.get_or_compute("the cat sat", embed_fn=my_embed)
    """

    def __init__(
        self,
        backend: str | Backend = "sqlite",
        *,
        model: str = "text-embedding-3-small",
        path: str = "./emb.db",
        redis_url: str = "redis://localhost:6379/0",
        lowercase: bool = False,
    ) -> None:
        """Create a cache.

        Args:
            backend: ``"sqlite"`` (default), ``"redis"``, or a ready-made
                :class:`Backend` instance.
            model: Embedding model name. Folded into cache keys and used for
                pricing in the savings report.
            path: SQLite database path (used when ``backend="sqlite"``).
            redis_url: Redis URL (used when ``backend="redis"``).
            lowercase: If true, lowercase text during normalization.
        """
        self.model = model
        self.lowercase = lowercase
        self.stats = Stats()
        self.backend = self._make_backend(backend, path=path, redis_url=redis_url)

    @staticmethod
    def _make_backend(
        backend: str | Backend, *, path: str, redis_url: str
    ) -> Backend:
        if isinstance(backend, Backend):
            return backend
        if backend == "sqlite":
            return SQLiteBackend(path=path)
        if backend == "redis":
            from .backends.redis_backend import RedisBackend

            return RedisBackend(redis_url)
        raise ValueError(
            f"unknown backend {backend!r}; expected 'sqlite', 'redis', or a Backend instance"
        )

    # -- core ---------------------------------------------------------------

    def get_or_compute(self, text: str, embed_fn: EmbedFn) -> Vector:
        """Return the embedding for ``text``, from cache or by computing it.

        On a hit the stored vector is returned with no call to ``embed_fn`` and
        the savings are recorded. On a miss ``embed_fn`` is called, the result
        stored, and the (paid) call recorded.

        Args:
            text: The text to embed.
            embed_fn: Callable that maps a single string to a vector.

        Returns:
            The embedding as a list of floats.
        """
        key = self._key(text)
        blob = self.backend.get(key)
        if blob is not None:
            self._record_hit(text)
            return loads(blob)

        vector, _ = self._compute_and_store(text, key, embed_fn)
        return vector

    def get_or_compute_many(
        self, texts: Sequence[str], embed_fn: BatchEmbedFn
    ) -> list[Vector]:
        """Batch variant of :meth:`get_or_compute`.

        Only the cache misses are passed to ``embed_fn``, and duplicate misses
        within the batch are de-duplicated so ``embed_fn`` is called once per
        unique text. Results are returned in the original order of ``texts``.

        Args:
            texts: The texts to embed.
            embed_fn: Callable mapping a list of strings to a list of vectors,
                in the same order. Only ever receives the unique miss texts.

        Returns:
            One vector per input text, in order.
        """
        results: list[Vector | None] = [None] * len(texts)
        keys = [self._key(t) for t in texts]

        # Map each unique missing key -> the first text/indices needing it.
        miss_texts: dict[str, str] = {}
        miss_indices: dict[str, list[int]] = {}

        for i, (text, key) in enumerate(zip(texts, keys)):
            blob = self.backend.get(key)
            if blob is not None:
                results[i] = loads(blob)
                self._record_hit(text)
            else:
                if key not in miss_texts:
                    miss_texts[key] = text
                    miss_indices[key] = []
                miss_indices[key].append(i)

        if miss_texts:
            unique_keys = list(miss_texts)
            unique_texts = [miss_texts[k] for k in unique_keys]
            start = time.perf_counter()
            vectors = embed_fn(unique_texts)
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._validate_batch(unique_texts, vectors)

            # Attribute latency evenly across the unique misses computed.
            per_miss_ms = elapsed_ms / len(unique_keys)
            for key, vec in zip(unique_keys, vectors):
                vector = list(vec)
                self.backend.set(key, dumps(vector))
                self.stats.record_miss(latency_ms=per_miss_ms)
                for idx in miss_indices[key]:
                    results[idx] = vector

        # All slots are filled by construction.
        return [v for v in results if v is not None]

    # -- decorator ----------------------------------------------------------

    def wrap(self, fn: EmbedFn) -> EmbedFn:
        """Decorator that turns a single-text embed function into a cached one.

        Example:
            >>> @cache.wrap
            ... def embed(text: str) -> list[float]:
            ...     return client.embeddings.create(input=text, model=...).data[0].embedding
        """

        @functools.wraps(fn)
        def wrapper(text: str) -> Vector:
            return self.get_or_compute(text, fn)

        return wrapper

    # -- reporting ----------------------------------------------------------

    def report(self) -> str:
        """Return a formatted savings report for the session."""
        return format_report(self.stats, model=self.model)

    def close(self) -> None:
        """Close the underlying backend."""
        self.backend.close()

    def __enter__(self) -> EmbeddingCache:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- internals ----------------------------------------------------------

    def _key(self, text: str) -> str:
        return make_key(text, self.model, lowercase=self.lowercase)

    def _tokens(self, text: str) -> int:
        return pricing.estimate_tokens(normalize(text, lowercase=self.lowercase))

    def _record_hit(self, text: str) -> None:
        tokens = self._tokens(text)
        self.stats.record_hit(
            tokens=tokens, dollars=pricing.cost_for_tokens(self.model, tokens)
        )

    def _compute_and_store(
        self, text: str, key: str, embed_fn: EmbedFn
    ) -> tuple[Vector, float]:
        start = time.perf_counter()
        vector = list(embed_fn(text))
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.backend.set(key, dumps(vector))
        self.stats.record_miss(latency_ms=elapsed_ms)
        return vector, elapsed_ms

    @staticmethod
    def _validate_batch(
        texts: Sequence[str], vectors: Sequence[Sequence[float]]
    ) -> None:
        if len(vectors) != len(texts):
            raise ValueError(
                f"embed_fn returned {len(vectors)} vectors for {len(texts)} texts; "
                "a batch embed function must return one vector per input, in order"
            )
