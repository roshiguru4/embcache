"""Async API: aget_or_compute / aget_or_compute_many / awrap.

Tests drive the coroutines with ``asyncio.run`` so no pytest plugin is needed.
The async paths share the hit/miss/dedup machinery with the sync ones, so these
focus on the async-specific contract: the embed coroutine is awaited only on a
miss, and the batch guarantees still hold.
"""

import asyncio

from embcache import EmbeddingCache


class AsyncEmbedder:
    def __init__(self):
        self.calls = 0

    async def __call__(self, text):
        self.calls += 1
        await asyncio.sleep(0)  # yield to the loop, like a real network call
        return [float(len(text)), 0.5]


class AsyncBatchEmbedder:
    def __init__(self):
        self.calls = 0
        self.batches_seen = []

    async def __call__(self, texts):
        self.calls += 1
        self.batches_seen.append(list(texts))
        await asyncio.sleep(0)
        return [[float(len(t)), 1.0] for t in texts]


def test_aget_or_compute_miss_then_hit():
    async def run():
        c = EmbeddingCache(backend="memory")
        embed = AsyncEmbedder()
        first = await c.aget_or_compute("hello", embed)
        second = await c.aget_or_compute("hello", embed)
        c.close()
        return embed.calls, first, second, c.stats

    calls, first, second, stats = asyncio.run(run())
    assert calls == 1  # second served from cache, coroutine not awaited
    assert first == second
    assert stats.hits == 1 and stats.misses == 1


def test_awrap_decorator():
    async def run():
        c = EmbeddingCache(backend="memory")
        embed = AsyncEmbedder()

        @c.awrap
        async def embed_text(text):
            return await embed(text)

        v1 = await embed_text("hi there")
        v2 = await embed_text("hi there")
        c.close()
        return embed.calls, v1, v2, embed_text.__name__

    calls, v1, v2, name = asyncio.run(run())
    assert calls == 1
    assert v1 == v2
    assert name == "embed_text"  # functools.wraps preserved


def test_aget_or_compute_many_misses_dedupe_order():
    async def run():
        c = EmbeddingCache(backend="memory")
        embed = AsyncBatchEmbedder()
        # Prime "a".
        await c.aget_or_compute_many(["a"], embed)
        embed2 = AsyncBatchEmbedder()
        out = await c.aget_or_compute_many(["a", "b", "b", "ccc"], embed2)
        c.close()
        return out, embed2.batches_seen

    out, batches = asyncio.run(run())
    assert len(out) == 4
    assert out[1] == out[2]  # duplicate "b" reused
    assert [v[0] for v in out] == [1.0, 1.0, 1.0, 3.0]  # order preserved
    # "a" was cached; only unique misses b, ccc computed, once.
    assert batches == [["b", "ccc"]]


def test_async_and_sync_share_one_cache():
    """A vector written synchronously is an async hit, and vice versa."""

    async def run():
        c = EmbeddingCache(backend="memory")
        c.get_or_compute("shared", lambda t: [9.0])  # sync miss, stores it
        embed = AsyncEmbedder()
        vec = await c.aget_or_compute("shared", embed)  # async hit
        c.close()
        return embed.calls, vec

    calls, vec = asyncio.run(run())
    assert calls == 0  # async path found the synchronously-stored vector
    assert vec == [9.0]


def test_async_context_manager_flushes():
    db_stats = {}

    async def run(tmpdb):
        async with EmbeddingCache(backend="sqlite", path=tmpdb, model="m") as c:
            embed = AsyncEmbedder()
            await c.aget_or_compute("x", embed)
            await c.aget_or_compute("x", embed)  # hit
        # On exit the cache flushed; reopen and read lifetime.
        reopened = EmbeddingCache(backend="sqlite", path=tmpdb, model="m")
        life = reopened.lifetime_stats()
        reopened.close()
        return life

    import tempfile
    import os

    with tempfile.TemporaryDirectory() as d:
        life = asyncio.run(run(os.path.join(d, "emb.db")))
        assert life.hits == 1
