"""The in-memory backend, exercised through EmbeddingCache."""

from embcache import EmbeddingCache
from embcache.backends import MemoryBackend


def embed(text):
    return [float(len(text)), 0.5]


def test_memory_backend_caches_within_session():
    c = EmbeddingCache(backend="memory")
    calls = {"n": 0}

    def counting(text):
        calls["n"] += 1
        return embed(text)

    c.get_or_compute("repeat me", counting)
    c.get_or_compute("repeat me", counting)
    assert calls["n"] == 1  # second was a hit
    assert len(c) == 1
    c.close()


def test_memory_backend_is_ephemeral():
    # A fresh MemoryBackend starts empty — nothing persists across instances.
    c1 = EmbeddingCache(backend="memory")
    c1.get_or_compute("x", embed)
    assert len(c1) == 1
    c1.close()

    c2 = EmbeddingCache(backend="memory")
    assert len(c2) == 0
    c2.close()


def test_memory_backend_info_and_lifetime():
    c = EmbeddingCache(backend="memory", model="text-embedding-3-large")
    c.get_or_compute("a", embed)
    c.get_or_compute("a", embed)  # hit
    info = c.info()
    assert info.backend == "MemoryBackend"
    assert info.entries == 1
    assert info.size_bytes is not None  # bytes held by stored vectors
    assert c.lifetime_stats().hits == 1
    c.close()


def test_memory_backend_instance_passthrough():
    backend = MemoryBackend()
    c = EmbeddingCache(backend=backend)
    c.get_or_compute("y", embed)
    assert backend.count() == 1
    c.close()
