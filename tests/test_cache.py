import pytest

from embcache import EmbeddingCache
from embcache.backends import SQLiteBackend


class CountingEmbedder:
    """A fake embed function that records how many times it was called."""

    def __init__(self):
        self.calls = 0
        self.texts_seen = []

    def __call__(self, text: str):
        self.calls += 1
        self.texts_seen.append(text)
        # Deterministic toy vector derived from the text.
        return [float(len(text)), float(sum(map(ord, text)) % 100), 0.5]


class CountingBatchEmbedder:
    def __init__(self):
        self.calls = 0
        self.batches_seen = []

    def __call__(self, texts):
        self.calls += 1
        self.batches_seen.append(list(texts))
        return [[float(len(t)), 1.0] for t in texts]


@pytest.fixture
def cache():
    c = EmbeddingCache(backend=SQLiteBackend(path=":memory:"), model="text-embedding-3-small")
    yield c
    c.close()


def test_miss_then_hit(cache):
    embed = CountingEmbedder()
    first = cache.get_or_compute("the cat sat", embed)
    second = cache.get_or_compute("the cat sat", embed)

    assert embed.calls == 1  # second call served from cache
    assert first == second
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1


def test_normalization_means_trivial_differences_hit(cache):
    embed = CountingEmbedder()
    cache.get_or_compute("the cat sat", embed)
    cache.get_or_compute("  the   cat  sat ", embed)
    assert embed.calls == 1
    assert cache.stats.hits == 1


def test_model_aware_keys_do_not_collide():
    backend = SQLiteBackend(path=":memory:")
    c1 = EmbeddingCache(backend=backend, model="model-a")
    c2 = EmbeddingCache(backend=backend, model="model-b")
    embed_a = CountingEmbedder()
    embed_b = CountingEmbedder()

    c1.get_or_compute("same text", embed_a)
    # Different model on the SAME shared backend must miss, not return stale.
    c2.get_or_compute("same text", embed_b)

    assert embed_a.calls == 1
    assert embed_b.calls == 1
    backend.close()


def test_wrap_decorator(cache):
    embed = CountingEmbedder()

    @cache.wrap
    def embed_text(text: str):
        return embed(text)

    v1 = embed_text("hello world")
    v2 = embed_text("hello world")
    assert embed.calls == 1
    assert v1 == v2


def test_wrap_preserves_function_metadata(cache):
    @cache.wrap
    def embed_text(text: str):
        """my docstring"""
        return [1.0]

    assert embed_text.__name__ == "embed_text"
    assert embed_text.__doc__ == "my docstring"


def test_batch_only_computes_misses(cache):
    embed = CountingBatchEmbedder()
    # Prime one entry.
    cache.get_or_compute_many(["a", "b"], embed)
    embed2 = CountingBatchEmbedder()
    out = cache.get_or_compute_many(["a", "b", "c"], embed2)

    assert len(out) == 3
    # Only "c" should be computed in the second batch.
    assert embed2.batches_seen == [["c"]]


def test_batch_dedupes_within_call(cache):
    embed = CountingBatchEmbedder()
    out = cache.get_or_compute_many(["x", "x", "y"], embed)
    assert len(out) == 3
    assert out[0] == out[1]
    # "x" computed once despite appearing twice.
    assert embed.batches_seen == [["x", "y"]]


def test_batch_preserves_order(cache):
    embed = CountingBatchEmbedder()
    out = cache.get_or_compute_many(["aaa", "b", "cc"], embed)
    assert [v[0] for v in out] == [3.0, 1.0, 2.0]


def test_batch_validates_vector_count(cache):
    def bad_embed(texts):
        return [[1.0]]  # wrong count

    with pytest.raises(ValueError):
        cache.get_or_compute_many(["a", "b"], bad_embed)


def test_report_contains_real_numbers(cache):
    embed = CountingEmbedder()
    cache.get_or_compute("text one", embed)
    cache.get_or_compute("text one", embed)
    report = cache.report()
    assert "Cache hits:" in report
    assert "1 / 2" in report
    assert "API calls saved:   1" in report


def test_savings_recorded_on_hit(cache):
    embed = CountingEmbedder()
    cache.get_or_compute("some text to embed", embed)
    cache.get_or_compute("some text to embed", embed)
    assert cache.stats.tokens_saved > 0
    assert cache.stats.dollars_saved > 0
    assert cache.stats.hit_rate == 0.5


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        EmbeddingCache(backend="cassandra")


def test_persistence_across_instances(tmp_path):
    db = str(tmp_path / "emb.db")
    embed = CountingEmbedder()

    c1 = EmbeddingCache(backend="sqlite", path=db, model="m")
    c1.get_or_compute("persisted text", embed)
    c1.close()

    c2 = EmbeddingCache(backend="sqlite", path=db, model="m")
    c2.get_or_compute("persisted text", embed)
    c2.close()

    # Second instance hit the on-disk cache; embed only called once total.
    assert embed.calls == 1
