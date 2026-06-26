"""Lifetime savings persistence: the v0.2 headline feature.

Session stats reset every process; lifetime stats are flushed into the cache and
accumulate across runs. These tests prove that the numbers survive a restart and
that flushing is delta-based (idempotent, no double-counting).
"""

import pytest

from embcache import EmbeddingCache
from embcache.backends import SQLiteBackend


class CountingEmbedder:
    def __init__(self):
        self.calls = 0

    def __call__(self, text: str):
        self.calls += 1
        return [float(len(text)), 0.5]


def test_lifetime_accumulates_across_instances(tmp_path):
    db = str(tmp_path / "emb.db")
    embed = CountingEmbedder()

    # Run 1: one miss, one hit.
    c1 = EmbeddingCache(backend="sqlite", path=db, model="m")
    c1.get_or_compute("hello", embed)
    c1.get_or_compute("hello", embed)
    c1.close()

    # Run 2: a fresh process-like instance over the same file. The cached
    # vector is on disk, so "hello" hits again.
    c2 = EmbeddingCache(backend="sqlite", path=db, model="m")
    c2.get_or_compute("hello", embed)
    life = c2.lifetime_stats()
    c2.close()

    assert life.hits == 2  # one hit per run
    assert life.misses == 1  # only the first run computed
    assert life.tokens_saved > 0


def test_session_stats_are_independent_of_lifetime(tmp_path):
    db = str(tmp_path / "emb.db")
    embed = CountingEmbedder()

    c1 = EmbeddingCache(backend="sqlite", path=db, model="m")
    c1.get_or_compute("x", embed)
    c1.get_or_compute("x", embed)
    c1.close()

    c2 = EmbeddingCache(backend="sqlite", path=db, model="m")
    c2.get_or_compute("x", embed)  # hit
    assert c2.stats.hits == 1  # this session only saw one lookup
    assert c2.lifetime_stats().hits == 2  # but lifetime knows about both
    c2.close()


def test_flush_is_idempotent(tmp_path):
    db = str(tmp_path / "emb.db")
    embed = CountingEmbedder()
    c = EmbeddingCache(backend="sqlite", path=db, model="m")
    c.get_or_compute("a", embed)
    c.get_or_compute("a", embed)  # hit

    c.flush()
    c.flush()  # second flush must not double-count
    c.flush()
    assert c.lifetime_stats().hits == 1
    c.close()
    # close() flushes again; still no double-count.
    assert EmbeddingCache(backend="sqlite", path=db, model="m").lifetime_stats().hits == 1


def test_report_scope_validation(tmp_path):
    c = EmbeddingCache(backend=SQLiteBackend(path=":memory:"), model="m")
    with pytest.raises(ValueError):
        c.report(scope="weekly")
    c.close()


def test_report_both_contains_session_and_lifetime():
    c = EmbeddingCache(backend=SQLiteBackend(path=":memory:"), model="m")
    embed = CountingEmbedder()
    c.get_or_compute("a", embed)
    c.get_or_compute("a", embed)
    out = c.report(scope="both")
    assert "[session]" in out
    assert "[lifetime]" in out
    c.close()


def test_concurrent_processes_accumulate_not_clobber(tmp_path):
    """Two instances sharing one file, interleaved, must sum their savings."""
    db = str(tmp_path / "emb.db")
    embed = CountingEmbedder()

    a = EmbeddingCache(backend="sqlite", path=db, model="m")
    b = EmbeddingCache(backend="sqlite", path=db, model="m")

    a.get_or_compute("shared", embed)  # miss, stores vector
    b.get_or_compute("shared", embed)  # hit from the shared file
    a.get_or_compute("shared", embed)  # hit

    a.flush()
    b.flush()

    c = EmbeddingCache(backend="sqlite", path=db, model="m")
    # 2 hits total (one from each instance), not lost to last-writer-wins.
    assert c.lifetime_stats().hits == 2
    a.close()
    b.close()
    c.close()
