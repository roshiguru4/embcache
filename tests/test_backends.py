"""Backend test suite, run against every backend implementation.

The same tests run against SQLite and Redis so the two are provably
interchangeable. Redis is skipped if neither a live server nor ``fakeredis`` is
available.
"""

import pytest

from embcache.backends import SQLiteBackend


def _make_redis_backend():
    """Return a RedisBackend or None if no Redis is reachable."""
    from embcache.backends import RedisBackend

    # Prefer fakeredis so the suite runs in CI without a server.
    try:
        import fakeredis

        return RedisBackend(client=fakeredis.FakeStrictRedis())
    except ImportError:
        pass

    try:
        import redis

        client = redis.Redis.from_url("redis://localhost:6379/0")
        client.ping()
        client.flushdb()
        return RedisBackend(client=client)
    except Exception:
        return None


@pytest.fixture(params=["sqlite", "redis"])
def backend(request):
    if request.param == "sqlite":
        b = SQLiteBackend(path=":memory:")
        yield b
        b.close()
    else:
        b = _make_redis_backend()
        if b is None:
            pytest.skip("no Redis server or fakeredis available")
        yield b
        b.close()


def test_get_missing_returns_none(backend):
    assert backend.get("nope") is None


def test_has_false_when_missing(backend):
    assert backend.has("nope") is False


def test_set_then_get_round_trips(backend):
    backend.set("k", b"\x01\x02\x03")
    assert backend.get("k") == b"\x01\x02\x03"


def test_has_true_after_set(backend):
    backend.set("k", b"data")
    assert backend.has("k") is True


def test_set_overwrites(backend):
    backend.set("k", b"first")
    backend.set("k", b"second")
    assert backend.get("k") == b"second"


# -- introspection + persisted stats (v0.2) --------------------------------


def test_count_reflects_entries(backend):
    assert backend.count() == 0
    backend.set("a", b"1")
    backend.set("b", b"2")
    assert backend.count() == 2


def test_clear_removes_entries_and_returns_count(backend):
    backend.set("a", b"1")
    backend.set("b", b"2")
    removed = backend.clear()
    assert removed == 2
    assert backend.count() == 0
    assert backend.get("a") is None


def test_stats_round_trip_and_accumulate(backend):
    assert backend.read_stats()["hits"] == 0.0
    backend.increment_stats({"hits": 3, "dollars_saved": 0.25})
    backend.increment_stats({"hits": 2, "dollars_saved": 0.10})
    stats = backend.read_stats()
    assert stats["hits"] == 5.0
    assert abs(stats["dollars_saved"] - 0.35) < 1e-9


def test_reset_stats_zeros_counters(backend):
    backend.increment_stats({"hits": 9})
    backend.reset_stats()
    assert backend.read_stats()["hits"] == 0.0


def test_stats_and_meta_do_not_count_as_entries(backend):
    backend.increment_stats({"hits": 1})
    backend.set_meta("created_at", "2026-01-01T00:00:00+00:00")
    backend.set("real", b"vec")
    # Only the real vector should be counted, not the stats/meta storage.
    assert backend.count() == 1


def test_clear_preserves_stats_and_meta(backend):
    backend.set("real", b"vec")
    backend.increment_stats({"hits": 4})
    backend.set_meta("created_at", "2026-01-01T00:00:00+00:00")
    backend.clear()
    assert backend.read_stats()["hits"] == 4.0
    assert backend.get_meta("created_at") == "2026-01-01T00:00:00+00:00"


def test_meta_round_trip(backend):
    assert backend.get_meta("missing") is None
    backend.set_meta("models", '["text-embedding-3-small"]')
    assert backend.get_meta("models") == '["text-embedding-3-small"]'
