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
