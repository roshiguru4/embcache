"""Introspection (`info`, `len`, `clear`) and the `embcache` CLI."""

import pytest

from embcache import CacheInfo, EmbeddingCache
from embcache.backends import SQLiteBackend
from embcache.cli import main
from embcache.info import format_size


class CountingEmbedder:
    def __init__(self):
        self.calls = 0

    def __call__(self, text: str):
        self.calls += 1
        return [float(len(text)), 0.5]


# -- info() / len() / clear() ----------------------------------------------


def test_info_reports_entries_and_models(tmp_path):
    db = str(tmp_path / "emb.db")
    c = EmbeddingCache(backend="sqlite", path=db, model="text-embedding-3-large")
    embed = CountingEmbedder()
    c.get_or_compute("a", embed)
    c.get_or_compute("b", embed)

    info = c.info()
    assert isinstance(info, CacheInfo)
    assert info.entries == 2
    assert info.backend == "SQLiteBackend"
    assert info.models == ["text-embedding-3-large"]
    assert info.size_bytes is not None and info.size_bytes > 0
    assert info.created_at is not None
    c.close()


def test_info_tracks_multiple_models(tmp_path):
    db = str(tmp_path / "emb.db")
    embed = CountingEmbedder()
    c1 = EmbeddingCache(backend="sqlite", path=db, model="model-a")
    c1.get_or_compute("x", embed)
    c1.close()
    c2 = EmbeddingCache(backend="sqlite", path=db, model="model-b")
    c2.get_or_compute("y", embed)
    info = c2.info()
    c2.close()
    assert set(info.models) == {"model-a", "model-b"}


def test_len_counts_vectors(tmp_path):
    c = EmbeddingCache(backend="sqlite", path=str(tmp_path / "emb.db"), model="m")
    embed = CountingEmbedder()
    assert len(c) == 0
    c.get_or_compute("a", embed)
    c.get_or_compute("a", embed)  # hit, no new entry
    c.get_or_compute("b", embed)
    assert len(c) == 2
    c.close()


def test_clear_keeps_lifetime_stats_by_default(tmp_path):
    db = str(tmp_path / "emb.db")
    c = EmbeddingCache(backend="sqlite", path=db, model="m")
    embed = CountingEmbedder()
    c.get_or_compute("a", embed)
    c.get_or_compute("a", embed)  # hit -> lifetime records a saving
    removed = c.clear()
    assert removed == 1
    assert len(c) == 0
    assert c.lifetime_stats().hits == 1  # savings preserved
    c.close()


def test_clear_reset_stats_wipes_everything(tmp_path):
    db = str(tmp_path / "emb.db")
    c = EmbeddingCache(backend="sqlite", path=db, model="m")
    embed = CountingEmbedder()
    c.get_or_compute("a", embed)
    c.get_or_compute("a", embed)
    c.clear(reset_stats=True)
    assert c.lifetime_stats().hits == 0
    assert c.stats.hits == 0
    c.close()


def test_format_size():
    assert format_size(None) == "unknown"
    assert format_size(512) == "512 B"
    assert format_size(2048) == "2.0 KB"
    assert format_size(5 * 1024 * 1024) == "5.0 MB"


# -- CLI -------------------------------------------------------------------


def _seed(db, model="text-embedding-3-small"):
    embed = CountingEmbedder()
    c = EmbeddingCache(backend="sqlite", path=db, model=model)
    c.get_or_compute("alpha", embed)
    c.get_or_compute("alpha", embed)  # hit
    c.get_or_compute("beta", embed)
    c.close()


def test_cli_report(tmp_path, capsys):
    db = str(tmp_path / "emb.db")
    _seed(db)
    rc = main(["report", db])
    out = capsys.readouterr().out
    assert rc == 0
    assert "[lifetime]" in out
    assert "API calls saved:   1" in out


def test_cli_info(tmp_path, capsys):
    db = str(tmp_path / "emb.db")
    _seed(db)
    rc = main(["info", db])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Entries:      2" in out
    assert "SQLiteBackend" in out


def test_cli_clear_with_yes(tmp_path, capsys):
    db = str(tmp_path / "emb.db")
    _seed(db)
    rc = main(["clear", db, "--yes"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Removed 2 entries" in out
    # Vectors gone, but lifetime stats preserved.
    assert main(["report", db]) == 0
    assert "API calls saved:   1" in capsys.readouterr().out


def test_cli_clear_abort(tmp_path, capsys, monkeypatch):
    db = str(tmp_path / "emb.db")
    _seed(db)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    rc = main(["clear", db])
    assert rc == 1
    assert "Aborted" in capsys.readouterr().out
    # Nothing deleted.
    c = EmbeddingCache(backend="sqlite", path=db, model="m")
    assert len(c) == 2
    c.close()


def test_cli_missing_file_errors(tmp_path, capsys):
    rc = main(["report", str(tmp_path / "nope.db")])
    assert rc == 2
    assert "no cache found" in capsys.readouterr().err


def test_cli_requires_command():
    with pytest.raises(SystemExit):
        main([])
