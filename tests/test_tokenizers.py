"""Pluggable token counting for the savings report."""

import pytest

from embcache import EmbeddingCache
from embcache.tokenizers import resolve_tokenizer


def embed(text):
    return [float(len(text)), 0.5]


def test_default_is_heuristic():
    c = EmbeddingCache(backend="memory")
    c.get_or_compute("the cat sat on the mat", embed)
    c.get_or_compute("the cat sat on the mat", embed)
    # ~4 chars/token over 22 chars ≈ 6 tokens credited to the hit.
    assert c.stats.tokens_saved == pytest.approx(6, abs=2)
    c.close()


def test_callable_tokenizer_is_used():
    c = EmbeddingCache(backend="memory", tokenizer=lambda s: 42)
    c.get_or_compute("anything", embed)
    c.get_or_compute("anything", embed)  # hit credits exactly 42 tokens
    assert c.stats.tokens_saved == 42
    c.close()


def test_unknown_tokenizer_name_raises():
    with pytest.raises(ValueError):
        EmbeddingCache(backend="memory", tokenizer="bpe9000")


def test_resolve_passes_through_callable():
    fn = lambda s: len(s)
    assert resolve_tokenizer(fn, "any-model") is fn


def test_tiktoken_tokenizer_matches_reference():
    tiktoken = pytest.importorskip("tiktoken")
    count = resolve_tokenizer("tiktoken", "text-embedding-3-small")
    enc = tiktoken.get_encoding("cl100k_base")
    text = "embeddings are fun to cache"
    assert count(text) == len(enc.encode(text))


def test_tiktoken_falls_back_for_unknown_model():
    pytest.importorskip("tiktoken")
    # An unrecognized model name must not blow up — it falls back to cl100k_base.
    count = resolve_tokenizer("tiktoken", "some-future-model-v9")
    assert count("hello world") > 0
