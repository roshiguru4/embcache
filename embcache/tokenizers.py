"""Token counting strategies for the savings report.

The dollar and token figures are only as good as the token count behind them.
By default embcache uses a zero-dependency heuristic (~4 chars/token); for exact
counts that match what the provider actually bills, opt into ``tiktoken``.

A tokenizer is just a ``Callable[[str], int]``. You can pass one of the built-in
names (``"heuristic"`` / ``"tiktoken"``) or your own callable to
:class:`~embcache.cache.EmbeddingCache`.
"""

from __future__ import annotations

from collections.abc import Callable

from . import pricing

Tokenizer = Callable[[str], int]


def heuristic_tokenizer() -> Tokenizer:
    """The default ~4-chars/token estimate. No third-party dependency."""
    return pricing.estimate_tokens


def tiktoken_tokenizer(model: str) -> Tokenizer:
    """An exact tokenizer backed by ``tiktoken``, matched to ``model``.

    Falls back to the ``cl100k_base`` encoding (used by the
    ``text-embedding-3-*`` and ``ada-002`` models) when the model name isn't
    recognized by tiktoken.

    Raises:
        ImportError: If the ``tiktoken`` package is not installed.
    """
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "the 'tiktoken' tokenizer requires the tiktoken package. "
            "Install it with: pip install embcache[tiktoken]"
        ) from exc

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    def count(text: str) -> int:
        return len(encoding.encode(text))

    return count


def resolve_tokenizer(spec: str | Tokenizer, model: str) -> Tokenizer:
    """Turn a tokenizer spec into a concrete ``Callable[[str], int]``.

    Args:
        spec: ``"heuristic"`` (default), ``"tiktoken"``, or a callable that maps
            a string to a token count.
        model: Model name, used to pick the right tiktoken encoding.

    Raises:
        ValueError: If ``spec`` is an unrecognized string.
    """
    if callable(spec):
        return spec
    if spec == "heuristic":
        return heuristic_tokenizer()
    if spec == "tiktoken":
        return tiktoken_tokenizer(model)
    raise ValueError(
        f"unknown tokenizer {spec!r}; expected 'heuristic', 'tiktoken', or a callable"
    )
