"""Per-model pricing and token estimation for the savings report.

Prices are USD per 1,000,000 input tokens, matching how embedding providers
publish them. The table is a best-effort snapshot; callers can override or
extend it, and unknown models simply contribute $0 to the savings estimate
(hits and tokens are still counted).

Token counting uses a cheap heuristic (~4 characters per token) so the package
has no dependency on a tokenizer. It is an estimate, and the report labels it as
such. Pass real token counts to the cache if you have them.
"""

from __future__ import annotations

# USD per 1,000,000 tokens.
PRICING_PER_MILLION: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}

# Average characters per token; rough but model-agnostic.
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text`` (~4 chars/token, min 1 for non-empty)."""
    if not text:
        return 0
    return max(1, round(len(text) / _CHARS_PER_TOKEN))


def price_per_million(model: str) -> float:
    """Return USD per 1M tokens for ``model``, or 0.0 if unknown."""
    return PRICING_PER_MILLION.get(model, 0.0)


def cost_for_tokens(model: str, tokens: int) -> float:
    """Return the USD cost of embedding ``tokens`` tokens under ``model``."""
    return tokens / 1_000_000 * price_per_million(model)
