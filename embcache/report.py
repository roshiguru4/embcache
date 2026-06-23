"""Formats :class:`~embcache.stats.Stats` into a readable, screenshot-able summary."""

from __future__ import annotations

from .stats import Stats


def _format_tokens(tokens: int) -> str:
    if tokens >= 1000:
        return f"~{tokens:,}"
    return str(tokens)


def _format_latency(ms: float) -> str:
    if ms >= 1000:
        return f"~{ms / 1000:.1f}s"
    return f"~{ms:.0f}ms"


def format_report(stats: Stats, *, model: str | None = None) -> str:
    """Render ``stats`` as a multi-line plain-text report.

    Args:
        stats: The accumulated session stats.
        model: Optional model name to include in the header.

    Returns:
        A formatted, human-readable summary string.
    """
    header = "embcache savings report"
    if model:
        header += f"  (model: {model})"

    rate_pct = stats.hit_rate * 100
    lines = [
        header,
        "-" * len(header),
        f"Cache hits:        {stats.hits:,} / {stats.total:,} ({rate_pct:.1f}%)",
        f"API calls saved:   {stats.hits:,}",
        f"Tokens saved:      {_format_tokens(stats.tokens_saved)}",
        f"Estimated $ saved: ${stats.dollars_saved:.4f}",
        f"Latency saved:     {_format_latency(stats.latency_saved_ms)}",
    ]
    return "\n".join(lines)
