"""Cache introspection: a snapshot of what a cache file holds.

:class:`CacheInfo` is the structured answer to "what's in this cache?" — entry
count, size on disk, which models have written to it, when it was created and
last written, and its lifetime savings. The CLI's ``embcache info`` renders it;
library users can read the fields directly via :meth:`EmbeddingCache.info`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .stats import Stats


@dataclass
class CacheInfo:
    """A point-in-time description of a cache's contents and savings."""

    backend: str
    model: str
    entries: int | None
    size_bytes: int | None
    lifetime: Stats
    models: list[str] = field(default_factory=list)
    created_at: str | None = None
    last_write_at: str | None = None


def format_size(num_bytes: int | None) -> str:
    """Render a byte count as a human-readable string (e.g. ``54.2 MB``)."""
    if num_bytes is None:
        return "unknown"
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"  # pragma: no cover - unreachable, loop returns first


def format_info(info: CacheInfo) -> str:
    """Render :class:`CacheInfo` as a multi-line plain-text block."""
    entries = "unknown" if info.entries is None else f"{info.entries:,}"
    models = ", ".join(info.models) if info.models else (info.model or "—")
    lines = [
        "embcache cache info",
        "-------------------",
        f"Backend:      {info.backend}",
        f"Models:       {models}",
        f"Entries:      {entries}",
        f"Size on disk: {format_size(info.size_bytes)}",
        f"Created:      {info.created_at or 'unknown'}",
        f"Last write:   {info.last_write_at or 'unknown'}",
    ]
    return "\n".join(lines)
