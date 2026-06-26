"""embcache: a drop-in caching layer for text embeddings.

Wrap any embedding function so identical text returns a cached vector instantly
and for free, instead of hitting the API again — and prove exactly how much you
saved with ``cache.report()``.
"""

from __future__ import annotations

from .cache import EmbeddingCache
from .info import CacheInfo
from .stats import Stats

__all__ = ["EmbeddingCache", "Stats", "CacheInfo"]
__version__ = "0.2.0"
