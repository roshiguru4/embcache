"""Session savings tracking.

This is the headline feature: every hit and miss is recorded so the cache can
*prove* what it saved. A hit means an API call we did not make, tokens we did not
pay for, and latency we did not wait on.

Latency saved is estimated from the average observed latency of real misses
during the session: each hit is credited the running average miss latency, since
that is roughly what the hit would have cost had it gone to the API.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Stats:
    """Mutable accumulator of cache savings for a session."""

    hits: int = 0
    misses: int = 0
    tokens_saved: int = 0
    dollars_saved: float = 0.0
    latency_saved_ms: float = 0.0

    # Internal running totals for the average-miss-latency estimate.
    _miss_latency_total_ms: float = 0.0

    def record_hit(self, *, tokens: int, dollars: float) -> None:
        """Record a cache hit that avoided ``tokens`` tokens / ``dollars`` cost."""
        self.hits += 1
        self.tokens_saved += tokens
        self.dollars_saved += dollars
        self.latency_saved_ms += self.avg_miss_latency_ms

    def record_miss(self, *, latency_ms: float) -> None:
        """Record a cache miss (a real API call) that took ``latency_ms``."""
        self.misses += 1
        self._miss_latency_total_ms += latency_ms

    @property
    def total(self) -> int:
        """Total lookups (hits + misses)."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups served from cache (0.0 if no lookups yet)."""
        return self.hits / self.total if self.total else 0.0

    @property
    def avg_miss_latency_ms(self) -> float:
        """Average latency of misses so far (0.0 before any miss is seen)."""
        return self._miss_latency_total_ms / self.misses if self.misses else 0.0

    def reset(self) -> None:
        """Zero all counters."""
        self.hits = 0
        self.misses = 0
        self.tokens_saved = 0
        self.dollars_saved = 0.0
        self.latency_saved_ms = 0.0
        self._miss_latency_total_ms = 0.0
