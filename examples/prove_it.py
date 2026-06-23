"""Prove it: embed the same 100 strings twice, watch the second run go all-hits.

Run with:  python examples/prove_it.py

No API key needed — this uses a fake embedder that sleeps briefly to simulate
network latency, so the savings report shows realistic numbers.
"""

from __future__ import annotations

import os
import tempfile
import time

from embcache import EmbeddingCache

MODEL = "text-embedding-3-small"


def fake_embed(text: str) -> list[float]:
    """Stand-in for a real embedding API call: a little latency, a toy vector."""
    time.sleep(0.01)  # ~10ms, pretend it's a network round-trip
    return [float(len(text)), float(sum(map(ord, text)) % 97), 0.5]


def main() -> None:
    texts = [f"sample sentence number {i} about cats and embeddings" for i in range(100)]

    db_path = os.path.join(tempfile.mkdtemp(), "prove_it.db")
    cache = EmbeddingCache(backend="sqlite", path=db_path, model=MODEL)
    embed = cache.wrap(fake_embed)

    print("First pass (cold cache — every call hits the 'API')...")
    t0 = time.perf_counter()
    for t in texts:
        embed(t)
    print(f"  took {time.perf_counter() - t0:.2f}s\n")

    print("Second pass (warm cache — every call should be a free hit)...")
    t0 = time.perf_counter()
    for t in texts:
        embed(t)
    print(f"  took {time.perf_counter() - t0:.2f}s\n")

    print(cache.report())
    cache.close()


if __name__ == "__main__":
    main()
