# embcache roadmap

Where embcache goes after v0.1. The north star is unchanged: **be the embedding
cache that proves what it saved.** Every item below is judged against that —
does it strengthen the "prove it" story, or remove a reason someone can't adopt
embcache in a real pipeline?

Current state (v0.1.0): exact-match caching, model-aware keys, SQLite + Redis
backends, packed-float32 storage, per-session savings tracking, 37 passing
tests, strict mypy, `py.typed`. Clean and complete — but session-scoped and
library-only.

---

## The two real gaps

1. **Savings are amnesiac.** `Stats` lives in memory and dies with the process.
   The headline feature — "see how much you saved" — resets every run. The most
   compelling number (savings *across* every CI run, every reindex, all month)
   is exactly the one embcache can't show today.
2. **It's library-only.** To inspect a shared `emb.db` — how big is it, how many
   entries, what has it saved cumulatively — you have to write Python. A cache
   people share across processes needs a way to look at it without importing it.

Everything else is reach, not gap.

---

## Milestones

### v0.2 — Persistent savings + CLI  ⭐ recommended first
**Thesis:** make the headline feature survive a restart, then make it inspectable
from the shell. This is the highest-leverage work because it deepens the one
thing that differentiates embcache rather than adding surface area.

- **Persist cumulative stats.** A small `meta` table (SQLite) / hash key (Redis)
  alongside the vectors: lifetime hits, misses, tokens saved, dollars saved,
  latency saved. `record_hit`/`record_miss` flush into it; `report()` can show
  session vs. lifetime.
- **Lifetime report.** `cache.report(scope="lifetime")` and a combined view.
- **`embcache` CLI** (`python -m embcache` + console-script entry point):
  - `embcache report <path>` — print the savings report for a cache file.
  - `embcache info <path>` — entry count, size on disk, model(s), created/last-write.
  - `embcache clear <path>` — wipe entries (with `--yes` guard).
- **Introspection API** the CLI builds on: `len(cache)`, `cache.info()`.

Risk: low. Backends gain one tiny table/namespace; no change to the hit/miss
path's correctness. Mostly additive.

### v0.3 — Broaden integration
**Thesis:** remove adoption blockers for real-world embedding code.

- **Async API.** `aget_or_compute` / `aget_or_compute_many` for async embed
  functions (OpenAI/httpx clients are async-first). Backend `get`/`set` stay
  sync (SQLite/Redis calls are fast); only the user's embed call is awaited.
- **Accurate token counts.** Optional `tiktoken` extra; `tokenizer="tiktoken"`
  swaps the ~4-chars/token heuristic for exact counts so dollar figures are
  real, not estimated. Heuristic stays the zero-dependency default.
- **In-memory backend.** `backend="memory"` — a dict store for tests and
  ephemeral runs, and a reference implementation of the `Backend` contract.

Risk: medium. Async doubles the core code path (keep it a thin async mirror, not
a fork). tiktoken is isolated behind the extra.

### v0.4 — Cache management
**Thesis:** turn a write-once store into a managed cache for long-lived,
bounded deployments.

- **TTL / expiration.** Per-entry `expires_at`; `evict_expired()`; optional
  lazy expiry on read.
- **Size bounds + LRU eviction.** `max_entries` / `max_bytes` with LRU eviction
  (needs an access-time/`last_used` column — a real schema change).
- **Namespaces.** Partition one backing store across projects/datasets.

Risk: higher. Eviction + TTL touch the schema and the read path; needs careful
migration handling for existing `emb.db` files. Deliberately last.

### Continuous — packaging & release
Not a milestone; runs alongside everything.

- **Ship to PyPI.** README says `pip install embcache`; make it true. Tag
  `v0.1.0`, build with hatchling, publish (Trusted Publishing via GitHub
  Actions).
- **CI.** GitHub Actions matrix (3.10–3.13) running `pytest` + `mypy --strict`,
  with `redis`/`fakeredis` installed so the 5 skipped backend tests actually run.
- **Coverage for the new surface.** Each milestone ships with tests; keep mypy
  strict clean.
- **Changelog + version bumps.** `CHANGELOG.md`, SemVer.

---

## Recommended order

1. **v0.2 (persistent savings + CLI)** — deepens the differentiator; low risk.
2. **CI + PyPI publish** — small, unblocks real adoption, makes the README honest.
3. **v0.3 (async + tiktoken + memory backend)** — broadens reach.
4. **v0.4 (TTL + eviction + namespaces)** — heaviest; do it when there's a user
   asking for it.

## Explicitly out of scope
Semantic / approximate matching (returns wrong vectors — a hard line in the
README), embedding *generation*, and becoming a vector database. embcache caches
exact text → exact vector and proves the savings. That's the whole product.
