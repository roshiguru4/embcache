# Changelog

All notable changes to embcache are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.2.0]

Persistent savings + a CLI. Savings now survive the process and you can inspect
a cache without writing Python.

### Added
- **Persistent lifetime savings.** Cumulative hits, misses, tokens, dollars, and
  latency saved are flushed into the cache and accumulate across runs and across
  processes sharing one cache file. Backed by atomic increments (SQLite UPSERT,
  Redis `HINCRBYFLOAT`) so concurrent writers add up instead of clobbering.
- **`report(scope=...)`** — `"session"` (default, unchanged), `"lifetime"`, or
  `"both"`. Plus `EmbeddingCache.lifetime_stats()` and `flush()`.
- **`embcache` CLI** with `report`, `info`, and `clear` subcommands; also
  runnable as `python -m embcache`. Targets a SQLite file by default or Redis
  via `--backend redis --redis-url ...`.
- **Introspection.** `len(cache)`, `cache.info()` returning a `CacheInfo`
  (entries, size on disk, models seen, created/last-write timestamps, lifetime
  stats), and `cache.clear(reset_stats=False)`.
- Backend contract gains `count`, `clear`, `size_bytes`, `increment_stats`,
  `read_stats`, `reset_stats`, `get_meta`, and `set_meta` — all with default
  implementations over the core three methods, so custom backends keep working.
- `fakeredis` added to the dev extra; the Redis backend tests now run in CI
  instead of skipping.

### Changed
- `EmbeddingCache.close()` (and context-manager exit) now flushes pending
  savings before closing the backend.
- Bumped to 0.2.0.

## [0.1.0]

Initial release: exact-match embedding cache with model-aware keys, SQLite and
Redis backends, packed-float32 storage, and per-session savings tracking.
