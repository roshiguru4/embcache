# embcache

**A drop-in caching layer for text embeddings. Stop paying to re-embed identical text.**

You re-embed the same text every test run, every reindex, every notebook re-execute — and you pay for all of it. embcache wraps your embedding function so identical text returns a cached vector instantly and for free. Then it tells you exactly how much it saved.

```python
from embcache import EmbeddingCache

cache = EmbeddingCache(model="text-embedding-3-small")

@cache.wrap
def embed(text: str) -> list[float]:
    return client.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding

embed("the cat sat")   # miss: calls the API, stores the vector
embed("the cat sat")   # hit: instant, free, no API call
```

That's the whole change. Five lines, and your duplicate embeddings stop costing money.

## Prove it

Other embedding caches exist. The difference here: **embcache makes the savings legible.** Every run tracks API calls avoided, tokens saved, dollars saved, and latency saved — and prints a report you can screenshot.

```python
print(cache.report())
```

```
embcache savings report  (model: text-embedding-3-small)
--------------------------------------------------------
Cache hits:        100 / 200 (50.0%)
API calls saved:   100
Tokens saved:      ~1,290
Estimated $ saved: $0.0000
Latency saved:     ~1.0s
```

That report is real output from [`examples/prove_it.py`](examples/prove_it.py), which embeds the same 100 strings twice. The first pass took **1.19s**; the warm second pass took **0.00s** — every call served from cache.

### Savings persist across runs

Per-process numbers reset when the process exits — but the real story is cumulative: every CI run, every reindex, all month. embcache flushes savings into the cache itself, so they add up across runs and across processes sharing one cache file.

```python
with EmbeddingCache(path="./emb.db", model="text-embedding-3-small") as cache:
    ...                              # flush happens on context exit (or cache.close())

print(cache.report(scope="lifetime"))   # cumulative, not just this run
print(cache.report(scope="both"))       # session + lifetime side by side
```

Cumulative stats use the backend's atomic increment, so two processes hammering the same shared cache accumulate correctly instead of clobbering each other.

### Before / after on a realistic workload

A test suite that embeds ~1,500 chunks per run, where 1,240 are repeats from previous runs (`text-embedding-3-small`, ~250 tokens each):

| Metric              | Without embcache | With embcache |
| ------------------- | ---------------- | ------------- |
| API calls           | 1,500            | **260**       |
| Tokens billed       | ~375,000         | **~65,000**   |
| Latency (re-embed)  | full             | **~148s saved** |
| $ per run           | $0.0075          | **$0.0013**   |

The dollar figures are small per run for the cheap models — but multiply by every CI run, every local iteration, every reindex, and by `text-embedding-3-large` (6.5× the price), and it adds up. More to the point: you can finally *see* it.

## Install

```bash
pip install embcache            # SQLite + memory backends, zero config
pip install embcache[redis]     # add the Redis backend
pip install embcache[tiktoken]  # exact token counts in the savings report
```

## How it works

1. Normalize the text (strip + collapse whitespace; optionally lowercase).
2. Build a **model-aware** key: `sha256(normalized_text + ":" + model)`.
3. Look it up in the backend.
   - **Hit** → deserialize the stored vector, record the savings, return it. No API call.
   - **Miss** → call your embed function, store the vector, record the cost, return it.

Vectors are stored as **packed float32 bytes**, not JSON — a flat 4 bytes per element instead of ~15 ASCII chars per float. The cache key includes the model name, so switching models never returns a stale vector from a different model.

## API

### Direct use

```python
cache = EmbeddingCache(backend="sqlite", path="./emb.db", model="text-embedding-3-small")
vec = cache.get_or_compute("the cat sat", embed_fn=my_embed_function)
```

### Decorator

```python
@cache.wrap
def embed(text: str) -> list[float]:
    ...
```

### Batch (only misses are computed, duplicates de-duped, order preserved)

```python
vecs = cache.get_or_compute_many(["text a", "text b"], embed_fn=my_batch_embed)
```

### Async

Embedding APIs are I/O-bound, so async is first-class. The cache lookup/store stay synchronous (they're fast); only your embed call — the network hop — is awaited, and on a hit it isn't awaited at all.

```python
@cache.awrap
async def embed(text: str) -> list[float]:
    resp = await aclient.embeddings.create(input=text, model="text-embedding-3-small")
    return resp.data[0].embedding

await embed("the cat sat")          # miss: awaits the API, stores the vector
await embed("the cat sat")          # hit: instant, free, nothing awaited

# or directly:
vec  = await cache.aget_or_compute("the cat sat", embed_fn=async_embed)
vecs = await cache.aget_or_compute_many(texts, embed_fn=async_batch_embed)

async with EmbeddingCache(path="./emb.db") as cache:
    ...                              # flushes savings on exit
```

Sync and async share one cache: a vector written by `get_or_compute` is an `aget_or_compute` hit, and vice versa.

### Token counting

The token and dollar figures default to a zero-dependency heuristic (~4 chars/token). For exact counts that match what the provider bills, opt into `tiktoken`:

```python
pip install embcache[tiktoken]

EmbeddingCache(model="text-embedding-3-small", tokenizer="tiktoken")   # exact
EmbeddingCache(tokenizer=lambda text: my_count(text))                  # or your own
```

### Backends

```python
EmbeddingCache(backend="sqlite", path="./emb.db")                       # default, local file
EmbeddingCache(backend="memory")                                        # ephemeral, in-process
EmbeddingCache(backend="redis", redis_url="redis://localhost:6379/0")   # shared / multi-process
```

All three implement the same three-method `Backend` contract (`get` / `set` / `has`) and pass the same test suite, so they're interchangeable. (Persistence and introspection layer on top via default methods, so a custom three-method backend still works.) The `memory` backend keeps nothing on disk — handy for tests and notebooks.

### Introspection

```python
len(cache)            # number of cached vectors
info = cache.info()   # entries, size on disk, models seen, created/last-write, lifetime savings
cache.clear()         # delete vectors; lifetime savings preserved (clear(reset_stats=True) to wipe both)
```

## CLI

Inspect a shared cache file without writing Python:

```bash
embcache report ./emb.db    # cumulative savings stored in the cache
embcache info   ./emb.db    # entries, size, models, timestamps
embcache clear  ./emb.db    # delete vectors (lifetime stats kept; --yes to skip prompt)
```

```
$ embcache info ./emb.db
embcache cache info
-------------------
Backend:      SQLiteBackend
Models:       text-embedding-3-large
Entries:      8,910
Size on disk: 54.2 MB
Created:      2026-05-01T12:30:00+00:00
Last write:   2026-06-25T23:30:12+00:00
```

Point at Redis with `--backend redis --redis-url redis://localhost:6379/0`. Also runnable as `python -m embcache ...`.

## Normalization

On by default: leading/trailing whitespace is stripped and internal whitespace runs collapse to a single space, so `"  the   cat sat "` and `"the cat sat"` hit the same entry. Lowercasing is **opt-in** (`EmbeddingCache(..., lowercase=True)`) because it can change meaning for some use cases.

## What embcache is — and isn't

**Is:** exact-match caching (same text → same vector), model-aware keys, SQLite + memory + Redis backends, sync **and** async APIs, optional exact token counting (tiktoken), cost/savings tracking that persists across runs, a CLI to inspect a cache, full type hints (`py.typed`).

**Isn't:**
- **No semantic matching.** "Similar" text never reuses a vector — that can return wrong results. Hard line.
- **No embedding generation.** You supply the embed function; embcache wraps it, it doesn't replace it.
- **Not a vector database.** This caches embeddings; it does not do retrieval or search.

## Development

```bash
pip install -e ".[dev]"
pytest          # 101 tests, SQLite + memory + Redis (via fakeredis)
mypy embcache   # clean, strict
```

## License

MIT
