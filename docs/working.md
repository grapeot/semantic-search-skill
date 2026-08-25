# Working Log

## Changelog

### 2026-06-30

- Created initial public-ready scaffold and migration plan for promoting the workspace-local semantic search tool into a standalone skill.
- Implemented the first functional cache contract: JSONL chunk metadata, single `embeddings.npy`, atomic save, OpenAI embedding client, `query`/`rebuild`/`doctor`/`stats` commands, and opt-in counter events.
- Tightened embedding input clamping after live workspace rebuild exposed very long chunks that exceeded OpenAI per-input limits.
- Changed `.env.example` to avoid a fixed relative counter path; relative paths depend on the caller's cwd and can accidentally write counters outside the intended cache directory.

### 2026-07-01

- Completed a full workspace rebuild against the local Qwen OpenAI-compatible endpoint. The rebuilt `.knowledge_cache` contains 1,980,682 chunks with 4096-dimension embeddings.
- Added defensive ranking normalization for non-finite embedding values after a live smoke query found two Qwen responses that produced `NaN` scores.

### 2026-07-13

- Earlier v1 work serialized CLI cache access with a cross-platform file lock and transactional multi-file recovery.
- The v2 storage work supersedes that approach for query/refresh paths by relying on SQLite WAL and short writer transactions.
- Replaced v1 monolithic `chunks.jsonl` + `embeddings.npy` storage with v2 SQLite metadata and append-only normalized `float32` NPY segments.
- Removed query-time file locking. Query now uses SQLite WAL snapshot reads, loads only matching file-list rows, and merges exact top-k across referenced segments.
- Changed refresh so changed detection, chunking, and embedding happen outside the writer transaction; publishing is a short SQLite transaction that marks old chunks inactive and appends new segment row references.
- Added explicit `migrate-v1`, preserving v1 files and streaming `chunks.jsonl` with mmap `embeddings.npy`.
- Added `doctor` orphan reporting/cleanup and a synthetic offline `scripts/benchmark_v2.py` JSON benchmark.
- Kept readers entirely lock-free while adding a writer-only lock shared by segment publication, migration, and orphan cleanup to close the physical-file race.
- Removed query-time materialization of the global file manifest and deferred chunk text loading until after top-k vector scoring.
- Preserved v1 manifest mtimes during migration so relative source paths are not incorrectly treated as changed when migration runs from another working directory.
- Profiled a 2.62-million-chunk, 4096-dimension global cache. A representative 17,354-file no-change query fell from about 13m34s and 93.8 GB peak memory footprint on v1 to 45.68s and 557 MB on v2, while preserving the same leading results.

### 2026-07-22

- Canonicalized CLI file-list entries to absolute paths before changed detection, publication, and query, preventing relative and absolute spellings from creating duplicate cache identities.
- Added explicit `--source-root` handling and file-list deduplication so callers can make relative path resolution independent of their working directory.
- Added a dry-run-first `canonicalize-paths` migration for existing v2 caches. It merges aliases in one SQLite transaction, preserves immutable vector segments, and recomputes zero embeddings.

### 2026-08-09

- Added optional transport-level endpoint failover in `EmbeddingClient`. When the primary OpenAI-compatible endpoint fails a batch after exhausting retries (connection error, timeout, rate limit, 4xx, or 5xx), the client sticky-switches to an optional fallback endpoint+model for the remainder of the process. A `threading.Lock` guarantees the switch warning prints at most once even under concurrent workers.
- Failover is cache-invisible: `CacheConfig.model` and cache metadata always reflect the primary model id. Two endpoints serving the same architecture with slightly different serving stacks (e.g. cosine similarity ~0.99) can therefore share one cache directory.
- One `EmbeddingClient` instance is now shared across all file batches within a single `refresh_index` call and reused for the query embedding in `run_query`, so the sticky switch survives the whole CLI invocation rather than resetting per batch.
- New env vars `SEMANTIC_SEARCH_BASE_URL`, `SEMANTIC_SEARCH_FALLBACK_BASE_URL`, `SEMANTIC_SEARCH_FALLBACK_MODEL` and CLI flags `--fallback-base-url` / `--fallback-model`. The fallback constructor parameters are keyword-only, preserving the original positional signature for backward compatibility. The public `self.client` attribute is preserved (eager primary construction restored).
- Strengthened the fallback unit tests to assert which client and model id each embed call targets, and added a regression test that `refresh_index` shares one embedder across batches.

### 2026-08-25

- Documented the silent-empty-result failure mode for relative file-list entries: they resolve against the CLI's cwd (or `--source-root`) rather than the directory the list was generated in, so a list generated from a workspace root and queried from another directory matches no cached identities and returns `[]` with no error or warning. Clarified the canonical skill to generate absolute-path lists or pin `--source-root` to the generation directory.

## Lessons Learned

- The old workspace cache failed through a truncated pickle file. The new public cache contract must avoid pickle metadata and use atomic writes.
- Privacy scan examples can self-match if they contain literal private patterns; write them as shell-concatenated patterns so the command still works without polluting scan output.
- Live embedding providers can return pathological vectors even when the batch request succeeds. Ranking should sanitize `NaN`/`Inf` defensively so a few bad vectors cannot dominate search results.
- Segment files should be treated as immutable publish artifacts. Query correctness comes from SQLite active metadata, so orphan physical files are safe but should be reported and cleaned explicitly.
- Refresh must not publish chunks if a source file changes during read/embed; skipping that file for the current run is safer than indexing a stale snapshot over newer source content.
- Persisted file paths are database identities, not display strings. Normalize them at the CLI boundary and require an explicit root when repairing legacy relative identities.
- Endpoint failover belongs in the transport layer, not the cache layer. Two OpenAI-compatible servers running the same model architecture produce near-identical (but not bit-exact) embeddings; recording only the primary model id in cache metadata lets a fallback server reuse an existing cache without invalidation, at the cost of ~1% vector noise that does not meaningfully change retrieval ranking.
- A sticky failover flag is useless if a new client is constructed per work unit. Long rebuilds that split files into batches must share one embedding client across all batches, otherwise a hard-down primary burns its full retry budget again on every batch.
- A relative file list is only as correct as the directory it is resolved against. Because entries resolve against the CLI's cwd (or `--source-root`), a list generated from one directory and queried from another silently matches nothing and returns an empty result with no warning. Prefer absolute-path lists, or pin `--source-root` to the generation directory; an empty query result on a known-populated cache is the first sign of this mismatch.
