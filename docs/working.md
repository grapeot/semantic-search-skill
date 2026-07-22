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

## Lessons Learned

- The old workspace cache failed through a truncated pickle file. The new public cache contract must avoid pickle metadata and use atomic writes.
- Privacy scan examples can self-match if they contain literal private patterns; write them as shell-concatenated patterns so the command still works without polluting scan output.
- Live embedding providers can return pathological vectors even when the batch request succeeds. Ranking should sanitize `NaN`/`Inf` defensively so a few bad vectors cannot dominate search results.
- Segment files should be treated as immutable publish artifacts. Query correctness comes from SQLite active metadata, so orphan physical files are safe but should be reported and cleaned explicitly.
- Refresh must not publish chunks if a source file changes during read/embed; skipping that file for the current run is safer than indexing a stale snapshot over newer source content.
- Persisted file paths are database identities, not display strings. Normalize them at the CLI boundary and require an explicit root when repairing legacy relative identities.
