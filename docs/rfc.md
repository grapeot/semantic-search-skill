# RFC: Semantic Search Skill Architecture

## Key Decisions

1. The standalone package is the canonical implementation. Workspace-local wrappers and old `tools/semantic_search` code are deprecated.
2. The public default embedding provider is OpenAI with `text-embedding-3-small`; callers may select any OpenAI-compatible provider through CLI flags or environment variables.
3. Credentials are resolved before Python starts. Python reads environment variables only and does not call 1Password.
4. A cache directory contains one logical global index. Segment files are physical shards of that single index, not per-corpus or per-file indexes.
5. v2 stores metadata in SQLite and vectors in append-only normalized `float32` NumPy segment files.
6. Query uses SQLite WAL snapshot reads and does not acquire a file lock. Segment publication, migration, and orphan cleanup share a writer-only lock; embedding computation happens before that lock is acquired.
7. Refresh does changed detection, file reading, chunking, and embedding before entering a writer transaction. The short transaction only flips active metadata and inserts new segment row references.
8. Small updates do not rewrite old segments. Old chunks are marked inactive and stop participating in query. Default refresh does not compact.
9. v1 `cache.json` + `manifest.json` + `chunks.jsonl` + `embeddings.npy` caches are explicit persisted data. They require `migrate-v1`; query/rebuild do not auto-upgrade or delete them.
10. The opt-in counter writes aggregate operational metrics only.

## Cache Layout

```text
<cache-dir>/
├── index.sqlite3
├── index.sqlite3-wal       # SQLite managed
├── index.sqlite3-shm       # SQLite managed
├── segments/
│   ├── segment-*.npy
│   └── segment-migrated-*.npy
├── tmp/
└── counter.jsonl           # optional
```

The cache directory and subdirectories are created with private directory permissions. SQLite DB, WAL sidecars, segment files, and temp files use private file permissions where the platform permits it.

## SQLite Schema

`meta` records schema and embedding identity:

- `schema_version`
- `embedding_provider`
- `embedding_model`
- `embedding_dimension`
- `created_at`
- `updated_at`

`files` records source file state:

- source path
- `mtime_ns`
- `size_bytes`
- active flag
- active chunk count

`segments` records immutable physical vector files:

- relative segment path
- row count
- dimension
- state
- size bytes
- created timestamp

`chunks` records chunk metadata and row references:

- chunk id
- source file
- text, header, line range, metadata JSON
- segment id and segment row
- active flag

SQLite indexes cover `(source_file, active)` and `(segment_id, active, segment_row)` so a query with a file list loads only matching chunk rows and only the segment files those rows reference.

## Query Path

1. If `--no-refresh` is absent, run refresh first.
2. Embed the query text once.
3. Read active chunks for the requested file list from SQLite.
4. Group matching rows by segment.
5. Memory-map each referenced segment, score matching rows in batches, and maintain an exact global top-k heap across segments.

The file list may span multiple segments. Top-k is exact for the supplied file list because scores from all matching rows enter the same merge.

## Refresh Path

1. Read current SQLite file metadata using a snapshot.
2. Compare requested source files by `mtime_ns` and size; deleted files count as changed if they were previously active.
3. Read, chunk, and embed changed existing files outside the writer transaction.
4. Re-stat files before publish. Files that changed during refresh are skipped for that run so stale chunks are not published over newer source content.
5. Acquire the writer-only lock, write the new normalized vector segment to a temp file, fsync it, and rename it into `segments/`.
6. Enter `BEGIN IMMEDIATE`, insert segment metadata, mark old chunks for updated files inactive, insert new chunk row references, update file rows, then commit and release the writer lock.

If a segment file is written but SQLite metadata is not committed, it is an orphan. Query never sees it because only SQLite active rows drive segment loading. `doctor` reports such files, and `doctor --cleanup-orphans` removes them.

## v1 Migration

`migrate-v1` reads v1 `chunks.jsonl` line by line and memory-maps v1 `embeddings.npy`. It writes v2 segment files first, builds a temporary SQLite database, then atomically publishes `index.sqlite3`. It does not delete or rewrite v1 files.

Source mtimes come from v1 `manifest.json`, not from resolving potentially relative source paths against the migration process's working directory. Because v1 did not persist source sizes, migrated file rows use an unknown-size sentinel until their next refresh.

Crash behavior:

- Before DB publish: rerun `migrate-v1`; any unreferenced segment files are reported as orphans.
- After DB publish: run `doctor`; referenced segments remain valid.
- Existing v2 DB: migration fails unless `--replace` is supplied explicitly.

## Doctor And Cleanup

`doctor` validates:

- schema/provider/model/dimension compatibility
- segment file existence
- segment dtype and shape
- chunk row references within segment bounds
- orphan segment files not referenced by SQLite
- abandoned temp files
- inactive chunk count and whether compaction would reclaim space

`doctor --cleanup-orphans` acquires the same writer-only lock and rechecks SQLite references before removing unreferenced physical segment files and abandoned temp files. It does not compact segments that still contain inactive chunks. A future explicit compaction command may rewrite active rows into new segments, but default update paths must not do this.

Inactive chunks still retain their previous text and vectors on disk. Deleting or rewriting a source file removes old chunks from search results, but it is not secure erasure. Users who need physical deletion must rebuild the cache or remove the cache directory until an explicit compaction/purge command exists.

## Counter Feature

The counter is disabled by default:

```dotenv
SEMANTIC_SEARCH_COUNTER_ENABLED=0
```

When enabled, events include aggregate fields such as files scanned, files updated, skipped concurrent updates, chunks added, embedding requests, estimated tokens, estimated cost, duration, and result count. Events must not include file paths, raw text, query text, or absolute cache paths.

## CLI Shape

```bash
semantic-search query --file-list tmp/files.txt --query "..." --top-k 10
semantic-search query --file-list tmp/files.txt --query "..." --no-refresh
semantic-search rebuild --file-list tmp/files.txt --cache-dir .knowledge_cache
semantic-search doctor --cache-dir .knowledge_cache
semantic-search doctor --cache-dir .knowledge_cache --cleanup-orphans
semantic-search stats --cache-dir .knowledge_cache
semantic-search migrate-v1 --v1-cache-dir old-cache --cache-dir .knowledge_cache
```
