# RFC: Semantic Search Skill Architecture

## Key Decisions

1. The standalone package is the canonical implementation. Workspace-local wrappers and old `tools/semantic_search` code are deprecated and may be deleted after migration.
2. The default public embedding provider is OpenAI with `text-embedding-3-small` because it is cheap, fast enough, and lower-dimensional than the old local 4096-dimensional model.
3. Credentials are resolved before Python starts. Users may set `OPENAI_API_KEY` directly or store an `op://...` reference in `.env` and run commands through `op run --env-file=.env -- ...`.
4. The default cache path is configurable by `SEMANTIC_SEARCH_CACHE_DIR`; a workspace overlay may set it to `.knowledge_cache` so global knowledge search stays fast.
5. The new cache format replaces pickle-based chunks with JSONL metadata plus a single `.npy` embedding matrix. Embeddings are not duplicated in chunk metadata.
6. Cache writes are atomic and versioned. Failed writes must not masquerade as valid cache state.
7. The implementation should expose an opt-in global counter for onboarding/debugging. When enabled, it appends JSONL events with timestamps, files scanned, files updated, chunks added, embedding requests, and rebuild reason.

## Cache Layout

```text
<cache-dir>/
├── cache.json
├── manifest.json
├── chunks.jsonl
├── embeddings.npy
├── index.lock
└── counter.jsonl        # optional, enabled by env or CLI
```

`cache.json` records:

- `schema_version`
- `embedding_provider`
- `embedding_model`
- `embedding_dimension`
- `created_at`
- `updated_at`
- `chunk_count`

`manifest.json` maps normalized source paths to:

- `mtime_ns`
- optional content hash when enabled
- `chunk_indices`
- `chunk_count`

`chunks.jsonl` stores one JSON object per chunk:

- `id`
- `source_file`
- `text`
- `header`
- `position`
- `metadata`

`embeddings.npy` stores one `float32` row per chunk in the same order as `chunks.jsonl`.

## Pickle Concern

The old cache used pickle because loading Python objects is convenient, but it made the cache fragile and implementation-coupled. JSONL metadata load time is acceptable because embeddings dominate disk size, while chunk records are much smaller once vectors are removed. The implementation can optimize later by adding a compact SQLite metadata store, but the first public contract should avoid pickle.

## Failure Behavior

- If cache metadata and embedding rows disagree, fail with a clear diagnostic.
- If cache schema/model/dimension does not match the requested embedding config, require `--rebuild` or a separate cache directory.
- If a write fails, leave the previous cache intact.
- Do not silently rebuild forever. Rebuild must be explicit or caused by a missing cache.

## Counter Feature

The counter is disabled by default for public users:

```dotenv
SEMANTIC_SEARCH_COUNTER_ENABLED=0
SEMANTIC_SEARCH_COUNTER_PATH=.knowledge_cache/counter.jsonl
```

When enabled, each indexing run appends one JSONL event. This makes onboarding and cron health checks observable without inspecting private document content.

## CLI Shape

Initial commands:

```bash
semantic-search query --file-list tmp/files.txt --query "..." --top-k 10
semantic-search rebuild --file-list tmp/files.txt --cache-dir .knowledge_cache
semantic-search doctor --cache-dir .knowledge_cache
semantic-search stats --cache-dir .knowledge_cache
```

Recommended high-throughput rebuild options:

```bash
semantic-search rebuild --file-list tmp/files.txt --workers 64 --batch-size 128
```

The implementation may tune concurrency to provider rate limits. OpenAI API calls should batch inputs and report failures transparently.
