# Test Plan

## Default Offline Tests

- Chunking preserves source file, header, and line ranges.
- Cache write/read roundtrip works without an embedding provider by using fake vectors.
- v2 stores SQLite metadata and append-only normalized `float32` NPY segments.
- File-list subset reads load only matching chunk rows and referenced segment files.
- Incremental updates append new segments and do not rewrite old segment files.
- Cross-segment top-k matches monolithic cosine ranking.
- Query readers do not create or wait on the writer lock and can read a SQLite snapshot while a writer holds an uncommitted transaction.
- Segment publication and orphan cleanup share the writer-only lock, preventing cleanup from deleting an in-flight segment.
- `doctor` detects schema mismatch, segment row-count mismatch, orphan segment files, abandoned temp files, and inactive chunks.
- `doctor --cleanup-orphans` removes only unreferenced segment/temp files.
- Tiny v1 migration streams `chunks.jsonl` plus mmap `embeddings.npy`, preserves v1 files, and creates valid v2 metadata.
- Crash-before-publish migration leftovers are recoverable through rerun plus doctor orphan cleanup.
- Delete/shrink marks old chunks inactive without compaction.
- CLI argument parsing supports `query`, `rebuild`, `stats`, `doctor`, and `migrate-v1`.
- Counter writes no private document content.
- Counter events contain aggregate metrics only.

## Smoke Tests

Use a tiny file list with 3-5 local Markdown files and a temporary cache directory:

```bash
semantic-search rebuild --file-list tmp/files.txt --cache-dir tmp/cache
semantic-search query --file-list tmp/files.txt --cache-dir tmp/cache --query "example topic" --top-k 3
semantic-search query --file-list tmp/files.txt --cache-dir tmp/cache --query "example topic" --top-k 3 --no-refresh
semantic-search doctor --cache-dir tmp/cache
python scripts/benchmark_v2.py --files 50 --chunks-per-file 3 --dimension 32
```

## Live Integration Tests

Live tests are skipped by default and require a real embedding provider:

```bash
SEMANTIC_SEARCH_ENABLE_LIVE_TESTS=1 \
op run --env-file=.env -- pytest -m live_integration
```

## Privacy Review

Before public push or merge:

```bash
rg -n "op://""dev|/""Users/|grapeot""@|sk-[A-Za-z0-9]{20,}|OPENAI_""API_KEY=.*[A-Za-z0-9_-]{20,}" .
```

Public fake examples such as `op://your-vault/your-item/your-field` are allowed.
