# Test Plan

## Default Offline Tests

- Chunking preserves source file, header, and line ranges.
- Cache write/read roundtrip works without an embedding provider by using fake vectors.
- Atomic save leaves previous valid cache intact when a simulated write fails.
- `doctor` detects missing files, schema mismatch, model mismatch, and row-count mismatch.
- CLI argument parsing supports `query`, `rebuild`, `stats`, and `doctor`.
- Counter writes no private document content.

## Smoke Tests

Use a tiny file list with 3-5 local Markdown files and a temporary cache directory:

```bash
semantic-search rebuild --file-list tmp/files.txt --cache-dir tmp/cache
semantic-search query --file-list tmp/files.txt --cache-dir tmp/cache --query "example topic" --top-k 3
semantic-search doctor --cache-dir tmp/cache
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
