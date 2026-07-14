---
name: semantic-search-skill
description: >-
  Indexes local text files with embeddings and searches them by natural-language query through the semantic-search CLI.
disable-model-invocation: true
---

# Semantic Search Skill

Use this skill when an AI agent needs semantic retrieval over local text files rather than exact keyword search.

## When To Use

- The user asks to find conceptually related prior writing or notes.
- Keyword search is likely to miss relevant documents because wording may differ.
- A workflow needs reusable local embeddings over Markdown, text, CSV, transcripts, or similar UTF-8 files.
- A workspace overlay has identified source directories that should be searched semantically.

## Prerequisites

- Install from this repository root with `uv pip install -e '.[dev]'`.
- Provide `OPENAI_API_KEY` directly or run through `op run --env-file=.env -- ...` if `.env` stores an `op://...` reference.
- Set `SEMANTIC_SEARCH_CACHE_DIR` or pass `--cache-dir` explicitly.

## Commands

Build or refresh an index:

```bash
semantic-search rebuild --file-list tmp/files.txt --cache-dir .knowledge_cache --workers 64
```

Query the indexed files:

```bash
semantic-search query \
  --file-list tmp/files.txt \
  --cache-dir .knowledge_cache \
  --query "natural-language question" \
  --top-k 10
```

Use `--no-refresh` when the task requires a pure read of the existing cache:

```bash
semantic-search query \
  --file-list tmp/files.txt \
  --cache-dir .knowledge_cache \
  --query "natural-language question" \
  --top-k 10 \
  --no-refresh
```

Inspect cache health:

```bash
semantic-search doctor --cache-dir .knowledge_cache
semantic-search stats --cache-dir .knowledge_cache
```

Explicitly migrate a v1 cache:

```bash
semantic-search migrate-v1 --v1-cache-dir old-cache --cache-dir .knowledge_cache
```

## Cache Contract

The v2 cache stores metadata in one SQLite database and vectors in append-only normalized `float32` NumPy segment files. Segments are physical shards of the same logical global index, not separate indexes.

Query reads do not acquire a file lock. SQLite WAL snapshot isolation lets queries run while a writer prepares a refresh. Refresh computes changed chunks and embeddings first, then uses a writer-only lock and a short SQLite transaction to publish the segment safely. Migration and orphan cleanup use the same writer lock.

Small updates append a new segment and mark old chunks inactive. Default query/rebuild refresh does not compact or rewrite old segments. Use `doctor` to inspect inactive chunks and orphan segment files; use `doctor --cleanup-orphans` only for unreferenced physical segment/temp files.

Inactive chunks remain present on disk even though queries ignore them. Source deletion is not secure cache erasure; rebuild or remove the cache when physical deletion is required.

If a cache still has v1 `cache.json` + `chunks.jsonl` + `embeddings.npy` files but no v2 `index.sqlite3`, run `migrate-v1` explicitly. The migration preserves v1 files and streams data instead of loading all chunks into memory.

Counter events are aggregate operational metrics only. They must not include source paths, raw document text, query text, or absolute cache paths.

## Workspace Overlays

This public skill does not prescribe private source directories. A workspace overlay should define common file-list generation commands, the preferred cache directory, and any local cron or launcher integration.
