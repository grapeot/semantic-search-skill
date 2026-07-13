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

Inspect cache health:

```bash
semantic-search doctor --cache-dir .knowledge_cache
semantic-search stats --cache-dir .knowledge_cache
```

## Cache Contract

The cache stores chunk metadata as JSONL and vectors as one `float32` NumPy matrix. The metadata records schema version, provider, model, and embedding dimension. If these do not match the current embedding configuration, rebuild explicitly instead of mixing caches.

Every CLI command acquires the cache lock before reading or refreshing the index. A process waiting behind another writer reports `Waiting to acquire cache lock` on stderr; do not treat that message as a hang or start a competing rebuild.

## Workspace Overlays

This public skill does not prescribe private source directories. A workspace overlay should define common file-list generation commands, the preferred cache directory, and any local cron or launcher integration.
