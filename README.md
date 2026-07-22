# Semantic Search Skill

Public-ready semantic search for AI agents. The CLI indexes local text files, stores embeddings in a reusable on-disk cache, and returns stable JSON results for natural-language queries.

The current cache format is v2: one logical global index per cache directory, backed by a single SQLite metadata database plus append-only normalized `float32` NumPy segment files. Segments are physical shards of the same index, not separate per-corpus indexes.

## Install

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e '.[dev]'
```

Configure credentials in your environment, or use a private `.env` resolved by 1Password:

```bash
cp .env.example .env
op run --env-file=.env -- semantic-search --help
```

## Basic Usage

```bash
semantic-search query \
  --file-list tmp/files.txt \
  --query "design tradeoffs in local-first AI tools" \
  --top-k 10 \
  --cache-dir .knowledge_cache
```

The default embedding model is `text-embedding-3-small`. The default cache directory can be set with `SEMANTIC_SEARCH_CACHE_DIR`.

File-list entries are canonicalized to absolute paths before cache access. Use `--source-root /path/to/workspace` with `query` or `rebuild` when relative entries must resolve independently of the caller's current directory.

`query` refreshes changed files by default. Use `--no-refresh` to perform a pure read against the existing cache. Query reads never acquire a file lock; SQLite WAL provides snapshot isolation. Segment publication, migration, and orphan cleanup share a writer-only lock so physical files cannot race with maintenance.

## Cache Operations

```bash
semantic-search rebuild --file-list tmp/files.txt --cache-dir .knowledge_cache
semantic-search doctor --cache-dir .knowledge_cache
semantic-search stats --cache-dir .knowledge_cache
```

`doctor` reports missing or corrupt segment files, orphan segment files, abandoned temp files, inactive chunks, and whether compaction would reclaim space. It does not compact by default. Use `doctor --cleanup-orphans` only to remove segment files that are not referenced by SQLite metadata and abandoned temp files.

Updates and source deletion make old chunks inactive but do not physically erase their text or vectors. If secure deletion matters, rebuild or remove the cache; `--cleanup-orphans` is not a purge operation.

## Migrating v1 Caches

v1 `cache.json` + `manifest.json` + `chunks.jsonl` + `embeddings.npy` caches are real persisted data. They are not auto-deleted and are not automatically upgraded during query or rebuild.

Run an explicit streaming migration:

```bash
semantic-search migrate-v1 \
  --v1-cache-dir old-cache \
  --cache-dir .knowledge_cache \
  --segment-size 50000
```

The migration reads `chunks.jsonl` line by line and memory-maps `embeddings.npy`, then atomically publishes the v2 SQLite database after segment files are safely written. If a process crashes before publish, rerun `migrate-v1`; leftover unreferenced segment files are reported by `doctor` and can be removed with `doctor --cleanup-orphans`.

Repair relative or duplicate source identities in an existing v2 cache with a dry run followed by an explicit apply. This changes SQLite metadata only and recomputes no embeddings:

```bash
semantic-search canonicalize-paths --cache-dir .knowledge_cache --source-root /path/to/workspace
semantic-search canonicalize-paths --cache-dir .knowledge_cache --source-root /path/to/workspace --apply
```

## Offline Benchmark

```bash
python scripts/benchmark_v2.py --files 200 --chunks-per-file 5 --dimension 64
```

The benchmark is pure synthetic, does not call an embedding provider, and prints JSON aggregate metrics only. It does not output source text, query text, or absolute paths.

## Agent Skill

Read `skills/skill_semantic_search.md` for the stable agent-facing contract. Workspace-specific source paths should be documented in a private overlay skill, not in this public repository.
