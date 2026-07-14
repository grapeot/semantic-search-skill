# PRD: Semantic Search Skill

## Goal

Provide a standalone public skill that lets AI agents semantically search local text collections with a stable CLI, reusable embedding cache, transparent configuration, and safe failure behavior.

## Users

- AI agents that need semantic retrieval over local files.
- Humans who want a simple CLI for indexing and querying Markdown, text, CSV, or other UTF-8 text files.
- Workspace maintainers who want a reusable cache shared across recurring jobs and ad hoc searches.

## Success Criteria

- A fresh clone can install the package with `uv pip install -e '.[dev]'` and run offline tests.
- `semantic-search query --file-list ... --query ...` returns stable JSON results.
- Cache metadata records schema version, embedding provider, model, dimension, source file state, and chunk counts.
- The cache keeps one logical global index per cache directory while storing vectors in append-only physical segments.
- Query reads do not acquire a file lock and can run against a SQLite WAL snapshot while a writer prepares a refresh.
- Segment publication, migration, and orphan cleanup use a writer-only lock; readers never wait on it.
- Refresh performs changed detection, chunking, and embedding outside the writer transaction, then publishes metadata with a short SQLite transaction.
- Small updates append a new segment and mark old chunks inactive; default refresh does not compact or rewrite old segments.
- Cache writes are atomic: new segments are fsynced before SQLite metadata references them.
- v1 `chunks.jsonl` + `embeddings.npy` caches require explicit `migrate-v1`; migration must preserve v1 files and stream data instead of loading all chunks into memory.
- Credentials work through direct environment variables or `op run --env-file=.env -- ...`.
- Public docs contain no private paths, real secrets, or private 1Password vault references.
- Counter events contain aggregate counts only, never source paths, document text, or query text.

## Non-Goals

- No vector database server.
- No web crawler.
- No workspace-specific default source path list.
- No automatic deletion of v1 caches.
- No default compaction during rebuild/query refresh.
- No backward compatibility with the old `tools/semantic_search` entrypoint.
- No automatic 1Password calls from Python code.
