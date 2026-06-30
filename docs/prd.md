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
- Cache writes are atomic and do not leave partial `chunks` or `embeddings` files as valid cache state.
- Credentials work through direct environment variables or `op run --env-file=.env -- ...`.
- Public docs contain no private paths, real secrets, or private 1Password vault references.

## Non-Goals

- No vector database server.
- No web crawler.
- No workspace-specific default source path list.
- No backward compatibility with the old `tools/semantic_search` entrypoint.
- No automatic 1Password calls from Python code.
