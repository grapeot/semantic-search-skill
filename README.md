# Semantic Search Skill

Public-ready semantic search for AI agents. The CLI indexes local text files, stores embeddings in a reusable on-disk cache, and returns stable JSON results for natural-language queries.

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

## Agent Skill

Read `skills/skill_semantic_search.md` for the stable agent-facing contract. Workspace-specific source paths should be documented in a private overlay skill, not in this public repository.
