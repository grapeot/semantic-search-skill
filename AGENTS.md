# Semantic Search Skill

## Project Role

This repository provides a public-ready semantic search skill for AI agents. It indexes local text files, stores reusable embeddings in a stable on-disk cache, and returns JSON search results for natural-language queries.

It is not a general RAG framework, vector database server, crawler, or workspace-specific knowledge system. Workspace-specific source paths and private defaults belong in private overlays, not in this public repository.

## Project Structure

- `README.md`: public installation and usage guide.
- `docs/prd.md`: product scope and success criteria.
- `docs/rfc.md`: architecture, cache contract, migration, and privacy decisions.
- `docs/migration.md`: promotion plan from the old workspace-local tool.
- `docs/test.md`: unit, smoke, integration, and privacy-review strategy.
- `docs/working.md`: changelog and lessons learned.
- `skills/skill_semantic_search.md`: canonical agent skill document.
- `src/semantic_search_skill/`: reusable Python package.
- `scripts/`: stable wrappers for humans and agents.
- `tests/`: offline tests by default; live embedding tests are opt-in.

## Environment Rules

- Use the project virtual environment: `uv venv .venv`, then `source .venv/bin/activate`.
- Install dependencies with `uv pip install -e '.[dev]'`; do not use bare `pip install`.
- Never commit real API keys, private source paths, local cache contents, raw private documents, or private `.env` files.
- Public examples must use fake values such as `op://your-vault/your-item/your-field` and `replace-with-your-real-key`.
- Authentication supports direct `OPENAI_API_KEY` values and values resolved before Python starts. Prefer `op run --env-file=.env -- <command>` for private 1Password references. The Python code consumes environment variables after resolution; it must not call 1Password itself.

## Cache Safety

- Cache writes must be atomic: write temporary files, flush/fsync, then rename.
- Cache metadata and embeddings must not duplicate vectors in two files.
- The cache schema must record provider, model, dimension, and schema version to prevent accidental mixed-model reuse.
- If cache load fails, report a clear error and require rebuild or explicit repair. Do not silently rebuild in a loop unless the caller opted into rebuild behavior.

## Maintenance

- Update `docs/working.md` after meaningful design or implementation changes.
- Keep `docs/rfc.md`, `docs/test.md`, and `skills/skill_semantic_search.md` aligned with CLI contract changes.
- This directory is intended to become an independent git repository. Commit from this repository root, not from a parent workspace.
