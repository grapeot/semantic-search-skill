# Migration: Workspace Tool to Standalone Public Skill

## Intent

Promote the old workspace-local `tools/semantic_search` into a standalone public skill and migrate all current workspace usage to the new package. The old implementation does not need compatibility wrappers; if something breaks, fix the new path directly.

## Workspace Decisions

- Keep the global workspace cache path as `.knowledge_cache`, but rebuild it with the new cache schema.
- Before rebuild, rename the old cache directory instead of deleting it immediately.
- Use the local OpenAI-compatible Qwen endpoint for the workspace rebuild and scheduled refreshes. Paid OpenAI embedding should only be used when explicitly authorized.
- Put private workspace defaults in the overlay skill, not in the public repo.
- Enable the global counter during onboarding to observe rebuild behavior and prevent accidental rebuild loops.
- After tests pass, keep old cache backups temporarily and delete the old `tools/semantic_search` implementation.

## Dependency Inventory To Migrate

- `rules/skills/semantic_search.md`: become private overlay pointing to the standalone skill and defining workspace defaults.
- `rules/skills/INDEX.md`: update the semantic search entry.
- `rules/skills/opencode_sessions_archive.md`: update AI sessions semantic index instructions.
- `contexts/ai_sessions/scripts/sync_sessions.sh`: switch cron indexing to the new CLI and local Qwen config.
- `periodic_jobs/backup_utility/scripts/borg_backup_knowledge_working.sh`: keep excluding `.knowledge_cache`.
- `tools/semantic_search/`: delete after migration, not wrap.
- `.knowledge_cache/`: rename old directory, rebuild in place, verify, then remove old backup.

## GitHub Plan

1. Scaffold `adhoc_jobs/semantic_search_skill` as an independent public-ready repo.
2. Initialize git with branch `master`.
3. Create GitHub public repo.
4. Push the scaffold commit to `master`.
5. Add branch protection to `master`: no direct bypass for admins, no required review.
6. Implement package, tests, and docs on a feature branch.
7. Run tests and privacy review.
8. Open PR, merge, then pull/update local master.
9. Migrate the workspace overlay and cron usage in the parent workspace.

## Rebuild Procedure

1. Stop writers to `.knowledge_cache` if any are running.
2. Rename `.knowledge_cache` to a timestamped backup.
3. Create or reuse a private `.env` for the standalone skill if credentials are needed; workspace-specific endpoints and model names belong in the private overlay.
4. Run rebuild through a background process manager after a short timed validation run; choose `--base-url`, `--model`, `--workers`, and `--batch-size` from the private overlay's tested settings. Global CLI options such as `--base-url` and `--model` must be placed before the `rebuild` or `query` subcommand.
5. Run `semantic-search doctor` and representative queries.
6. Run the migrated AI sessions sync manually.
7. Check `counter.jsonl` for repeated full rebuilds or unexpectedly high API request counts.
8. Delete the old cache backup after the new cache is verified.

## Privacy Review

Before public push and before PR merge, scan tracked files for secrets or private paths:

```bash
rg -n "op://""dev|/""Users/|grapeot""@|OPENAI_""API_KEY=.*[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{20,}" .
```

Fake `op://your-vault/...` examples are allowed. Real vault paths are not.

## Current Status

- Scaffold: complete.
- GitHub repo: complete at `https://github.com/grapeot/semantic-search-skill`.
- Implementation: merged through the cache/counter migration PRs.
- Workspace migration: active clients migrated to the standalone CLI; old `tools/semantic_search` implementation deleted from the parent workspace.
- Cache rebuild: complete. `.knowledge_cache` now uses Qwen 4096-dimension embeddings and passed `doctor` plus smoke query validation.
