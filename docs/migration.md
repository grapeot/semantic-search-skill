# Migration: v1 to v2 Cache

## Intent

Migrate existing v1 JSONL+NPY caches to v2 SQLite metadata plus append-only NPY segments without deleting persisted data.

## Deployment Decisions

- Keep one global cache path, but migrate or rebuild it with the v2 cache schema.
- v1 cache files are concrete persisted data. Do not delete or overwrite them automatically.
- Use `migrate-v1` when preserving current embeddings matters; use rebuild only when intentionally regenerating embeddings.
- Keep provider endpoints and private defaults in the host workspace, not in this public repository.
- Enable the global counter during onboarding to observe rebuild behavior and prevent accidental rebuild loops.
- After tests pass, keep old cache backups temporarily and delete the old `tools/semantic_search` implementation.

## Host Integration

Update private overlay instructions, scheduled refresh jobs, and backup exclusions to use the standalone CLI and the selected cache directory. Keep those workspace-specific paths and provider settings outside this public repository.

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

## v1 Migration Procedure

1. Stop scheduled writers if a deployment requires a fully quiet cutover. Query readers may continue on the old cache until the command publishes v2.
2. Choose the target v2 cache directory. It may be the same directory as the v1 files because v2 uses `index.sqlite3` and `segments/`, but keeping a temporary copy is safer for first rollout.
3. Run:

```bash
semantic-search migrate-v1 \
  --v1-cache-dir old-cache \
  --cache-dir .knowledge_cache \
  --segment-size 50000
```

4. Run `semantic-search doctor --cache-dir .knowledge_cache`.
5. Run representative `query --no-refresh` commands to validate retrieval without triggering new embedding writes.
6. If `doctor` reports orphan segment files from a crashed migration attempt, run `semantic-search doctor --cache-dir .knowledge_cache --cleanup-orphans` after verifying the active DB is healthy.
7. Keep v1 files until the v2 cache has passed smoke queries and scheduled refresh has run successfully.

Migration preserves v1 `manifest.json` mtimes even when source paths are relative to a different working directory. v1 did not store source file sizes, so migrated rows use an unknown-size sentinel until that source is next refreshed; mtime remains the change detector during that interval.

## Rebuild Procedure

Use rebuild when you intentionally want fresh embeddings instead of preserving v1 vectors:

1. Create or reuse a private `.env` if credentials are needed; workspace-specific endpoints and model names belong in private overlays.
2. Run a small timed validation first, then run the full rebuild with provider-specific `--workers` and `--batch-size` settings.
3. Run `semantic-search doctor` and representative queries.
4. Check `counter.jsonl` for repeated full rebuilds or unexpectedly high API request counts. Counter events should contain counts only.

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
- v2 global index performance work: implemented locally on the feature branch with SQLite metadata, append-only normalized segments, explicit v1 migration, and offline tests/benchmark.
