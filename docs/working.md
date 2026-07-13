# Working Log

## Changelog

### 2026-06-30

- Created initial public-ready scaffold and migration plan for promoting the workspace-local semantic search tool into a standalone skill.
- Implemented the first functional cache contract: JSONL chunk metadata, single `embeddings.npy`, atomic save, OpenAI embedding client, `query`/`rebuild`/`doctor`/`stats` commands, and opt-in counter events.
- Tightened embedding input clamping after live workspace rebuild exposed very long chunks that exceeded OpenAI per-input limits.
- Changed `.env.example` to avoid a fixed relative counter path; relative paths depend on the caller's cwd and can accidentally write counters outside the intended cache directory.

### 2026-07-01

- Completed a full workspace rebuild against the local Qwen OpenAI-compatible endpoint. The rebuilt `.knowledge_cache` contains 1,980,682 chunks with 4096-dimension embeddings.
- Added defensive ranking normalization for non-finite embedding values after a live smoke query found two Qwen responses that produced `NaN` scores.

### 2026-07-13

- Serialized CLI cache access with a cross-platform file lock and explicit stderr wait/acquired messages.
- Added cleanup for interrupted atomic-write directories after a lock holder exits unexpectedly.
- Added transaction-marker recovery so interruption during the multi-file commit restores the previous complete cache.

## Lessons Learned

- The old workspace cache failed through a truncated pickle file. The new public cache contract must avoid pickle metadata and use atomic writes.
- Privacy scan examples can self-match if they contain literal private patterns; write them as shell-concatenated patterns so the command still works without polluting scan output.
- Live embedding providers can return pathological vectors even when the batch request succeeds. Ranking should sanitize `NaN`/`Inf` defensively so a few bad vectors cannot dominate search results.
