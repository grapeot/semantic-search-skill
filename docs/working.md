# Working Log

## Changelog

### 2026-06-30

- Created initial public-ready scaffold and migration plan for promoting the workspace-local semantic search tool into a standalone skill.
- Implemented the first functional cache contract: JSONL chunk metadata, single `embeddings.npy`, atomic save, OpenAI embedding client, `query`/`rebuild`/`doctor`/`stats` commands, and opt-in counter events.
- Tightened embedding input clamping after live workspace rebuild exposed very long chunks that exceeded OpenAI per-input limits.
- Changed `.env.example` to avoid a fixed relative counter path; relative paths depend on the caller's cwd and can accidentally write counters outside the intended cache directory.

## Lessons Learned

- The old workspace cache failed through a truncated pickle file. The new public cache contract must avoid pickle metadata and use atomic writes.
- Privacy scan examples can self-match if they contain literal private patterns; write them as shell-concatenated patterns so the command still works without polluting scan output.
