# Working Log

## Changelog

### 2026-06-30

- Created initial public-ready scaffold and migration plan for promoting the workspace-local semantic search tool into a standalone skill.

## Lessons Learned

- The old workspace cache failed through a truncated pickle file. The new public cache contract must avoid pickle metadata and use atomic writes.
