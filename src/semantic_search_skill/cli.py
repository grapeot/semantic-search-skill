from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from .chunker import MarkdownChunker, read_text_file
from .counter import append_counter_event, counter_enabled, counter_path
from .embedding import DEFAULT_MODEL, DEFAULT_PROVIDER, EmbeddingClient, estimate_embedding_input_chars
from .index import CacheConfig, CacheError, FileState, SemanticIndex, canonical_source_path, file_state, migrate_v1_cache, normalize_query, normalize_vectors
from .models import SearchResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic search over local text files")
    parser.add_argument("--env-file", default=None, help="Optional .env file to load before reading configuration")
    parser.add_argument("--model", default=None, help="Embedding model; defaults to SEMANTIC_SEARCH_EMBEDDING_MODEL or text-embedding-3-small")
    parser.add_argument("--provider", default=None, help="Embedding provider name for cache metadata")
    parser.add_argument("--base-url", default=None, help="Optional OpenAI-compatible base URL")
    parser.add_argument("--fallback-base-url", default=None, help="Optional fallback OpenAI-compatible base URL")
    parser.add_argument("--fallback-model", default=None, help="Embedding model for the fallback endpoint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    query = subparsers.add_parser("query", help="Search an indexed file list")
    query.add_argument("--file-list", required=True)
    query.add_argument("--source-root", default=None, help="Root for relative paths in the file list; defaults to the current directory")
    query.add_argument("--query", required=True)
    query.add_argument("--top-k", type=int, default=10)
    query.add_argument("--cache-dir", default=None)
    query.add_argument("--workers", type=int, default=64)
    query.add_argument("--batch-size", type=int, default=128)
    query.add_argument("--file-batch-size", type=int, default=2000)
    query.add_argument("--no-refresh", action="store_true", help="Do not index changed files before querying")
    query.add_argument("--counter", action="store_true", default=None, help="Enable counter event for this run")

    rebuild = subparsers.add_parser("rebuild", help="Build or refresh cache for a file list")
    rebuild.add_argument("--file-list", required=True)
    rebuild.add_argument("--source-root", default=None, help="Root for relative paths in the file list; defaults to the current directory")
    rebuild.add_argument("--cache-dir", default=None)
    rebuild.add_argument("--workers", type=int, default=64)
    rebuild.add_argument("--batch-size", type=int, default=128)
    rebuild.add_argument("--file-batch-size", type=int, default=2000)
    rebuild.add_argument("--counter", action="store_true", default=None, help="Enable counter event for this run")

    doctor = subparsers.add_parser("doctor", help="Validate cache health")
    doctor.add_argument("--cache-dir", default=None)
    doctor.add_argument("--cleanup-orphans", action="store_true", help="Remove abandoned temp files and segment files not referenced by SQLite metadata")

    stats = subparsers.add_parser("stats", help="Print cache statistics")
    stats.add_argument("--cache-dir", default=None)

    migrate = subparsers.add_parser("migrate-v1", help="Explicitly migrate a v1 JSONL+NPY cache to the v2 SQLite+segments layout")
    migrate.add_argument("--v1-cache-dir", required=True)
    migrate.add_argument("--cache-dir", default=None, help="Target v2 cache directory; defaults like other commands")
    migrate.add_argument("--segment-size", type=int, default=50000)

    canonicalize = subparsers.add_parser("canonicalize-paths", help="Rewrite existing source identities to canonical absolute paths without recomputing embeddings")
    canonicalize.add_argument("--cache-dir", default=None)
    canonicalize.add_argument("--source-root", required=True, help="Root used to resolve legacy relative source paths")
    canonicalize.add_argument("--apply", action="store_true", help="Apply the migration; the default is a read-only dry run")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.env_file:
        load_dotenv(args.env_file)
    else:
        load_dotenv()
    fallback_config_from_args(args)
    try:
        if args.command == "rebuild":
            return run_rebuild(args)
        if args.command == "query":
            return run_query(args)
        if args.command == "doctor":
            return run_doctor(args)
        if args.command == "stats":
            return run_stats(args)
        if args.command == "migrate-v1":
            return run_migrate_v1(args)
        if args.command == "canonicalize-paths":
            return run_canonicalize_paths(args)
    except CacheError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    return 1


def cache_dir_from_args(args: argparse.Namespace) -> Path:
    value = getattr(args, "cache_dir", None) or os.environ.get("SEMANTIC_SEARCH_CACHE_DIR") or ".knowledge_cache"
    return Path(value).expanduser().resolve(strict=False)


def config_from_args(args: argparse.Namespace) -> CacheConfig:
    return CacheConfig(
        provider=args.provider or os.environ.get("SEMANTIC_SEARCH_EMBEDDING_PROVIDER") or DEFAULT_PROVIDER,
        model=args.model or os.environ.get("SEMANTIC_SEARCH_EMBEDDING_MODEL") or DEFAULT_MODEL,
    )


def fallback_config_from_args(args: argparse.Namespace) -> tuple[str | None, str | None]:
    url = getattr(args, "fallback_base_url", None) or os.environ.get("SEMANTIC_SEARCH_FALLBACK_BASE_URL")
    model = getattr(args, "fallback_model", None) or os.environ.get("SEMANTIC_SEARCH_FALLBACK_MODEL")
    if url and not model:
        raise SystemExit("SEMANTIC_SEARCH_FALLBACK_MODEL / --fallback-model required when fallback URL is set")
    return url, model


def build_embedder(args: argparse.Namespace, model: str) -> EmbeddingClient:
    fallback_base_url, fallback_model = fallback_config_from_args(args)
    return EmbeddingClient(
        model=model,
        base_url=args.base_url or os.environ.get("SEMANTIC_SEARCH_BASE_URL"),
        fallback_base_url=fallback_base_url,
        fallback_model=fallback_model,
    )


def read_file_list(path: str, source_root: str | Path | None = None) -> list[str]:
    file_list = Path(path).expanduser().resolve(strict=False)
    root = Path(source_root).expanduser().resolve(strict=False) if source_root is not None else Path.cwd()
    return list(dict.fromkeys(canonical_source_path(line.strip(), root) for line in file_list.read_text(encoding="utf-8").splitlines() if line.strip()))


def run_rebuild(args: argparse.Namespace) -> int:
    file_paths = read_file_list(args.file_list, args.source_root)
    cache_dir = cache_dir_from_args(args)
    config = config_from_args(args)
    index = SemanticIndex(cache_dir, config)
    event = refresh_index(index, file_paths, args)
    event["command"] = "rebuild"
    maybe_write_counter(args, cache_dir, event)
    print(json.dumps({"ok": True, **event, "cache_dir": str(cache_dir)}, ensure_ascii=False, indent=2))
    return 0


def run_query(args: argparse.Namespace) -> int:
    file_paths = read_file_list(args.file_list, args.source_root)
    cache_dir = cache_dir_from_args(args)
    config = config_from_args(args)
    index = SemanticIndex(cache_dir, config)
    embedder = build_embedder(args, config.model)
    if args.no_refresh:
        index.load(require_exists=True)
        event = {"files_scanned": len(file_paths), "files_updated": 0, "chunks_added": 0, "embedding_requests": 0}
    else:
        event = refresh_index(index, file_paths, args, embedder=embedder)
    query_vector = np.asarray(embedder.embed_batch([args.query])[0], dtype=np.float32)
    results = index.search(file_paths, query_vector, args.top_k)
    event = {"command": "query", **event, "query_embedding_requests": 1, "result_count": len(results)}
    maybe_write_counter(args, cache_dir, event)
    print(json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2))
    return 0


def run_doctor(args: argparse.Namespace) -> int:
    index = SemanticIndex(cache_dir_from_args(args), config_from_args(args))
    index.load(require_exists=True)
    report = index.doctor(cleanup_orphans=args.cleanup_orphans)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 2


def run_migrate_v1(args: argparse.Namespace) -> int:
    cache_dir = cache_dir_from_args(args)
    config = config_from_args(args)
    event = migrate_v1_cache(args.v1_cache_dir, cache_dir, config, segment_size=args.segment_size)
    print(json.dumps({"ok": True, **event}, ensure_ascii=False, indent=2))
    return 0


def run_canonicalize_paths(args: argparse.Namespace) -> int:
    index = SemanticIndex(cache_dir_from_args(args), config_from_args(args))
    event = index.canonicalize_paths(args.source_root, apply=args.apply)
    print(json.dumps({"ok": True, **event}, ensure_ascii=False, indent=2))
    return 0


def run_stats(args: argparse.Namespace) -> int:
    index = SemanticIndex(cache_dir_from_args(args), config_from_args(args))
    index.load(require_exists=True)
    stats = index.stats()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


def refresh_index(index: SemanticIndex, file_paths: list[str], args: argparse.Namespace, embedder: EmbeddingClient | None = None) -> dict[str, int]:
    if embedder is None:
        embedder = build_embedder(args, index.config.model)
    totals = {
        "files_scanned": len(file_paths),
        "files_updated": 0,
        "files_skipped_concurrent_updates": 0,
        "chunks_added": 0,
        "embedding_requests": 0,
        "embedding_input_chars": 0,
        "estimated_embedding_tokens": 0,
        "estimated_embedding_cost_usd": 0.0,
    }
    file_batch_size = max(1, int(getattr(args, "file_batch_size", 2000)))
    cache_dir = cache_dir_from_args(args)
    for start in range(0, len(file_paths), file_batch_size):
        batch = file_paths[start : start + file_batch_size]
        batch_started = time.monotonic()
        event = refresh_index_batch(index, batch, args, embedder)
        duration_s = round(time.monotonic() - batch_started, 3)
        event["duration_s"] = duration_s
        event["estimated_tokens_per_minute"] = estimate_rate_per_minute(event["estimated_embedding_tokens"], duration_s)
        totals["files_updated"] += event["files_updated"]
        totals["files_skipped_concurrent_updates"] += event.get("files_skipped_concurrent_updates", 0)
        totals["chunks_added"] += event["chunks_added"]
        totals["embedding_requests"] += event["embedding_requests"]
        totals["embedding_input_chars"] += event["embedding_input_chars"]
        totals["estimated_embedding_tokens"] += event["estimated_embedding_tokens"]
        totals["estimated_embedding_cost_usd"] += event["estimated_embedding_cost_usd"]
        progress_event = {
            "command": getattr(args, "command", "refresh"),
            "progress": True,
            "files_seen": min(start + len(batch), len(file_paths)),
            "files_total": len(file_paths),
            **event,
        }
        if counter_enabled(True if getattr(args, "counter", None) else None):
            append_counter_event(counter_path(cache_dir), progress_event)
        print(json.dumps(progress_event, ensure_ascii=False), file=sys.stderr)
    return totals


def refresh_index_batch(index: SemanticIndex, file_paths: list[str], args: argparse.Namespace, embedder: EmbeddingClient | None = None) -> dict[str, int]:
    changed = index.changed_files(file_paths)
    if not changed:
        return empty_refresh_event()
    chunker = MarkdownChunker()
    chunks = []
    file_states: dict[str, FileState] = {}
    skipped_concurrent_updates = 0
    for file_path in changed:
        state = file_state(file_path)
        file_states[file_path] = state
        if not state.exists:
            continue
        try:
            content = read_text_file(file_path)
        except FileNotFoundError:
            file_states[file_path] = FileState(exists=False)
            continue
        if file_state(file_path) != state:
            skipped_concurrent_updates += 1
            file_states.pop(file_path, None)
            continue
        chunks.extend(chunker.chunk(file_path, content))
    if chunks:
        if embedder is None:
            embedder = build_embedder(args, index.config.model)
        texts = [chunk.text for chunk in chunks]
        input_chars = estimate_embedding_input_chars(texts)
        vectors, requests = embedder.embed_batches_parallel(texts, args.batch_size, args.workers)
        embeddings = np.asarray(vectors, dtype=np.float32)
    else:
        requests = 0
        input_chars = 0
        embeddings = np.empty((0, index.config.dimension or 0), dtype=np.float32)
    stable_files = {path for path, state in file_states.items() if file_state(path) == state}
    skipped_concurrent_updates += len(file_states) - len(stable_files)
    publish_chunks = [chunk for chunk in chunks if chunk.source_file in stable_files]
    if len(publish_chunks) != len(chunks):
        keep = [idx for idx, chunk in enumerate(chunks) if chunk.source_file in stable_files]
        embeddings = embeddings[keep] if len(keep) else np.empty((0, embeddings.shape[1] if embeddings.ndim == 2 else index.config.dimension or 0), dtype=np.float32)
    publish_files = [path for path in changed if path in stable_files]
    publish_states = {path: file_states[path] for path in publish_files}
    estimated_tokens = estimate_tokens(input_chars)
    skipped_at_publish = index.replace_files(publish_chunks, embeddings, publish_files, file_states=publish_states)
    skipped_concurrent_updates += len(skipped_at_publish)
    published_files = len(publish_files) - len(skipped_at_publish)
    skipped_at_publish_set = set(skipped_at_publish)
    published_chunks = sum(1 for chunk in publish_chunks if chunk.source_file not in skipped_at_publish_set)
    return {
        "files_updated": published_files,
        "files_skipped_concurrent_updates": skipped_concurrent_updates,
        "chunks_added": published_chunks,
        "embedding_requests": requests,
        "embedding_input_chars": input_chars,
        "estimated_embedding_tokens": estimated_tokens,
        "estimated_embedding_cost_usd": estimate_embedding_cost_usd(estimated_tokens),
    }


def rank(chunks: list, embeddings: np.ndarray, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
    normalized = normalize_vectors(embeddings)
    normalized_query = normalize_query(query_vector)
    scores = normalized @ normalized_query
    indices = np.lexsort((np.arange(len(chunks)), -scores))[: min(top_k, len(chunks))]
    return [SearchResult(chunk=chunks[index], score=float(scores[index])) for index in indices]


def maybe_write_counter(args: argparse.Namespace, cache_dir: Path, event: dict) -> None:
    explicit = True if getattr(args, "counter", None) else None
    if counter_enabled(explicit):
        append_counter_event(counter_path(cache_dir), event)


def estimate_tokens(chars: int) -> int:
    return int((chars + 3) // 4)


def estimate_embedding_cost_usd(tokens: int) -> float:
    price = float(os.environ.get("SEMANTIC_SEARCH_EMBEDDING_USD_PER_1M_TOKENS", "0.02"))
    return round(tokens / 1_000_000 * price, 6)


def estimate_rate_per_minute(tokens: int, duration_s: float) -> int:
    if duration_s <= 0:
        return 0
    return int(tokens / duration_s * 60)


def empty_refresh_event() -> dict[str, int | float]:
    return {
        "files_updated": 0,
        "files_skipped_concurrent_updates": 0,
        "chunks_added": 0,
        "embedding_requests": 0,
        "embedding_input_chars": 0,
        "estimated_embedding_tokens": 0,
        "estimated_embedding_cost_usd": 0.0,
    }


if __name__ == "__main__":
    raise SystemExit(main())
