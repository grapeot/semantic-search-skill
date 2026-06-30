from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from .chunker import MarkdownChunker, read_text_file
from .counter import append_counter_event, counter_enabled, counter_path
from .embedding import DEFAULT_MODEL, DEFAULT_PROVIDER, EmbeddingClient
from .index import CacheConfig, CacheError, SemanticIndex
from .models import SearchResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic search over local text files")
    parser.add_argument("--env-file", default=None, help="Optional .env file to load before reading configuration")
    parser.add_argument("--model", default=None, help="Embedding model; defaults to SEMANTIC_SEARCH_EMBEDDING_MODEL or text-embedding-3-small")
    parser.add_argument("--provider", default=None, help="Embedding provider name for cache metadata")
    parser.add_argument("--base-url", default=None, help="Optional OpenAI-compatible base URL")
    subparsers = parser.add_subparsers(dest="command", required=True)

    query = subparsers.add_parser("query", help="Search an indexed file list")
    query.add_argument("--file-list", required=True)
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
    rebuild.add_argument("--cache-dir", default=None)
    rebuild.add_argument("--workers", type=int, default=64)
    rebuild.add_argument("--batch-size", type=int, default=128)
    rebuild.add_argument("--file-batch-size", type=int, default=2000)
    rebuild.add_argument("--counter", action="store_true", default=None, help="Enable counter event for this run")

    doctor = subparsers.add_parser("doctor", help="Validate cache health")
    doctor.add_argument("--cache-dir", default=None)

    stats = subparsers.add_parser("stats", help="Print cache statistics")
    stats.add_argument("--cache-dir", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.env_file:
        load_dotenv(args.env_file)
    else:
        load_dotenv()
    try:
        if args.command == "rebuild":
            return run_rebuild(args)
        if args.command == "query":
            return run_query(args)
        if args.command == "doctor":
            return run_doctor(args)
        if args.command == "stats":
            return run_stats(args)
    except CacheError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    return 1


def cache_dir_from_args(args: argparse.Namespace) -> Path:
    value = args.cache_dir or os.environ.get("SEMANTIC_SEARCH_CACHE_DIR") or ".knowledge_cache"
    return Path(value)


def config_from_args(args: argparse.Namespace) -> CacheConfig:
    return CacheConfig(
        provider=args.provider or os.environ.get("SEMANTIC_SEARCH_EMBEDDING_PROVIDER") or DEFAULT_PROVIDER,
        model=args.model or os.environ.get("SEMANTIC_SEARCH_EMBEDDING_MODEL") or DEFAULT_MODEL,
    )


def read_file_list(path: str) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def run_rebuild(args: argparse.Namespace) -> int:
    file_paths = read_file_list(args.file_list)
    cache_dir = cache_dir_from_args(args)
    config = config_from_args(args)
    index = SemanticIndex(cache_dir, config)
    index.load(require_exists=False)
    event = refresh_index(index, file_paths, args)
    event["command"] = "rebuild"
    maybe_write_counter(args, cache_dir, event)
    print(json.dumps({"ok": True, **event, "cache_dir": str(cache_dir)}, ensure_ascii=False, indent=2))
    return 0


def run_query(args: argparse.Namespace) -> int:
    file_paths = read_file_list(args.file_list)
    cache_dir = cache_dir_from_args(args)
    config = config_from_args(args)
    index = SemanticIndex(cache_dir, config)
    index.load(require_exists=False)
    event = {"files_scanned": len(file_paths), "files_updated": 0, "chunks_added": 0, "embedding_requests": 0}
    if not args.no_refresh:
        event = refresh_index(index, file_paths, args)
    else:
        index.load(require_exists=True)
    chunks, embeddings = index.subset(file_paths)
    if not chunks or embeddings.size == 0:
        print(json.dumps([], ensure_ascii=False))
        return 0
    embedder = EmbeddingClient(model=config.model, base_url=args.base_url)
    query_vector = np.asarray(embedder.embed_batch([args.query])[0], dtype=np.float32)
    results = rank(chunks, embeddings, query_vector, args.top_k)
    event = {"command": "query", **event, "query_embedding_requests": 1, "result_count": len(results)}
    maybe_write_counter(args, cache_dir, event)
    print(json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2))
    return 0


def run_doctor(args: argparse.Namespace) -> int:
    index = SemanticIndex(cache_dir_from_args(args), config_from_args(args))
    index.load(require_exists=True)
    print(json.dumps({"ok": True, "stats": index.stats()}, ensure_ascii=False, indent=2))
    return 0


def run_stats(args: argparse.Namespace) -> int:
    index = SemanticIndex(cache_dir_from_args(args), config_from_args(args))
    index.load(require_exists=True)
    print(json.dumps(index.stats(), ensure_ascii=False, indent=2))
    return 0


def refresh_index(index: SemanticIndex, file_paths: list[str], args: argparse.Namespace) -> dict[str, int]:
    totals = {"files_scanned": len(file_paths), "files_updated": 0, "chunks_added": 0, "embedding_requests": 0}
    file_batch_size = max(1, int(getattr(args, "file_batch_size", 2000)))
    for start in range(0, len(file_paths), file_batch_size):
        batch = file_paths[start : start + file_batch_size]
        event = refresh_index_batch(index, batch, args)
        totals["files_updated"] += event["files_updated"]
        totals["chunks_added"] += event["chunks_added"]
        totals["embedding_requests"] += event["embedding_requests"]
        print(
            json.dumps(
                {
                    "progress": True,
                    "files_seen": min(start + len(batch), len(file_paths)),
                    "files_total": len(file_paths),
                    **event,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
    return totals


def refresh_index_batch(index: SemanticIndex, file_paths: list[str], args: argparse.Namespace) -> dict[str, int]:
    changed = index.changed_files(file_paths)
    if not changed:
        return {"files_updated": 0, "chunks_added": 0, "embedding_requests": 0}
    chunker = MarkdownChunker()
    chunks = []
    for file_path in changed:
        chunks.extend(chunker.chunk(file_path, read_text_file(file_path)))
    if chunks:
        embedder = EmbeddingClient(model=index.config.model, base_url=args.base_url)
        vectors, requests = embedder.embed_batches_parallel([chunk.text for chunk in chunks], args.batch_size, args.workers)
        embeddings = np.asarray(vectors, dtype=np.float32)
    else:
        requests = 0
        embeddings = np.empty((0, index.config.dimension or 0), dtype=np.float32)
    index.replace_files(chunks, embeddings, changed)
    return {
        "files_updated": len(changed),
        "chunks_added": len(chunks),
        "embedding_requests": requests,
    }


def rank(chunks: list, embeddings: np.ndarray, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = embeddings / norms
    query_norm = np.linalg.norm(query_vector)
    normalized_query = query_vector / query_norm if query_norm else query_vector
    scores = normalized @ normalized_query
    indices = np.argsort(scores)[-min(top_k, len(chunks)) :][::-1]
    return [SearchResult(chunk=chunks[index], score=float(scores[index])) for index in indices]


def maybe_write_counter(args: argparse.Namespace, cache_dir: Path, event: dict) -> None:
    explicit = True if getattr(args, "counter", None) else None
    if counter_enabled(explicit):
        append_counter_event(counter_path(cache_dir), event)


if __name__ == "__main__":
    raise SystemExit(main())
