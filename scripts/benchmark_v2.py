from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from semantic_search_skill.cli import rank
from semantic_search_skill.index import CacheConfig, SemanticIndex
from semantic_search_skill.models import Chunk


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline synthetic benchmark for the v2 SQLite+segment cache")
    parser.add_argument("--files", type=int, default=200)
    parser.add_argument("--chunks-per-file", type=int, default=5)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--segment-chunks", type=int, default=500)
    parser.add_argument("--queries", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rng = np.random.default_rng(args.seed)
    with tempfile.TemporaryDirectory(prefix="semantic-search-bench-") as tmp_name:
        root = Path(tmp_name)
        sources = []
        for index in range(args.files):
            path = root / "sources" / f"doc_{index:05d}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Synthetic\ncontent omitted from benchmark output\n", encoding="utf-8")
            sources.append(path)
        cache = root / "cache"
        index = SemanticIndex(cache, CacheConfig(provider="synthetic", model="offline", dimension=args.dimension))
        chunks: list[Chunk] = []
        vectors: list[np.ndarray] = []
        for file_index, source in enumerate(sources):
            for chunk_index in range(args.chunks_per_file):
                chunks.append(
                    Chunk(
                        id=f"doc_{file_index:05d}:{chunk_index}",
                        source_file=str(source),
                        text="synthetic text omitted",
                        header="# Synthetic",
                        position=(chunk_index + 1, chunk_index + 1),
                    )
                )
                vectors.append(rng.normal(size=args.dimension).astype(np.float32))
        build_started = time.perf_counter()
        files_per_segment = max(1, args.segment_chunks // max(1, args.chunks_per_file))
        chunks_per_segment = files_per_segment * args.chunks_per_file
        for start in range(0, len(chunks), chunks_per_segment):
            batch_chunks = chunks[start : start + chunks_per_segment]
            batch_vectors = np.asarray(vectors[start : start + chunks_per_segment], dtype=np.float32)
            files = sorted({chunk.source_file for chunk in batch_chunks})
            index.replace_files(batch_chunks, batch_vectors, files)
        build_seconds = time.perf_counter() - build_started
        all_files = [str(path) for path in sources]
        query_seconds = []
        subset_query_seconds = []
        exact_topk_match = True
        for _ in range(args.queries):
            query = rng.normal(size=args.dimension).astype(np.float32)
            started = time.perf_counter()
            results = index.search(all_files, query, args.top_k)
            query_seconds.append(time.perf_counter() - started)
            expected = rank(chunks, np.asarray(vectors, dtype=np.float32), query, args.top_k)
            exact_topk_match = exact_topk_match and [result.chunk.id for result in results] == [result.chunk.id for result in expected]
            subset_files = all_files[: max(1, min(10, len(all_files)))]
            started = time.perf_counter()
            _ = index.search(subset_files, query, args.top_k)
            subset_query_seconds.append(time.perf_counter() - started)
        stats = index.stats()
        output = {
            "schema_version": 2,
            "seed": args.seed,
            "files": args.files,
            "chunks": len(chunks),
            "dimension": args.dimension,
            "segments": stats["segment_count"],
            "top_k": args.top_k,
            "build_seconds": round(build_seconds, 6),
            "query_seconds_avg": round(float(np.mean(query_seconds)), 6),
            "subset_query_seconds_avg": round(float(np.mean(subset_query_seconds)), 6),
            "exact_topk_match": exact_topk_match,
        }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if exact_topk_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
