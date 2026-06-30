import numpy as np

from semantic_search_skill.cli import build_parser, rank
from semantic_search_skill.models import Chunk


def test_query_parser_accepts_core_arguments() -> None:
    parser = build_parser()

    args = parser.parse_args([
        "query",
        "--file-list",
        "tmp/files.txt",
        "--query",
        "local-first AI notes",
        "--top-k",
        "5",
    ])

    assert args.command == "query"
    assert args.file_list == "tmp/files.txt"
    assert args.query == "local-first AI notes"
    assert args.top_k == 5


def test_rank_returns_highest_cosine_similarity() -> None:
    chunks = [
        Chunk(id="a:0", source_file="a", text="alpha"),
        Chunk(id="b:0", source_file="b", text="beta"),
    ]
    results = rank(chunks, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), np.asarray([0.0, 1.0], dtype=np.float32), 1)

    assert results[0].chunk.id == "b:0"
