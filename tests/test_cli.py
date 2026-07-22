import numpy as np

import semantic_search_skill.cli as cli
from semantic_search_skill.cli import build_parser, estimate_rate_per_minute, maybe_write_counter, rank, read_file_list, run_query
from semantic_search_skill.index import CacheConfig, SemanticIndex
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


def test_parser_accepts_migrate_v1_command() -> None:
    parser = build_parser()

    args = parser.parse_args(["migrate-v1", "--v1-cache-dir", "old-cache", "--cache-dir", "new-cache", "--segment-size", "10"])

    assert args.command == "migrate-v1"
    assert args.v1_cache_dir == "old-cache"
    assert args.segment_size == 10
    assert not hasattr(args, "replace")


def test_parser_accepts_canonicalize_paths_dry_run_and_apply() -> None:
    parser = build_parser()

    dry_run = parser.parse_args(["canonicalize-paths", "--cache-dir", "cache", "--source-root", "workspace"])
    apply = parser.parse_args(["canonicalize-paths", "--cache-dir", "cache", "--source-root", "workspace", "--apply"])

    assert dry_run.command == "canonicalize-paths"
    assert dry_run.apply is False
    assert apply.apply is True


def test_read_file_list_canonicalizes_and_deduplicates_paths(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "note.md"
    source.write_text("hello", encoding="utf-8")
    file_list = tmp_path / "files.txt"
    file_list.write_text(f"note.md\n{source}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    paths = read_file_list(str(file_list), workspace)

    assert paths == [str(source.resolve())]


def test_rank_returns_highest_cosine_similarity() -> None:
    chunks = [
        Chunk(id="a:0", source_file="a", text="alpha"),
        Chunk(id="b:0", source_file="b", text="beta"),
    ]
    results = rank(chunks, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), np.asarray([0.0, 1.0], dtype=np.float32), 1)

    assert results[0].chunk.id == "b:0"


def test_rank_ignores_nonfinite_embedding_values() -> None:
    chunks = [
        Chunk(id="bad:0", source_file="bad", text="bad"),
        Chunk(id="good:0", source_file="good", text="good"),
    ]

    results = rank(
        chunks,
        np.asarray([[np.nan, np.nan], [0.0, 1.0]], dtype=np.float32),
        np.asarray([0.0, 1.0], dtype=np.float32),
        2,
    )

    assert results[0].chunk.id == "good:0"
    assert all(np.isfinite(result.score) for result in results)


def test_rank_uses_input_order_for_equal_scores() -> None:
    chunks = [Chunk(id=str(index), source_file="same", text=str(index)) for index in range(3)]
    embeddings = np.asarray([[1.0, 0.0]] * 3, dtype=np.float32)

    results = rank(chunks, embeddings, np.asarray([1.0, 0.0], dtype=np.float32), 2)

    assert [result.chunk.id for result in results] == ["0", "1"]


def test_estimate_rate_per_minute() -> None:
    assert estimate_rate_per_minute(120, 60) == 120
    assert estimate_rate_per_minute(120, 0) == 0


def test_query_reader_does_not_create_or_use_filelock(tmp_path, monkeypatch, capsys) -> None:
    source = tmp_path / "note.md"
    source.write_text("hello", encoding="utf-8")
    file_list = tmp_path / "files.txt"
    file_list.write_text(str(source), encoding="utf-8")
    cache = tmp_path / "cache"
    index = SemanticIndex(cache, CacheConfig(provider="openai", model="text-embedding-3-small"))
    index.replace_files([Chunk(id=f"{source}:0", source_file=str(source), text="hello")], np.asarray([[1.0, 0.0]], dtype=np.float32), [str(source)])

    class FakeEmbedder:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def embed_batch(self, texts):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(cli, "EmbeddingClient", FakeEmbedder)

    status = run_query(
        build_parser().parse_args([
            "query",
            "--file-list",
            str(file_list),
            "--query",
            "hello",
            "--cache-dir",
            str(cache),
            "--no-refresh",
        ])
    )
    stderr = capsys.readouterr().err

    assert status == 0
    assert "cache lock" not in stderr
    assert not (cache / "index.lock").exists()


def test_counter_event_contains_only_aggregate_fields(tmp_path, monkeypatch) -> None:
    private_path = str(tmp_path / "private" / "note.md")
    cache = tmp_path / "cache"
    event = {"command": "query", "files_scanned": 1, "files_updated": 0, "chunks_added": 0, "result_count": 1}

    monkeypatch.setenv("SEMANTIC_SEARCH_COUNTER_ENABLED", "1")
    maybe_write_counter(type("Args", (), {"counter": False})(), cache, event)

    content = (cache / "counter.jsonl").read_text(encoding="utf-8")

    assert private_path not in content
    assert "files_scanned" in content
    assert "query" in content
