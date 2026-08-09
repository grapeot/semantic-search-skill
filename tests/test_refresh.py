from types import SimpleNamespace

import numpy as np

import semantic_search_skill.cli as cli
from semantic_search_skill.index import CacheConfig, SemanticIndex


class FakeEmbedder:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def embed_batches_parallel(self, texts, batch_size, workers):
        return [[1.0, 0.0] for _ in texts], 1 if texts else 0


def test_refresh_index_saves_incremental_batches(tmp_path, monkeypatch) -> None:
    files = []
    for index in range(3):
        path = tmp_path / f"note{index}.md"
        path.write_text(f"# Note {index}\nhello", encoding="utf-8")
        files.append(str(path))
    monkeypatch.setattr(cli, "EmbeddingClient", FakeEmbedder)
    index = SemanticIndex(tmp_path / "cache", CacheConfig(provider="openai", model="text-embedding-3-small"))
    args = SimpleNamespace(batch_size=16, workers=1, base_url=None, file_batch_size=1)

    event = cli.refresh_index(index, files, args)

    assert event["files_scanned"] == 3
    assert event["files_updated"] == 3
    assert event["chunks_added"] == 3
    assert event["files_skipped_concurrent_updates"] == 0
    assert event["estimated_embedding_tokens"] > 0
    assert event["estimated_embedding_cost_usd"] >= 0
    chunks, embeddings = index.subset(files)
    assert len(chunks) == 3
    assert np.asarray(embeddings).shape == (3, 2)


def test_refresh_index_handles_unchanged_batches(tmp_path, monkeypatch) -> None:
    path = tmp_path / "note.md"
    path.write_text("# Note\nhello", encoding="utf-8")
    monkeypatch.setattr(cli, "EmbeddingClient", FakeEmbedder)
    index = SemanticIndex(tmp_path / "cache", CacheConfig(provider="openai", model="text-embedding-3-small"))
    args = SimpleNamespace(batch_size=16, workers=1, base_url=None, file_batch_size=1)

    _ = cli.refresh_index(index, [str(path)], args)
    event = cli.refresh_index(index, [str(path)], args)

    assert event["files_scanned"] == 1
    assert event["files_updated"] == 0
    assert event["files_skipped_concurrent_updates"] == 0
    assert event["estimated_embedding_tokens"] == 0
    assert event["estimated_embedding_cost_usd"] == 0.0


def test_relative_and_absolute_file_list_entries_share_cache_identity(tmp_path, monkeypatch) -> None:
    source = tmp_path / "note.md"
    source.write_text("# Note\nhello", encoding="utf-8")
    relative_list = tmp_path / "relative.txt"
    absolute_list = tmp_path / "absolute.txt"
    relative_list.write_text("note.md\n", encoding="utf-8")
    absolute_list.write_text(f"{source}\n", encoding="utf-8")
    monkeypatch.setattr(cli, "EmbeddingClient", FakeEmbedder)
    index = SemanticIndex(tmp_path / "cache", CacheConfig(provider="openai", model="text-embedding-3-small"))
    args = SimpleNamespace(batch_size=16, workers=1, base_url=None, file_batch_size=1)

    relative_paths = cli.read_file_list(str(relative_list), tmp_path)
    absolute_paths = cli.read_file_list(str(absolute_list), tmp_path)
    first = cli.refresh_index(index, relative_paths, args)
    second = cli.refresh_index(index, absolute_paths, args)

    assert relative_paths == absolute_paths == [str(source.resolve())]
    assert first["embedding_requests"] == 1
    assert second["files_updated"] == 0
    assert second["embedding_requests"] == 0
    assert index.stats()["file_count"] == 1


def test_refresh_index_publishes_source_deletion(tmp_path, monkeypatch) -> None:
    path = tmp_path / "note.md"
    path.write_text("# Note\nhello", encoding="utf-8")
    monkeypatch.setattr(cli, "EmbeddingClient", FakeEmbedder)
    index = SemanticIndex(tmp_path / "cache", CacheConfig(provider="openai", model="text-embedding-3-small"))
    args = SimpleNamespace(batch_size=16, workers=1, base_url=None, file_batch_size=1)
    _ = cli.refresh_index(index, [str(path)], args)
    path.unlink()

    event = cli.refresh_index(index, [str(path)], args)

    assert event["files_updated"] == 1
    assert event["chunks_added"] == 0
    assert index.search([str(path)], np.asarray([1.0, 0.0], dtype=np.float32), 1) == []


def test_refresh_index_shares_one_embedder_across_batches(tmp_path, monkeypatch) -> None:
    files = []
    for index in range(3):
        path = tmp_path / f"note{index}.md"
        path.write_text(f"# Note {index}\nhello", encoding="utf-8")
        files.append(str(path))
    monkeypatch.setattr(cli, "EmbeddingClient", FakeEmbedder)
    index = SemanticIndex(tmp_path / "cache", CacheConfig(provider="openai", model="text-embedding-3-small"))
    args = SimpleNamespace(batch_size=16, workers=1, base_url=None, file_batch_size=1)
    seen = []
    real_batch = cli.refresh_index_batch

    def spy(idx, file_paths, a, embedder=None):
        seen.append(embedder)
        return real_batch(idx, file_paths, a, embedder)

    monkeypatch.setattr(cli, "refresh_index_batch", spy)

    cli.refresh_index(index, files, args)

    assert len(seen) == 3
    assert all(e is seen[0] for e in seen), "embedder must be a single shared instance across all batches"
