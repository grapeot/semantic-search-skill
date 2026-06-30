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
    assert event["estimated_embedding_tokens"] > 0
    assert event["estimated_embedding_cost_usd"] >= 0
    assert index.embeddings is not None
    assert np.asarray(index.embeddings).shape == (3, 2)


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
    assert event["estimated_embedding_tokens"] == 0
    assert event["estimated_embedding_cost_usd"] == 0.0
