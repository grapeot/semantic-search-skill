import json

import numpy as np
import pytest

from semantic_search_skill.index import CacheConfig, CacheError, SemanticIndex
from semantic_search_skill.models import Chunk


def test_index_roundtrip_without_pickle(tmp_path) -> None:
    source = tmp_path / "note.md"
    source.write_text("# Note\nhello world", encoding="utf-8")
    cache = tmp_path / "cache"
    index = SemanticIndex(cache, CacheConfig(provider="openai", model="text-embedding-3-small"))
    chunks = [Chunk(id=f"{source}:0", source_file=str(source), text="hello world", header="# Note", position=(1, 2))]

    index.replace_files(chunks, np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), [str(source)])

    assert (cache / "chunks.jsonl").exists()
    assert not (cache / "chunks.pkl").exists()

    loaded = SemanticIndex(cache, CacheConfig(provider="openai", model="text-embedding-3-small"))
    loaded.load()

    assert loaded.chunks == chunks
    assert loaded.embeddings is not None
    assert loaded.embeddings.shape == (1, 3)
    assert loaded.stats()["meta"]["embedding_dimension"] == 3


def test_doctor_detects_embedding_row_mismatch(tmp_path) -> None:
    source = tmp_path / "note.md"
    source.write_text("hello", encoding="utf-8")
    cache = tmp_path / "cache"
    index = SemanticIndex(cache, CacheConfig(provider="openai", model="text-embedding-3-small"))
    index.replace_files(
        [Chunk(id=f"{source}:0", source_file=str(source), text="hello")],
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        [str(source)],
    )
    np.save(cache / "embeddings.npy", np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))

    broken = SemanticIndex(cache, CacheConfig(provider="openai", model="text-embedding-3-small"))
    with pytest.raises(CacheError, match="does not match"):
        broken.load()


def test_model_mismatch_requires_separate_cache_or_rebuild(tmp_path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "cache.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "embedding_provider": "openai",
                "embedding_model": "text-embedding-3-small",
                "embedding_dimension": 3,
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
                "chunk_count": 0,
            }
        ),
        encoding="utf-8",
    )

    index = SemanticIndex(cache, CacheConfig(provider="openai", model="text-embedding-3-large"))

    with pytest.raises(CacheError, match="model"):
        index.load()
