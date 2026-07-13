import io
import json
import os
from threading import Event, Thread

import numpy as np
import pytest

from semantic_search_skill.index import (
    BACKUP_TMP_PREFIX,
    WRITE_TMP_PREFIX,
    CacheConfig,
    CacheError,
    SemanticIndex,
)
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


def test_lock_reports_waiting_and_cleans_abandoned_write(tmp_path) -> None:
    cache = tmp_path / "cache"
    abandoned = cache / f"{WRITE_TMP_PREFIX}old"
    abandoned.mkdir(parents=True)
    (abandoned / "embeddings.npy").write_bytes(b"partial")
    stream = io.StringIO()
    index = SemanticIndex(cache, CacheConfig(provider="openai", model="text-embedding-3-small"))

    with index.lock(status_stream=stream):
        assert not abandoned.exists()

    output = stream.getvalue()
    assert "Waiting to acquire cache lock:" in output
    assert "Acquired cache lock:" in output
    assert str(cache / "index.lock") in output


def test_lock_blocks_competing_cache_access(tmp_path) -> None:
    cache = tmp_path / "cache"
    config = CacheConfig(provider="openai", model="text-embedding-3-small")
    holder_acquired = Event()
    release_holder = Event()
    waiter_acquired = Event()
    waiter_stream = io.StringIO()

    def hold_lock() -> None:
        with SemanticIndex(cache, config).lock(status_stream=io.StringIO()):
            holder_acquired.set()
            assert release_holder.wait(timeout=2)

    def wait_for_lock() -> None:
        with SemanticIndex(cache, config).lock(status_stream=waiter_stream):
            waiter_acquired.set()

    holder = Thread(target=hold_lock)
    waiter = Thread(target=wait_for_lock)
    holder.start()
    assert holder_acquired.wait(timeout=1)
    waiter.start()

    assert not waiter_acquired.wait(timeout=0.1)
    assert "Waiting to acquire cache lock:" in waiter_stream.getvalue()
    assert "Acquired cache lock:" not in waiter_stream.getvalue()

    release_holder.set()
    assert waiter_acquired.wait(timeout=1)
    holder.join(timeout=1)
    waiter.join(timeout=1)
    assert "Acquired cache lock:" in waiter_stream.getvalue()


def test_interrupted_commit_restores_previous_cache(tmp_path, monkeypatch) -> None:
    source = tmp_path / "note.md"
    source.write_text("old", encoding="utf-8")
    cache = tmp_path / "cache"
    config = CacheConfig(provider="openai", model="text-embedding-3-small")
    index = SemanticIndex(cache, config)
    old_chunk = Chunk(id=f"{source}:0", source_file=str(source), text="old")
    index.replace_files([old_chunk], np.asarray([[1.0, 0.0]], dtype=np.float32), [str(source)])

    def interrupt_after_first_file(staged_dir, backup_dir) -> None:
        name = "cache.json"
        os.replace(cache / name, backup_dir / name)
        os.replace(staged_dir / name, cache / name)
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(index, "_commit_staged_files", interrupt_after_first_file)
    new_chunk = Chunk(id=f"{source}:0", source_file=str(source), text="new")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        index.replace_files([new_chunk], np.asarray([[0.0, 1.0]], dtype=np.float32), [str(source)])

    restored = SemanticIndex(cache, config)
    restored.load()
    assert restored.chunks == [old_chunk]
    assert np.array_equal(restored.embeddings, np.asarray([[1.0, 0.0]], dtype=np.float32))
    assert not restored.transaction_path.exists()


def test_next_lock_holder_recovers_abandoned_transaction(tmp_path) -> None:
    source = tmp_path / "note.md"
    source.write_text("old", encoding="utf-8")
    cache = tmp_path / "cache"
    config = CacheConfig(provider="openai", model="text-embedding-3-small")
    index = SemanticIndex(cache, config)
    old_chunk = Chunk(id=f"{source}:0", source_file=str(source), text="old")
    index.replace_files([old_chunk], np.asarray([[1.0, 0.0]], dtype=np.float32), [str(source)])

    backup_dir = cache / f"{BACKUP_TMP_PREFIX}abandoned"
    backup_dir.mkdir()
    os.replace(cache / "cache.json", backup_dir / "cache.json")
    (cache / "cache.json").write_text("partial replacement", encoding="utf-8")
    index.transaction_path.write_text(
        json.dumps(
            {
                "backup_dir": backup_dir.name,
                "existing_files": ["cache.json", "manifest.json", "chunks.jsonl", "embeddings.npy"],
            }
        ),
        encoding="utf-8",
    )

    recovered = SemanticIndex(cache, config)
    with recovered.lock(status_stream=io.StringIO()):
        recovered.load()

    assert recovered.chunks == [old_chunk]
    assert np.array_equal(recovered.embeddings, np.asarray([[1.0, 0.0]], dtype=np.float32))
    assert not recovered.transaction_path.exists()
    assert not backup_dir.exists()
