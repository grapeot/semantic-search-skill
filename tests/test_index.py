import json
import sqlite3
import time
from pathlib import Path
from threading import Event, Thread

import numpy as np
import pytest

from semantic_search_skill.cli import rank
from semantic_search_skill.index import CacheConfig, CacheError, FileState, SemanticIndex, file_state, migrate_v1_cache
from semantic_search_skill.models import Chunk


CONFIG = CacheConfig(provider="openai", model="text-embedding-3-small")


def _chunk(path: Path, text: str, index: int = 0) -> Chunk:
    return Chunk(id=f"{path}:{index}", source_file=str(path), text=text, header="# H", position=(1, 2))


def _segment_paths(cache: Path) -> list[Path]:
    return sorted((cache / "segments").glob("*.npy"))


def test_index_roundtrip_uses_sqlite_and_segments(tmp_path) -> None:
    source = tmp_path / "note.md"
    source.write_text("# Note\nhello world", encoding="utf-8")
    cache = tmp_path / "cache"
    index = SemanticIndex(cache, CONFIG)
    chunks = [_chunk(source, "hello world")]

    index.replace_files(chunks, np.asarray([[3.0, 0.0, 0.0]], dtype=np.float32), [str(source)])

    assert (cache / "index.sqlite3").exists()
    assert _segment_paths(cache)
    assert not (cache / "chunks.jsonl").exists()
    assert not (cache / "embeddings.npy").exists()
    assert oct((cache / "index.sqlite3").stat().st_mode & 0o777) == "0o600"
    assert oct((cache / "segments").stat().st_mode & 0o777) == "0o700"

    loaded = SemanticIndex(cache, CONFIG)
    loaded.load(require_exists=True)
    subset_chunks, embeddings = loaded.subset([str(source)])

    assert subset_chunks == chunks
    assert embeddings.shape == (1, 3)
    assert np.allclose(embeddings[0], np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    assert loaded.stats()["active_chunk_count"] == 1


def test_schema_mismatch_gives_clear_error(tmp_path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    conn = sqlite3.connect(cache / "index.sqlite3")
    conn.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta VALUES ('schema_version', '99')")
    conn.commit()
    conn.close()

    with pytest.raises(CacheError, match="unsupported cache schema"):
        SemanticIndex(cache, CONFIG).load(require_exists=True)


def test_v1_cache_requires_explicit_migration(tmp_path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "cache.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    with pytest.raises(CacheError, match="run migrate-v1 explicitly"):
        SemanticIndex(cache, CONFIG).load(require_exists=True)


def test_subset_loads_only_matching_segment_files(tmp_path, monkeypatch) -> None:
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_text("alpha", encoding="utf-8")
    b.write_text("beta", encoding="utf-8")
    cache = tmp_path / "cache"
    index = SemanticIndex(cache, CONFIG)
    index.replace_files([_chunk(a, "alpha")], np.asarray([[1.0, 0.0]], dtype=np.float32), [str(a)])
    first_segment = _segment_paths(cache)[0]
    index.replace_files([_chunk(b, "beta")], np.asarray([[0.0, 1.0]], dtype=np.float32), [str(b)])
    loaded_paths: list[Path] = []
    real_load = np.load

    def recording_load(path, *args, **kwargs):
        loaded_paths.append(Path(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", recording_load)

    chunks, embeddings = index.subset([str(a)])

    assert [chunk.source_file for chunk in chunks] == [str(a)]
    assert embeddings.shape == (1, 2)
    assert loaded_paths == [first_segment]


def test_incremental_update_does_not_rewrite_old_segment(tmp_path) -> None:
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_text("alpha", encoding="utf-8")
    b.write_text("beta", encoding="utf-8")
    cache = tmp_path / "cache"
    index = SemanticIndex(cache, CONFIG)
    index.replace_files([_chunk(a, "alpha")], np.asarray([[1.0, 0.0]], dtype=np.float32), [str(a)])
    old_segment = _segment_paths(cache)[0]
    old_mtime = old_segment.stat().st_mtime_ns
    time.sleep(0.001)

    index.replace_files([_chunk(b, "beta")], np.asarray([[0.0, 1.0]], dtype=np.float32), [str(b)])

    assert old_segment.exists()
    assert old_segment.stat().st_mtime_ns == old_mtime
    assert len(_segment_paths(cache)) == 2

    a.write_text("alpha updated", encoding="utf-8")
    index.replace_files([_chunk(a, "alpha updated")], np.asarray([[0.5, 0.5]], dtype=np.float32), [str(a)])

    assert old_segment.stat().st_mtime_ns == old_mtime
    assert len(_segment_paths(cache)) == 3
    assert index.search([str(a)], np.asarray([0.5, 0.5], dtype=np.float32), 1)[0].chunk.text == "alpha updated"


def test_load_does_not_materialize_global_file_manifest(tmp_path) -> None:
    source = tmp_path / "note.md"
    source.write_text("hello", encoding="utf-8")
    cache = tmp_path / "cache"
    index = SemanticIndex(cache, CONFIG)
    index.replace_files([_chunk(source, "hello")], np.asarray([[1.0, 0.0]], dtype=np.float32), [str(source)])

    index.load(require_exists=True)

    assert index.meta["schema_version"] == 2
    assert index.manifest == {}


def test_cross_segment_top_k_matches_monolithic_rank(tmp_path) -> None:
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    c = tmp_path / "c.md"
    for path in (a, b, c):
        path.write_text(path.stem, encoding="utf-8")
    cache = tmp_path / "cache"
    index = SemanticIndex(cache, CONFIG)
    first_chunks = [_chunk(a, "alpha"), _chunk(b, "beta")]
    first_vectors = np.asarray([[1.0, 0.0], [0.7, 0.7]], dtype=np.float32)
    index.replace_files(first_chunks, first_vectors, [str(a), str(b)])
    second_chunks = [_chunk(c, "gamma")]
    second_vectors = np.asarray([[0.0, 1.0]], dtype=np.float32)
    index.replace_files(second_chunks, second_vectors, [str(c)])

    query = np.asarray([0.0, 1.0], dtype=np.float32)
    results = index.search([str(a), str(b), str(c)], query, 2)
    expected = rank(first_chunks + second_chunks, np.vstack([first_vectors, second_vectors]), query, 2)

    assert [result.chunk.id for result in results] == [result.chunk.id for result in expected]


def test_query_reader_uses_sqlite_snapshot_during_writer_transaction(tmp_path) -> None:
    source = tmp_path / "note.md"
    source.write_text("old", encoding="utf-8")
    cache = tmp_path / "cache"
    index = SemanticIndex(cache, CONFIG)
    index.replace_files([_chunk(source, "old")], np.asarray([[1.0, 0.0]], dtype=np.float32), [str(source)])
    writer_started = Event()
    release_writer = Event()
    writer_error: list[BaseException] = []

    def writer() -> None:
        try:
            with index.connect(readonly=False) as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("UPDATE files SET chunk_count = chunk_count WHERE path = ?", (str(source),))
                writer_started.set()
                assert release_writer.wait(timeout=2)
                conn.rollback()
        except BaseException as exc:  # pragma: no cover - reported below
            writer_error.append(exc)

    thread = Thread(target=writer)
    thread.start()
    assert writer_started.wait(timeout=1)

    results = index.search([str(source)], np.asarray([1.0, 0.0], dtype=np.float32), 1)

    release_writer.set()
    thread.join(timeout=1)
    assert not writer_error
    assert results[0].chunk.text == "old"


def test_query_uses_one_snapshot_across_file_list_batches(tmp_path) -> None:
    sources = []
    chunks = []
    vectors = []
    for index_value in range(901):
        source = tmp_path / f"note-{index_value}.md"
        source.write_text("old", encoding="utf-8")
        sources.append(source)
        chunks.append(_chunk(source, "old"))
        vectors.append([1.0, 0.0])
    cache = tmp_path / "cache"
    index = SemanticIndex(cache, CONFIG)
    index.replace_files(chunks, np.asarray(vectors, dtype=np.float32), [str(path) for path in sources])

    with index.connect(readonly=True) as conn:
        conn.execute("BEGIN")
        rows = index._iter_active_rows(conn, [str(path) for path in sources], include_content=True)
        first_batch = [next(rows) for _ in range(900)]
        last_source = sources[-1]
        last_source.write_text("new", encoding="utf-8")
        index.replace_files([_chunk(last_source, "new")], np.asarray([[0.0, 1.0]], dtype=np.float32), [str(last_source)])
        last_row = next(rows)
        assert list(rows) == []

    assert all(row["text"] == "old" for row in first_batch)
    assert last_row["text"] == "old"


def test_query_reader_does_not_wait_for_writer_file_lock(tmp_path) -> None:
    source = tmp_path / "note.md"
    source.write_text("old", encoding="utf-8")
    cache = tmp_path / "cache"
    index = SemanticIndex(cache, CONFIG)
    index.replace_files([_chunk(source, "old")], np.asarray([[1.0, 0.0]], dtype=np.float32), [str(source)])

    with index.writer_lock():
        results = index.search([str(source)], np.asarray([1.0, 0.0], dtype=np.float32), 1)

    assert results[0].chunk.text == "old"


def test_doctor_reports_and_cleans_orphan_segment_files(tmp_path) -> None:
    source = tmp_path / "note.md"
    source.write_text("hello", encoding="utf-8")
    cache = tmp_path / "cache"
    index = SemanticIndex(cache, CONFIG)
    index.replace_files([_chunk(source, "hello")], np.asarray([[1.0, 0.0]], dtype=np.float32), [str(source)])
    orphan = cache / "segments" / "orphan.npy"
    np.save(orphan, np.asarray([[0.0, 1.0]], dtype=np.float32))

    report = index.doctor()
    assert report["ok"] is True
    assert report["orphan_segment_files"] == 1

    cleaned = index.doctor(cleanup_orphans=True)
    assert cleaned["cleaned_orphan_segment_files"] == 1
    assert not orphan.exists()


def test_orphan_cleanup_waits_for_inflight_segment_publish(tmp_path, monkeypatch) -> None:
    source = tmp_path / "note.md"
    source.write_text("hello", encoding="utf-8")
    cache = tmp_path / "cache"
    index = SemanticIndex(cache, CONFIG)
    with index.connect(readonly=False):
        pass
    segment_written = Event()
    release_writer = Event()
    cleanup_done = Event()
    real_write = index._write_segment

    def paused_write(embeddings):
        result = real_write(embeddings)
        segment_written.set()
        assert release_writer.wait(timeout=2)
        return result

    monkeypatch.setattr(index, "_write_segment", paused_write)
    writer = Thread(
        target=lambda: index.replace_files(
            [_chunk(source, "hello")],
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            [str(source)],
        )
    )
    cleaner = Thread(target=lambda: (index.doctor(cleanup_orphans=True), cleanup_done.set()))
    writer.start()
    assert segment_written.wait(timeout=1)
    cleaner.start()
    assert not cleanup_done.wait(timeout=0.1)
    release_writer.set()
    writer.join(timeout=2)
    cleaner.join(timeout=2)

    assert cleanup_done.is_set()
    assert index.search([str(source)], np.asarray([1.0, 0.0], dtype=np.float32), 1)[0].chunk.text == "hello"


def test_publish_skips_file_changed_while_segment_is_written(tmp_path, monkeypatch) -> None:
    source = tmp_path / "note.md"
    source.write_text("old", encoding="utf-8")
    cache = tmp_path / "cache"
    index = SemanticIndex(cache, CONFIG)
    index.replace_files([_chunk(source, "old")], np.asarray([[1.0, 0.0]], dtype=np.float32), [str(source)])
    source.write_text("new", encoding="utf-8")
    state = file_state(source)
    real_write = index._write_segment

    def write_then_change(embeddings):
        result = real_write(embeddings)
        source.write_text("newer", encoding="utf-8")
        return result

    monkeypatch.setattr(index, "_write_segment", write_then_change)

    skipped = index.replace_files(
        [_chunk(source, "new")],
        np.asarray([[0.0, 1.0]], dtype=np.float32),
        [str(source)],
        file_states={str(source): state},
    )

    assert skipped == [str(source)]
    results = index.search([str(source)], np.asarray([1.0, 0.0], dtype=np.float32), 1)
    assert results[0].chunk.text == "old"


def test_doctor_detects_segment_row_mismatch(tmp_path) -> None:
    source = tmp_path / "note.md"
    source.write_text("hello", encoding="utf-8")
    cache = tmp_path / "cache"
    index = SemanticIndex(cache, CONFIG)
    index.replace_files([_chunk(source, "hello")], np.asarray([[1.0, 0.0]], dtype=np.float32), [str(source)])
    np.save(_segment_paths(cache)[0], np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))

    report = index.doctor()

    assert report["ok"] is False
    assert any("row count" in problem for problem in report["problems"])


def test_doctor_reports_truncated_segment_instead_of_crashing(tmp_path) -> None:
    source = tmp_path / "note.md"
    source.write_text("hello", encoding="utf-8")
    cache = tmp_path / "cache"
    index = SemanticIndex(cache, CONFIG)
    index.replace_files([_chunk(source, "hello")], np.asarray([[1.0, 0.0]], dtype=np.float32), [str(source)])
    _segment_paths(cache)[0].write_bytes(b"not-an-npy")

    report = index.doctor()

    assert report["ok"] is False
    assert any("cannot be loaded" in problem for problem in report["problems"])


def test_doctor_refuses_cleanup_through_symlinked_tmp_directory(tmp_path) -> None:
    source = tmp_path / "note.md"
    source.write_text("hello", encoding="utf-8")
    cache = tmp_path / "cache"
    index = SemanticIndex(cache, CONFIG)
    index.replace_files([_chunk(source, "hello")], np.asarray([[1.0, 0.0]], dtype=np.float32), [str(source)])
    outside = tmp_path / "outside"
    outside.mkdir()
    index.tmp_dir.rmdir()
    index.tmp_dir.symlink_to(outside, target_is_directory=True)

    with pytest.raises(CacheError, match="refusing cleanup"):
        index.doctor(cleanup_orphans=True)


def test_delete_or_shrink_marks_old_chunks_inactive_without_compaction(tmp_path) -> None:
    source = tmp_path / "note.md"
    source.write_text("hello", encoding="utf-8")
    cache = tmp_path / "cache"
    index = SemanticIndex(cache, CONFIG)
    index.replace_files([_chunk(source, "old")], np.asarray([[1.0, 0.0]], dtype=np.float32), [str(source)])
    old_segment = _segment_paths(cache)[0]
    source.unlink()

    index.replace_files([], np.empty((0, 2), dtype=np.float32), [str(source)], file_states={str(source): FileState(exists=False)})

    assert index.search([str(source)], np.asarray([1.0, 0.0], dtype=np.float32), 1) == []
    assert old_segment.exists()
    report = index.doctor()
    assert report["inactive_chunk_count"] == 1
    assert report["compaction_recommended"] is True


def test_tiny_v1_migration_streams_to_v2_and_preserves_v1_files(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "note.md"
    source.write_text("hello", encoding="utf-8")
    v1 = tmp_path / "v1"
    v2 = tmp_path / "v2"
    v1.mkdir()
    relative_source = Path("note.md")
    chunks = [_chunk(relative_source, "hello", 0), _chunk(relative_source, "world", 1)]
    (v1 / "cache.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "embedding_provider": "openai",
                "embedding_model": "text-embedding-3-small",
                "embedding_dimension": 2,
                "created_at": "2026-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    with (v1 / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    (v1 / "manifest.json").write_text(
        json.dumps({str(relative_source): {"mtime_ns": source.stat().st_mtime_ns, "chunk_count": 2}}),
        encoding="utf-8",
    )
    np.save(v1 / "embeddings.npy", np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32))

    monkeypatch.chdir(tmp_path)
    event = migrate_v1_cache(v1, v2, CONFIG, segment_size=1)

    assert event["chunks_migrated"] == 2
    assert event["segments_created"] == 2
    assert (v1 / "chunks.jsonl").exists()
    migrated = SemanticIndex(v2, CONFIG)
    monkeypatch.chdir(workspace)
    result_chunks, embeddings = migrated.subset([str(relative_source)])
    assert [chunk.text for chunk in result_chunks] == ["hello", "world"]
    assert np.allclose(embeddings[1], np.asarray([0.0, 1.0], dtype=np.float32))
    assert migrated.changed_files([str(relative_source)]) == []


def test_migration_is_rerunnable_after_pre_publish_crash_orphan(tmp_path) -> None:
    v1 = tmp_path / "v1"
    v2 = tmp_path / "v2"
    v1.mkdir()
    source = tmp_path / "note.md"
    source.write_text("hello", encoding="utf-8")
    (v1 / "cache.json").write_text(
        json.dumps({"schema_version": 1, "embedding_provider": "openai", "embedding_model": "text-embedding-3-small", "embedding_dimension": 2}),
        encoding="utf-8",
    )
    (v1 / "chunks.jsonl").write_text(json.dumps(_chunk(source, "hello").to_dict()) + "\n", encoding="utf-8")
    np.save(v1 / "embeddings.npy", np.asarray([[1.0, 0.0]], dtype=np.float32))
    (v2 / "segments").mkdir(parents=True)
    np.save(v2 / "segments" / "segment-abandoned.npy", np.asarray([[0.0, 1.0]], dtype=np.float32))

    migrate_v1_cache(v1, v2, CONFIG)
    index = SemanticIndex(v2, CONFIG)

    assert index.doctor()["orphan_segment_files"] == 1
    assert index.doctor(cleanup_orphans=True)["cleaned_orphan_segment_files"] == 1


def test_failed_migration_removes_segments_created_by_that_attempt(tmp_path) -> None:
    v1 = tmp_path / "v1"
    v2 = tmp_path / "v2"
    v1.mkdir()
    source = tmp_path / "note.md"
    source.write_text("hello", encoding="utf-8")
    (v1 / "cache.json").write_text(
        json.dumps({"schema_version": 1, "embedding_provider": "openai", "embedding_model": "text-embedding-3-small", "embedding_dimension": 2}),
        encoding="utf-8",
    )
    (v1 / "chunks.jsonl").write_text(json.dumps(_chunk(source, "hello").to_dict()) + "\nnot-json\n", encoding="utf-8")
    np.save(v1 / "embeddings.npy", np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))

    with pytest.raises(json.JSONDecodeError):
        migrate_v1_cache(v1, v2, CONFIG, segment_size=1)

    assert not list((v2 / "segments").glob("*.npy"))
    assert not (v2 / "index.sqlite3").exists()


def test_cache_subdirectories_reject_symlinks(tmp_path) -> None:
    cache = tmp_path / "cache"
    outside = tmp_path / "outside"
    cache.mkdir()
    outside.mkdir()
    (cache / "segments").symlink_to(outside, target_is_directory=True)
    source = tmp_path / "note.md"
    source.write_text("hello", encoding="utf-8")

    with pytest.raises(CacheError, match="must not be a symlink"):
        SemanticIndex(cache, CONFIG).replace_files(
            [_chunk(source, "hello")],
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            [str(source)],
        )


def test_changed_files_detects_shrink_and_delete(tmp_path) -> None:
    source = tmp_path / "note.md"
    source.write_text("hello world", encoding="utf-8")
    cache = tmp_path / "cache"
    index = SemanticIndex(cache, CONFIG)
    index.replace_files([_chunk(source, "hello world")], np.asarray([[1.0, 0.0]], dtype=np.float32), [str(source)])
    source.write_text("hi", encoding="utf-8")

    assert index.changed_files([str(source)]) == [str(source)]

    index.replace_files([_chunk(source, "hi")], np.asarray([[0.0, 1.0]], dtype=np.float32), [str(source)], file_states={str(source): file_state(source)})
    source.unlink()

    assert index.changed_files([str(source)]) == [str(source)]
