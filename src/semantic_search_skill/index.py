from __future__ import annotations

import heapq
import json
import os
import shutil
import sqlite3
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from filelock import FileLock

from .models import Chunk, SearchResult

SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1
DB_NAME = "index.sqlite3"
SEGMENTS_DIR = "segments"
TMP_DIR = "tmp"
WRITER_LOCK_NAME = "writer.lock"
WRITE_TMP_PREFIX = ".semantic-search-write-"
MIGRATE_TMP_PREFIX = ".semantic-search-migrate-"
PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


class CacheError(RuntimeError):
    pass


@dataclass(frozen=True)
class CacheConfig:
    provider: str
    model: str
    dimension: int | None = None


@dataclass(frozen=True)
class FileState:
    exists: bool
    mtime_ns: int = 0
    size_bytes: int = 0


class SemanticIndex:
    def __init__(self, cache_dir: str | Path, config: CacheConfig) -> None:
        self.cache_dir = Path(cache_dir)
        self.config = config
        self.db_path = self.cache_dir / DB_NAME
        self.segments_dir = self.cache_dir / SEGMENTS_DIR
        self.tmp_dir = self.cache_dir / TMP_DIR
        self.writer_lock_path = self.cache_dir / WRITER_LOCK_NAME
        self.legacy_cache_path = self.cache_dir / "cache.json"
        self.legacy_chunks_path = self.cache_dir / "chunks.jsonl"
        self.legacy_embeddings_path = self.cache_dir / "embeddings.npy"
        self.meta: dict[str, Any] = {}
        self.manifest: dict[str, Any] = {}
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None

    def exists(self) -> bool:
        return self.db_path.exists() or self.legacy_cache_path.exists() or self.legacy_chunks_path.exists() or self.legacy_embeddings_path.exists()

    @contextmanager
    def connect(self, *, readonly: bool = False) -> Iterator[sqlite3.Connection]:
        if readonly:
            self._raise_if_legacy_only()
            if not self.db_path.exists():
                raise CacheError(f"v2 cache database missing: {self.db_path}")
            uri = f"file:{self.db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
        else:
            _ensure_private_dir(self.cache_dir)
            _ensure_private_dir(self.segments_dir)
            _ensure_private_dir(self.tmp_dir)
            self._raise_if_legacy_only()
            conn = sqlite3.connect(self.db_path)
            _chmod_private_file(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout = 30000")
            if not readonly:
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = FULL")
                conn.execute("PRAGMA foreign_keys = ON")
                _create_schema(conn)
                self._ensure_meta(conn)
                conn.commit()
                _chmod_sqlite_sidecars(self.db_path)
            self._validate_meta(conn)
            yield conn
        finally:
            conn.close()

    def _raise_if_legacy_only(self) -> None:
        if self.db_path.exists():
            return
        if self.legacy_cache_path.exists() or self.legacy_chunks_path.exists() or self.legacy_embeddings_path.exists():
            raise CacheError("cache uses v1 JSONL+NPY layout; run migrate-v1 explicitly before using the v2 SQLite cache")

    def _ensure_meta(self, conn: sqlite3.Connection) -> None:
        row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        if row is not None:
            return
        now = _now()
        _set_meta(
            conn,
            {
                "schema_version": SCHEMA_VERSION,
                "embedding_provider": self.config.provider,
                "embedding_model": self.config.model,
                "embedding_dimension": self.config.dimension,
                "created_at": now,
                "updated_at": now,
            },
        )

    def _validate_meta(self, conn: sqlite3.Connection) -> None:
        meta = _read_meta(conn)
        if not meta:
            return
        if int(meta.get("schema_version", -1)) != SCHEMA_VERSION:
            raise CacheError(f"unsupported cache schema: {meta.get('schema_version')}; expected {SCHEMA_VERSION}")
        if meta.get("embedding_provider") != self.config.provider:
            raise CacheError("cache provider does not match requested provider")
        if meta.get("embedding_model") != self.config.model:
            raise CacheError("cache model does not match requested model")
        dimension = meta.get("embedding_dimension")
        if self.config.dimension and dimension and int(dimension) != self.config.dimension:
            raise CacheError("cache dimension does not match requested dimension")

    def load(self, *, mmap: bool = True, require_exists: bool = False) -> None:
        _ = mmap
        if not self.db_path.exists():
            self._raise_if_legacy_only()
            if require_exists:
                raise CacheError(f"v2 cache database missing: {self.db_path}")
            return
        with self.connect(readonly=True) as conn:
            self.meta = _read_meta(conn)
            self.manifest = {}
            self.chunks = []
            self.embeddings = None

    @contextmanager
    def writer_lock(self) -> Iterator[None]:
        _ensure_private_dir(self.cache_dir)
        with FileLock(self.writer_lock_path):
            _chmod_private_file(self.writer_lock_path)
            yield

    def changed_files(self, file_paths: list[str]) -> list[str]:
        if not self.db_path.exists():
            self._raise_if_legacy_only()
            return [path for path in file_paths if Path(path).exists()]
        changed: list[str] = []
        states = {path: file_state(path) for path in file_paths}
        records = self._file_records(file_paths)
        for path in file_paths:
            state = states[path]
            record = records.get(path)
            if state.exists:
                stored_size = int(record["size_bytes"]) if record is not None else -1
                if (
                    record is None
                    or not record["active"]
                    or int(record["mtime_ns"]) != state.mtime_ns
                    or (stored_size >= 0 and stored_size != state.size_bytes)
                ):
                    changed.append(path)
            elif record is not None and record["active"]:
                changed.append(path)
        return changed

    def _file_records(self, file_paths: Sequence[str]) -> dict[str, sqlite3.Row]:
        if not file_paths:
            return {}
        records: dict[str, sqlite3.Row] = {}
        with self.connect(readonly=True) as conn:
            for batch in _batched(list(file_paths), 900):
                placeholders = ",".join("?" for _ in batch)
                for row in conn.execute(f"SELECT path, mtime_ns, size_bytes, active FROM files WHERE path IN ({placeholders})", batch):
                    records[str(row["path"])] = row
        return records

    def replace_files(
        self,
        new_chunks: list[Chunk],
        new_embeddings: np.ndarray,
        updated_files: list[str],
        file_states: dict[str, FileState] | None = None,
    ) -> list[str]:
        if not updated_files:
            return []
        states = file_states or {path: file_state(path) for path in updated_files}
        embeddings = _as_float32_matrix(new_embeddings, self.config.dimension)
        if len(new_chunks) != int(embeddings.shape[0]):
            raise CacheError(f"chunk count {len(new_chunks)} does not match embedding rows {embeddings.shape[0]}")
        dimension = int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.shape[1] else self.config.dimension
        segment_id: int | None = None
        segment_relpath: str | None = None
        with self.writer_lock():
            publish_files = list(updated_files)
            publish_chunks = list(new_chunks)
            publish_embeddings = embeddings
            skipped: list[str] = []
            while True:
                unstable = {path for path in publish_files if file_state(path) != states[path]}
                if unstable:
                    skipped.extend(path for path in publish_files if path in unstable)
                    keep_files = set(publish_files) - unstable
                    keep_rows = [index for index, chunk in enumerate(publish_chunks) if chunk.source_file in keep_files]
                    publish_files = [path for path in publish_files if path in keep_files]
                    publish_chunks = [publish_chunks[index] for index in keep_rows]
                    publish_embeddings = (
                        publish_embeddings[keep_rows]
                        if keep_rows
                        else np.empty((0, publish_embeddings.shape[1] if publish_embeddings.ndim == 2 else dimension or 0), dtype=np.float32)
                    )
                if not publish_chunks:
                    segment_path = None
                    segment_row_count = 0
                    break
                normalized = normalize_vectors(publish_embeddings)
                segment_relpath, segment_path = self._write_segment(normalized)
                segment_row_count = int(normalized.shape[0])
                changed_during_write = {path for path in publish_files if file_state(path) != states[path]}
                if not changed_during_write:
                    break
                segment_path.unlink(missing_ok=True)
                _fsync_dir(self.segments_dir)
                segment_relpath = None
            with self.connect(readonly=False) as conn:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    if dimension:
                        _update_dimension(conn, dimension)
                    if segment_relpath is not None and segment_path is not None:
                        cursor = conn.execute(
                            """
                            INSERT INTO segments(path, row_count, dimension, state, size_bytes, created_at)
                            VALUES (?, ?, ?, 'active', ?, ?)
                            """,
                            (segment_relpath, segment_row_count, dimension, segment_path.stat().st_size, _now()),
                        )
                        segment_id = int(cursor.lastrowid)
                    for path in publish_files:
                        state = states[path]
                        conn.execute("UPDATE chunks SET active = 0 WHERE source_file = ? AND active = 1", (path,))
                        conn.execute(
                            """
                            INSERT INTO files(path, mtime_ns, size_bytes, active, chunk_count, updated_at)
                            VALUES (?, ?, ?, ?, 0, ?)
                            ON CONFLICT(path) DO UPDATE SET
                                mtime_ns = excluded.mtime_ns,
                                size_bytes = excluded.size_bytes,
                                active = excluded.active,
                                chunk_count = 0,
                                updated_at = excluded.updated_at
                            """,
                            (path, state.mtime_ns, state.size_bytes, 1 if state.exists else 0, _now()),
                        )
                    if segment_id is not None:
                        for row_index, chunk in enumerate(publish_chunks):
                            conn.execute(
                                """
                                INSERT INTO chunks(
                                    chunk_uid, source_file, text, header, start_line, end_line,
                                    metadata_json, segment_id, segment_row, active, created_at
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                                """,
                                (
                                    chunk.id,
                                    chunk.source_file,
                                    chunk.text,
                                    chunk.header,
                                    int(chunk.position[0]),
                                    int(chunk.position[1]),
                                    json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True),
                                    segment_id,
                                    row_index,
                                    _now(),
                                ),
                            )
                            conn.execute("UPDATE files SET chunk_count = chunk_count + 1 WHERE path = ?", (chunk.source_file,))
                    _set_meta(conn, {"updated_at": _now()})
                    conn.commit()
                except BaseException:
                    conn.rollback()
                    if segment_path is not None:
                        segment_path.unlink(missing_ok=True)
                        _fsync_dir(self.segments_dir)
                    raise
        _fsync_dir(self.cache_dir)
        return skipped

    def _write_segment(self, embeddings: np.ndarray) -> tuple[str, Path]:
        _ensure_private_dir(self.cache_dir)
        _ensure_private_dir(self.segments_dir)
        _ensure_private_dir(self.tmp_dir)
        name = f"segment-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}-{uuid.uuid4().hex}.npy"
        final = self.segments_dir / name
        with tempfile.NamedTemporaryFile(prefix=WRITE_TMP_PREFIX, suffix=".npy", dir=self.tmp_dir, delete=False) as handle:
            tmp_path = Path(handle.name)
            np.save(handle, embeddings.astype(np.float32, copy=False))
            handle.flush()
            os.fsync(handle.fileno())
        _chmod_private_file(tmp_path)
        os.replace(tmp_path, final)
        _chmod_private_file(final)
        _fsync_dir(self.segments_dir)
        return f"{SEGMENTS_DIR}/{name}", final

    def subset(self, file_paths: list[str]) -> tuple[list[Chunk], np.ndarray]:
        with self.connect(readonly=True) as conn:
            conn.execute("BEGIN")
            rows = list(self._iter_active_rows(conn, file_paths, include_content=True))
        if not rows:
            return [], np.empty((0, self.config.dimension or 0), dtype=np.float32)
        chunks: list[Chunk] = []
        vectors: list[np.ndarray] = []
        for segment_id, segment_rows in _group_rows_by_segment(rows).items():
            segment_path = self._segment_path(segment_rows[0]["segment_path"])
            matrix = np.load(segment_path, mmap_mode="r")
            indices = [int(row["segment_row"]) for row in segment_rows]
            selected = np.asarray(matrix[indices], dtype=np.float32)
            vectors.append(selected)
            chunks.extend(_chunk_from_row(row) for row in segment_rows)
            _ = segment_id
        return chunks, np.vstack(vectors).astype(np.float32, copy=False)

    def iter_active_rows(self, file_paths: list[str], *, include_content: bool = True) -> Iterator[sqlite3.Row]:
        if not file_paths:
            return
        with self.connect(readonly=True) as conn:
            conn.execute("BEGIN")
            yield from self._iter_active_rows(conn, file_paths, include_content=include_content)

    def _iter_active_rows(self, conn: sqlite3.Connection, file_paths: list[str], *, include_content: bool) -> Iterator[sqlite3.Row]:
        for batch in _batched(file_paths, 900):
            placeholders = ",".join("?" for _ in batch)
            content_columns = ", c.chunk_uid, c.source_file, c.text, c.header, c.start_line, c.end_line, c.metadata_json" if include_content else ""
            sql = f"""
                SELECT c.id, c.segment_id, c.segment_row, s.path AS segment_path{content_columns}
                FROM chunks c
                JOIN segments s ON s.id = c.segment_id
                WHERE c.active = 1 AND s.state = 'active' AND c.source_file IN ({placeholders})
                ORDER BY c.segment_id, c.segment_row
            """
            yield from conn.execute(sql, batch)

    def search(self, file_paths: list[str], query_vector: np.ndarray, top_k: int, *, row_batch_size: int = 4096) -> list[SearchResult]:
        if top_k <= 0:
            return []
        query = normalize_query(query_vector)
        heap: list[tuple[float, int, int]] = []
        counter = 0
        pending_segment_id: int | None = None
        pending_rows: list[sqlite3.Row] = []

        def flush() -> None:
            nonlocal counter, pending_rows
            if not pending_rows:
                return
            segment_path = self._segment_path(pending_rows[0]["segment_path"])
            matrix = np.load(segment_path, mmap_mode="r")
            for start in range(0, len(pending_rows), row_batch_size):
                rows = pending_rows[start : start + row_batch_size]
                indices = [int(row["segment_row"]) for row in rows]
                vectors = np.nan_to_num(np.asarray(matrix[indices], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                scores = vectors @ query
                for row, score in zip(rows, scores, strict=True):
                    item = (float(score), counter, int(row["id"]))
                    counter += 1
                    if len(heap) < top_k:
                        heapq.heappush(heap, item)
                    elif score > heap[0][0]:
                        heapq.heapreplace(heap, item)
            pending_rows = []

        with self.connect(readonly=True) as conn:
            conn.execute("BEGIN")
            for row in self._iter_active_rows(conn, file_paths, include_content=False):
                segment_id = int(row["segment_id"])
                if pending_segment_id is None:
                    pending_segment_id = segment_id
                if segment_id != pending_segment_id:
                    flush()
                    pending_segment_id = segment_id
                pending_rows.append(row)
            flush()
            ranked = sorted(heap, key=lambda value: (value[0], -value[1]), reverse=True)
            chunks_by_id = self._chunks_by_ids(conn, [item[2] for item in ranked])
        return [SearchResult(chunk=chunks_by_id[item[2]], score=item[0]) for item in ranked]

    def _chunks_by_ids(self, conn: sqlite3.Connection, chunk_ids: list[int]) -> dict[int, Chunk]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = conn.execute(
            f"""
            SELECT id, chunk_uid, source_file, text, header, start_line, end_line, metadata_json
            FROM chunks WHERE id IN ({placeholders})
            """,
            chunk_ids,
        )
        return {int(row["id"]): _chunk_from_row(row) for row in rows}

    def _segment_path(self, relpath: str) -> Path:
        if Path(relpath).is_absolute():
            raise CacheError("invalid absolute segment path")
        cache_root = self.cache_dir.resolve()
        path = (cache_root / relpath).resolve()
        if not path.is_relative_to(cache_root):
            raise CacheError("invalid segment path outside cache directory")
        return path

    def stats(self) -> dict[str, Any]:
        with self.connect(readonly=True) as conn:
            meta = _read_meta(conn)
            counts = conn.execute(
                """
                SELECT
                    (SELECT count(*) FROM files) AS file_count,
                    (SELECT count(*) FROM files WHERE active = 1) AS active_file_count,
                    (SELECT count(*) FROM chunks) AS chunk_count,
                    (SELECT count(*) FROM chunks WHERE active = 1) AS active_chunk_count,
                    (SELECT count(*) FROM segments) AS segment_count
                """
            ).fetchone()
            return {
                "cache_dir": str(self.cache_dir),
                "schema_version": SCHEMA_VERSION,
                "file_count": int(counts["file_count"]),
                "active_file_count": int(counts["active_file_count"]),
                "chunk_count": int(counts["chunk_count"]),
                "active_chunk_count": int(counts["active_chunk_count"]),
                "segment_count": int(counts["segment_count"]),
                "meta": meta,
            }

    def doctor(self, *, cleanup_orphans: bool = False) -> dict[str, Any]:
        problems: list[str] = []
        segments_safe = not self.segments_dir.is_symlink()
        tmp_safe = not self.tmp_dir.is_symlink()
        if not segments_safe:
            problems.append("segments directory must not be a symlink")
        if not tmp_safe:
            problems.append("tmp directory must not be a symlink")
        with self.connect(readonly=True) as conn:
            meta = _read_meta(conn)
            dimension = int(meta.get("embedding_dimension") or 0)
            referenced_segment_paths = set()
            segment_rows = conn.execute("SELECT id, path, row_count, dimension FROM segments") if segments_safe else []
            for row in segment_rows:
                referenced_segment_paths.add(str(row["path"]))
                path = self._segment_path(str(row["path"]))
                if not path.exists():
                    problems.append(f"missing segment file for segment {row['id']}: {row['path']}")
                    continue
                try:
                    matrix = np.load(path, mmap_mode="r")
                except (OSError, ValueError) as exc:
                    problems.append(f"segment {row['id']} cannot be loaded: {type(exc).__name__}")
                    continue
                if matrix.ndim != 2:
                    problems.append(f"segment {row['id']} must be a 2D matrix")
                    continue
                if matrix.dtype != np.float32:
                    problems.append(f"segment {row['id']} dtype is {matrix.dtype}, expected float32")
                if int(matrix.shape[0]) != int(row["row_count"]):
                    problems.append(f"segment {row['id']} row count {matrix.shape[0]} does not match metadata {row['row_count']}")
                if dimension and int(matrix.shape[1]) != dimension:
                    problems.append(f"segment {row['id']} dimension {matrix.shape[1]} does not match cache dimension {dimension}")
            bad_refs = conn.execute(
                """
                SELECT count(*) AS count
                FROM chunks c JOIN segments s ON s.id = c.segment_id
                WHERE c.segment_row < 0 OR c.segment_row >= s.row_count
                """
            ).fetchone()["count"]
            if int(bad_refs):
                problems.append(f"{bad_refs} chunk rows reference segment rows outside segment bounds")
            inactive_chunks = int(conn.execute("SELECT count(*) AS count FROM chunks WHERE active = 0").fetchone()["count"])
        segment_files = (
            {f"{SEGMENTS_DIR}/{path.name}" for path in self.segments_dir.glob("*.npy")}
            if segments_safe and self.segments_dir.exists()
            else set()
        )
        orphan_segment_files = sorted(segment_files - referenced_segment_paths)
        abandoned_temp_paths = (
            sorted(
                path.name
                for prefix in (WRITE_TMP_PREFIX, MIGRATE_TMP_PREFIX)
                for path in self.tmp_dir.glob(f"{prefix}*")
            )
            if tmp_safe and self.tmp_dir.exists()
            else []
        )
        cleaned_orphans = 0
        cleaned_temp = 0
        if cleanup_orphans:
            if not segments_safe or not tmp_safe:
                raise CacheError("refusing cleanup because a cache subdirectory is a symlink")
            with self.writer_lock():
                with self.connect(readonly=True) as conn:
                    live_paths = {str(row["path"]) for row in conn.execute("SELECT path FROM segments")}
                orphan_segment_files = sorted(segment_files - live_paths)
                for relpath in orphan_segment_files:
                    self._segment_path(relpath).unlink(missing_ok=True)
                    cleaned_orphans += 1
                for name in abandoned_temp_paths:
                    path = self.tmp_dir / name
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink(missing_ok=True)
                    cleaned_temp += 1
                if cleaned_orphans:
                    _fsync_dir(self.segments_dir)
                if cleaned_temp:
                    _fsync_dir(self.tmp_dir)
        return {
            "ok": not problems,
            "problems": problems,
            "orphan_segment_files": len(orphan_segment_files),
            "abandoned_temp_paths": len(abandoned_temp_paths),
            "cleaned_orphan_segment_files": cleaned_orphans,
            "cleaned_temp_paths": cleaned_temp,
            "inactive_chunk_count": inactive_chunks,
            "compaction_recommended": inactive_chunks > 0,
            "stats": self.stats(),
        }


def migrate_v1_cache(v1_cache_dir: str | Path, v2_cache_dir: str | Path, config: CacheConfig, *, segment_size: int = 50_000) -> dict[str, Any]:
    v1_dir = Path(v1_cache_dir)
    v2_dir = Path(v2_cache_dir)
    v1_meta_path = v1_dir / "cache.json"
    v1_chunks_path = v1_dir / "chunks.jsonl"
    v1_embeddings_path = v1_dir / "embeddings.npy"
    v1_manifest_path = v1_dir / "manifest.json"
    if not v1_meta_path.exists() or not v1_chunks_path.exists() or not v1_embeddings_path.exists():
        raise CacheError("v1 cache must contain cache.json, chunks.jsonl, and embeddings.npy")
    v1_meta = json.loads(v1_meta_path.read_text(encoding="utf-8"))
    v1_manifest = json.loads(v1_manifest_path.read_text(encoding="utf-8")) if v1_manifest_path.exists() else {}
    if int(v1_meta.get("schema_version", -1)) != LEGACY_SCHEMA_VERSION:
        raise CacheError(f"migrate-v1 expected schema_version 1, found {v1_meta.get('schema_version')}")
    if v1_meta.get("embedding_provider") != config.provider:
        raise CacheError("v1 cache provider does not match requested provider")
    if v1_meta.get("embedding_model") != config.model:
        raise CacheError("v1 cache model does not match requested model")
    _ensure_private_dir(v2_dir)
    _ensure_private_dir(v2_dir / SEGMENTS_DIR)
    _ensure_private_dir(v2_dir / TMP_DIR)
    final_db = v2_dir / DB_NAME
    if final_db.exists():
        raise CacheError("v2 cache database already exists; choose a different target cache directory")
    embeddings = np.load(v1_embeddings_path, mmap_mode="r")
    if embeddings.ndim != 2:
        raise CacheError("v1 embeddings.npy must be a 2D matrix")
    dimension = int(embeddings.shape[1])
    if config.dimension and dimension != config.dimension:
        raise CacheError("v1 cache dimension does not match requested dimension")
    lock = FileLock(v2_dir / WRITER_LOCK_NAME)
    lock.acquire()
    _chmod_private_file(v2_dir / WRITER_LOCK_NAME)
    if final_db.exists():
        lock.release()
        raise CacheError("v2 cache database was created by another process")
    temp_db = v2_dir / TMP_DIR / f"{MIGRATE_TMP_PREFIX}{uuid.uuid4().hex}.sqlite3"
    conn: sqlite3.Connection | None = None
    files_seen: dict[str, tuple[int, int, int]] = {}
    created_segments: list[Path] = []
    rows_written = 0
    segments_written = 0
    published = False
    try:
        conn = sqlite3.connect(temp_db)
        _chmod_private_file(temp_db)
        conn.row_factory = sqlite3.Row
        _create_schema(conn)
        _set_meta(
            conn,
            {
                "schema_version": SCHEMA_VERSION,
                "embedding_provider": config.provider,
                "embedding_model": config.model,
                "embedding_dimension": dimension,
                "created_at": v1_meta.get("created_at") or _now(),
                "updated_at": _now(),
                "migrated_from_schema_version": LEGACY_SCHEMA_VERSION,
            },
        )
        with v1_chunks_path.open("r", encoding="utf-8") as handle:
            while True:
                chunk_dicts: list[dict[str, Any]] = []
                for _ in range(max(1, segment_size)):
                    line = handle.readline()
                    if not line:
                        break
                    if line.strip():
                        chunk_dicts.append(json.loads(line))
                if not chunk_dicts:
                    break
                start_row = rows_written
                end_row = start_row + len(chunk_dicts)
                if end_row > int(embeddings.shape[0]):
                    raise CacheError("v1 chunks.jsonl has more rows than embeddings.npy")
                segment_matrix = normalize_vectors(np.asarray(embeddings[start_row:end_row], dtype=np.float32))
                segment_name = f"segment-migrated-{segments_written:06d}-{uuid.uuid4().hex}.npy"
                segment_relpath = f"{SEGMENTS_DIR}/{segment_name}"
                segment_path = v2_dir / segment_relpath
                _write_npy_private(segment_path, segment_matrix)
                created_segments.append(segment_path)
                cursor = conn.execute(
                    """
                    INSERT INTO segments(path, row_count, dimension, state, size_bytes, created_at)
                    VALUES (?, ?, ?, 'active', ?, ?)
                    """,
                    (segment_relpath, len(chunk_dicts), dimension, segment_path.stat().st_size, _now()),
                )
                segment_id = int(cursor.lastrowid)
                for row_index, data in enumerate(chunk_dicts):
                    chunk = Chunk.from_dict(data)
                    state = files_seen.get(chunk.source_file)
                    if state is None:
                        manifest_record = v1_manifest.get(chunk.source_file, {})
                        state = (int(manifest_record.get("mtime_ns", 0)), -1, 0)
                    files_seen[chunk.source_file] = (state[0], state[1], state[2] + 1)
                    conn.execute(
                        """
                        INSERT INTO chunks(
                            chunk_uid, source_file, text, header, start_line, end_line,
                            metadata_json, segment_id, segment_row, active, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                        """,
                        (
                            chunk.id,
                            chunk.source_file,
                            chunk.text,
                            chunk.header,
                            int(chunk.position[0]),
                            int(chunk.position[1]),
                            json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True),
                            segment_id,
                            row_index,
                            _now(),
                        ),
                    )
                conn.commit()
                rows_written = end_row
                segments_written += 1
        if rows_written != int(embeddings.shape[0]):
            raise CacheError(f"v1 chunk count {rows_written} does not match embedding rows {embeddings.shape[0]}")
        for path, (mtime_ns, size_bytes, chunk_count) in files_seen.items():
            conn.execute(
                """
                INSERT INTO files(path, mtime_ns, size_bytes, active, chunk_count, updated_at)
                VALUES (?, ?, ?, 1, ?, ?)
                """,
                (path, mtime_ns, size_bytes, chunk_count, _now()),
            )
        _set_meta(conn, {"updated_at": _now()})
        conn.commit()
        conn.close()
        _chmod_private_file(temp_db)
        _fsync_file(temp_db)
        os.replace(temp_db, final_db)
        published = True
        _chmod_private_file(final_db)
        _fsync_dir(v2_dir)
    except BaseException:
        if conn is not None:
            conn.close()
        for path in temp_db.parent.glob(f"{temp_db.name}*"):
            path.unlink(missing_ok=True)
        if not published:
            for segment_path in created_segments:
                segment_path.unlink(missing_ok=True)
            if created_segments:
                _fsync_dir(v2_dir / SEGMENTS_DIR)
        raise
    finally:
        lock.release()
    return {
        "files_migrated": len(files_seen),
        "chunks_migrated": rows_written,
        "segments_created": segments_written,
        "v1_cache_preserved": True,
        "cache_dir": str(v2_dir),
    }


def normalize_vectors(values: np.ndarray) -> np.ndarray:
    matrix = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if matrix.ndim != 2:
        raise CacheError("embedding matrix must be 2D")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32, copy=False)


def normalize_query(value: np.ndarray) -> np.ndarray:
    vector = np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return (vector / norm).astype(np.float32, copy=False)


def file_state(path: str | Path) -> FileState:
    source = Path(path)
    try:
        if not source.exists() or not source.is_file():
            return FileState(exists=False)
        stat = source.stat()
    except OSError:
        return FileState(exists=False)
    return FileState(exists=True, mtime_ns=int(stat.st_mtime_ns), size_bytes=int(stat.st_size))


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA user_version = 2;
        CREATE TABLE IF NOT EXISTS meta(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS files(
            path TEXT PRIMARY KEY,
            mtime_ns INTEGER NOT NULL,
            size_bytes INTEGER NOT NULL,
            active INTEGER NOT NULL,
            chunk_count INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS segments(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            row_count INTEGER NOT NULL,
            dimension INTEGER NOT NULL,
            state TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_uid TEXT NOT NULL,
            source_file TEXT NOT NULL,
            text TEXT NOT NULL,
            header TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            metadata_json TEXT NOT NULL,
            segment_id INTEGER NOT NULL REFERENCES segments(id),
            segment_row INTEGER NOT NULL,
            active INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_source_active ON chunks(source_file, active);
        CREATE INDEX IF NOT EXISTS idx_chunks_segment_active ON chunks(segment_id, active, segment_row);
        CREATE INDEX IF NOT EXISTS idx_segments_state ON segments(state);
        """
    )


def _read_meta(conn: sqlite3.Connection) -> dict[str, Any]:
    try:
        rows = conn.execute("SELECT key, value FROM meta").fetchall()
    except sqlite3.OperationalError as exc:
        raise CacheError("cache database is not a semantic-search v2 SQLite cache") from exc
    return {str(row["key"]): json.loads(row["value"]) for row in rows}


def _set_meta(conn: sqlite3.Connection, values: dict[str, Any]) -> None:
    for key, value in values.items():
        conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value, ensure_ascii=False, sort_keys=True)),
        )


def _update_dimension(conn: sqlite3.Connection, dimension: int) -> None:
    meta = _read_meta(conn)
    existing = meta.get("embedding_dimension")
    if existing and int(existing) != dimension:
        raise CacheError("embedding dimension mismatch")
    _set_meta(conn, {"embedding_dimension": dimension})


def _chunk_from_row(row: sqlite3.Row) -> Chunk:
    return Chunk(
        id=str(row["chunk_uid"]),
        source_file=str(row["source_file"]),
        text=str(row["text"]),
        header=str(row["header"]),
        position=(int(row["start_line"]), int(row["end_line"])),
        metadata=json.loads(row["metadata_json"] or "{}"),
    )


def _group_rows_by_segment(rows: Sequence[sqlite3.Row]) -> dict[int, list[sqlite3.Row]]:
    grouped: dict[int, list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault(int(row["segment_id"]), []).append(row)
    return grouped


def _as_float32_matrix(values: np.ndarray, dimension: int | None) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.size == 0:
        return np.empty((0, dimension or 0), dtype=np.float32)
    if matrix.ndim != 2:
        raise CacheError("embedding matrix must be 2D")
    return matrix


def _batched(values: Sequence[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def _write_npy_private(path: Path, matrix: np.ndarray) -> None:
    _ensure_private_dir(path.parent)
    with tempfile.NamedTemporaryFile(prefix=WRITE_TMP_PREFIX, suffix=".npy", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        np.save(handle, matrix.astype(np.float32, copy=False))
        handle.flush()
        os.fsync(handle.fileno())
    _chmod_private_file(tmp)
    os.replace(tmp, path)
    _chmod_private_file(path)
    _fsync_dir(path.parent)


def _ensure_private_dir(path: Path) -> None:
    if path.is_symlink():
        raise CacheError(f"cache directory must not be a symlink: {path}")
    path.mkdir(parents=True, exist_ok=True, mode=PRIVATE_DIR_MODE)
    try:
        path.chmod(PRIVATE_DIR_MODE)
    except PermissionError:
        pass


def _chmod_private_file(path: Path) -> None:
    try:
        path.chmod(PRIVATE_FILE_MODE)
    except FileNotFoundError:
        pass
    except PermissionError:
        pass


def _chmod_sqlite_sidecars(db_path: Path) -> None:
    _chmod_private_file(db_path)
    _chmod_private_file(Path(f"{db_path}-wal"))
    _chmod_private_file(Path(f"{db_path}-shm"))


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
