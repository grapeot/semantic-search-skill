from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, TextIO

import numpy as np
from filelock import FileLock

from .models import Chunk

SCHEMA_VERSION = 1
WRITE_TMP_PREFIX = ".semantic-search-write-"
BACKUP_TMP_PREFIX = ".semantic-search-backup-"
CACHE_FILES = ("cache.json", "manifest.json", "chunks.jsonl", "embeddings.npy")


class CacheError(RuntimeError):
    pass


@dataclass(frozen=True)
class CacheConfig:
    provider: str
    model: str
    dimension: int | None = None


class SemanticIndex:
    def __init__(self, cache_dir: str | Path, config: CacheConfig) -> None:
        self.cache_dir = Path(cache_dir)
        self.config = config
        self.cache_path = self.cache_dir / "cache.json"
        self.manifest_path = self.cache_dir / "manifest.json"
        self.chunks_path = self.cache_dir / "chunks.jsonl"
        self.embeddings_path = self.cache_dir / "embeddings.npy"
        self.lock_path = self.cache_dir / "index.lock"
        self.meta: dict[str, Any] = {}
        self.manifest: dict[str, Any] = {}
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None

    def exists(self) -> bool:
        return self.cache_path.exists() or self.chunks_path.exists() or self.embeddings_path.exists()

    @contextmanager
    def lock(self, status_stream: TextIO | None = None) -> Iterator[None]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        stream = status_stream or sys.stderr
        print(f"Waiting to acquire cache lock: {self.lock_path}", file=stream, flush=True)
        with FileLock(self.lock_path):
            print(f"Acquired cache lock: {self.lock_path}", file=stream, flush=True)
            self._recover_interrupted_write()
            self._cleanup_abandoned_writes()
            yield

    def _cleanup_abandoned_writes(self) -> None:
        for prefix in (WRITE_TMP_PREFIX, BACKUP_TMP_PREFIX):
            for path in self.cache_dir.glob(f"{prefix}*"):
                if path.is_dir():
                    shutil.rmtree(path)

    @property
    def transaction_path(self) -> Path:
        return self.cache_dir / "write_transaction.json"

    def _recover_interrupted_write(self) -> None:
        if not self.transaction_path.exists():
            return
        transaction = json.loads(self.transaction_path.read_text(encoding="utf-8"))
        backup_name = transaction.get("backup_dir", "")
        if Path(backup_name).name != backup_name or not backup_name.startswith(BACKUP_TMP_PREFIX):
            raise CacheError("invalid cache write transaction backup path")
        existing_files = set(transaction.get("existing_files", []))
        if not existing_files.issubset(CACHE_FILES):
            raise CacheError("invalid cache write transaction file list")
        backup_dir = self.cache_dir / backup_name
        for name in CACHE_FILES:
            target = self.cache_dir / name
            backup = backup_dir / name
            if name in existing_files:
                if backup.exists():
                    target.unlink(missing_ok=True)
                    os.replace(backup, target)
            else:
                target.unlink(missing_ok=True)
        self.transaction_path.unlink()
        _fsync_dir(self.cache_dir)
        if backup_dir.exists():
            shutil.rmtree(backup_dir)

    def load(self, *, mmap: bool = True, require_exists: bool = False) -> None:
        if not self.cache_path.exists():
            if require_exists:
                raise CacheError(f"cache metadata missing: {self.cache_path}")
            return
        self.meta = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self._validate_meta()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8")) if self.manifest_path.exists() else {}
        self.chunks = []
        if self.chunks_path.exists():
            with self.chunks_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        self.chunks.append(Chunk.from_dict(json.loads(line)))
        if self.embeddings_path.exists():
            self.embeddings = np.load(self.embeddings_path, mmap_mode="r" if mmap else None)
        self._validate_shape()

    def _validate_meta(self) -> None:
        if self.meta.get("schema_version") != SCHEMA_VERSION:
            raise CacheError(f"unsupported cache schema: {self.meta.get('schema_version')}")
        if self.meta.get("embedding_provider") != self.config.provider:
            raise CacheError("cache provider does not match requested provider")
        if self.meta.get("embedding_model") != self.config.model:
            raise CacheError("cache model does not match requested model")
        if self.config.dimension and self.meta.get("embedding_dimension") != self.config.dimension:
            raise CacheError("cache dimension does not match requested dimension")

    def _validate_shape(self) -> None:
        if self.embeddings is None:
            if self.chunks:
                raise CacheError("chunks exist but embeddings.npy is missing")
            return
        if len(self.chunks) != int(self.embeddings.shape[0]):
            raise CacheError(f"chunk count {len(self.chunks)} does not match embedding rows {self.embeddings.shape[0]}")
        if self.config.dimension and int(self.embeddings.shape[1]) != self.config.dimension:
            raise CacheError("embedding matrix dimension mismatch")

    def changed_files(self, file_paths: list[str]) -> list[str]:
        changed: list[str] = []
        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists() or not path.is_file():
                continue
            record = self.manifest.get(file_path)
            mtime_ns = path.stat().st_mtime_ns
            if not record or int(record.get("mtime_ns", -1)) != mtime_ns:
                changed.append(file_path)
        return changed

    def replace_files(self, new_chunks: list[Chunk], new_embeddings: np.ndarray, updated_files: list[str]) -> None:
        updated = set(updated_files)
        kept_pairs: list[tuple[Chunk, np.ndarray]] = []
        if self.embeddings is not None:
            for index, chunk in enumerate(self.chunks):
                if chunk.source_file not in updated:
                    kept_pairs.append((chunk, np.asarray(self.embeddings[index], dtype=np.float32)))
        kept_chunks = [chunk for chunk, _ in kept_pairs]
        kept_embeddings = [embedding for _, embedding in kept_pairs]
        all_chunks = kept_chunks + new_chunks
        if kept_embeddings:
            embeddings = np.vstack([np.asarray(kept_embeddings, dtype=np.float32), new_embeddings]) if len(new_chunks) else np.asarray(kept_embeddings, dtype=np.float32)
        else:
            embeddings = new_embeddings.astype(np.float32, copy=False)
        self.chunks = all_chunks
        self.embeddings = embeddings
        self._rebuild_manifest()
        self.save()

    def _rebuild_manifest(self) -> None:
        manifest: dict[str, Any] = {}
        for index, chunk in enumerate(self.chunks):
            path = Path(chunk.source_file)
            record = manifest.setdefault(chunk.source_file, {"mtime_ns": path.stat().st_mtime_ns if path.exists() else 0, "chunk_indices": []})
            record["chunk_indices"].append(index)
        for record in manifest.values():
            record["chunk_count"] = len(record["chunk_indices"])
        self.manifest = manifest

    def save(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        dimension = int(self.embeddings.shape[1]) if self.embeddings is not None and self.embeddings.size else self.config.dimension
        now = datetime.now(timezone.utc).isoformat()
        created_at = self.meta.get("created_at") or now
        self.meta = {
            "schema_version": SCHEMA_VERSION,
            "embedding_provider": self.config.provider,
            "embedding_model": self.config.model,
            "embedding_dimension": dimension,
            "created_at": created_at,
            "updated_at": now,
            "chunk_count": len(self.chunks),
        }
        with tempfile.TemporaryDirectory(prefix=WRITE_TMP_PREFIX, dir=self.cache_dir) as tmp_name:
            tmp = Path(tmp_name)
            _write_json(tmp / "cache.json", self.meta)
            _write_json(tmp / "manifest.json", self.manifest)
            with (tmp / "chunks.jsonl").open("w", encoding="utf-8") as handle:
                for chunk in self.chunks:
                    handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if self.embeddings is None:
                np.save(tmp / "embeddings.npy", np.empty((0, dimension or 0), dtype=np.float32))
            else:
                np.save(tmp / "embeddings.npy", np.asarray(self.embeddings, dtype=np.float32))
            _fsync_file(tmp / "embeddings.npy")
            backup_dir = Path(tempfile.mkdtemp(prefix=BACKUP_TMP_PREFIX, dir=self.cache_dir))
            existing_files = [name for name in CACHE_FILES if (self.cache_dir / name).exists()]
            staged_transaction = tmp / "write_transaction.json"
            _write_json(
                staged_transaction,
                {"backup_dir": backup_dir.name, "existing_files": existing_files},
            )
            os.replace(staged_transaction, self.transaction_path)
            _fsync_dir(self.cache_dir)
            try:
                self._commit_staged_files(tmp, backup_dir)
                _fsync_dir(self.cache_dir)
                self.transaction_path.unlink()
                _fsync_dir(self.cache_dir)
                shutil.rmtree(backup_dir)
            except BaseException:
                self._recover_interrupted_write()
                raise
        self.load(mmap=True)

    def _commit_staged_files(self, staged_dir: Path, backup_dir: Path) -> None:
        for name in CACHE_FILES:
            target = self.cache_dir / name
            if target.exists():
                os.replace(target, backup_dir / name)
            os.replace(staged_dir / name, target)

    def subset(self, file_paths: list[str]) -> tuple[list[Chunk], np.ndarray]:
        if self.embeddings is None:
            return [], np.empty((0, 0), dtype=np.float32)
        file_set = set(file_paths)
        indices = [index for index, chunk in enumerate(self.chunks) if chunk.source_file in file_set]
        return [self.chunks[index] for index in indices], np.asarray(self.embeddings[indices], dtype=np.float32)

    def stats(self) -> dict[str, Any]:
        return {
            "cache_dir": str(self.cache_dir),
            "chunk_count": len(self.chunks),
            "file_count": len(self.manifest),
            "embedding_shape": list(self.embeddings.shape) if self.embeddings is not None else None,
            "meta": self.meta,
        }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
