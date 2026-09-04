"""Local embedding orchestration and content-addressed caching."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from splitguard.models.embedder import Embedder, Float32Array, normalize_embeddings
from splitguard.schemas import ImageRecord, canonical_json

_CACHE_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UNIT_NORM_ABSOLUTE_TOLERANCE = 1e-5


class EmbeddingError(RuntimeError):
    """Base error for local embedding work."""


class EmbeddingInputError(EmbeddingError):
    """Raised when a validated record no longer matches its local source."""


class EmbeddingCacheError(EmbeddingError):
    """Raised when a cache entry cannot be written safely."""


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """ID-aligned embedding matrix and non-sensitive runtime metadata."""

    record_ids: tuple[str, ...]
    vectors: Float32Array
    model_identity: str
    preprocessing_version: str
    dimension: int
    cache_hits: int
    cache_misses: int
    deduplicated_records: int
    duration_seconds: float

    @property
    def record_count(self) -> int:
        return len(self.record_ids)

    @property
    def unique_content_count(self) -> int:
        return self.cache_hits + self.cache_misses


def embedding_cache_key(
    content_sha256: str,
    model_identity: str,
    preprocessing_version: str,
) -> str:
    """Return a full SHA-256 key without incorporating a source path."""

    if _SHA256_RE.fullmatch(content_sha256) is None:
        raise ValueError("content_sha256 must be a lowercase SHA-256 digest")
    if not model_identity.strip():
        raise ValueError("model_identity cannot be empty")
    if not preprocessing_version.strip():
        raise ValueError("preprocessing_version cannot be empty")
    payload = canonical_json(
        {
            "content_sha256": content_sha256,
            "model_identity": model_identity,
            "preprocessing_version": preprocessing_version,
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _expected_cache_metadata(
    *,
    key: str,
    content_sha256: str,
    embedder: Embedder,
) -> dict[str, str | int]:
    return {
        "cache_key": key,
        "content_sha256": content_sha256,
        "dimension": embedder.dimension,
        "dtype": "float32",
        "model_identity": embedder.model_identity,
        "preprocessing_version": embedder.preprocessing_version,
        "schema_version": _CACHE_SCHEMA_VERSION,
    }


def _is_valid_unit_vector(vector: np.ndarray[Any, Any], dimension: int) -> bool:
    if vector.shape != (dimension,) or vector.dtype != np.dtype(np.float32):
        return False
    if not np.isfinite(vector).all():
        return False
    norm = float(np.linalg.norm(vector.astype(np.float64)))
    return math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=_UNIT_NORM_ABSOLUTE_TOLERANCE)


class EmbeddingCache:
    """Atomic, content-addressed local vector cache."""

    def __init__(self, cache_dir: str | os.PathLike[str]) -> None:
        candidate = Path(cache_dir)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        self._root = Path(os.path.abspath(candidate))
        try:
            self._root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise EmbeddingCacheError("embedding cache directory could not be created") from exc
        if not self._root.is_dir():
            raise EmbeddingCacheError("embedding cache location must be a directory")

    @property
    def root(self) -> Path:
        return self._root

    def path_for_key(self, key: str) -> Path:
        if _SHA256_RE.fullmatch(key) is None:
            raise ValueError("cache key must be a lowercase SHA-256 digest")
        return self._root / key[:2] / f"{key}.npz"

    def load(self, content_sha256: str, embedder: Embedder) -> Float32Array | None:
        key = embedding_cache_key(
            content_sha256,
            embedder.model_identity,
            embedder.preprocessing_version,
        )
        path = self.path_for_key(key)
        expected_metadata = _expected_cache_metadata(
            key=key,
            content_sha256=content_sha256,
            embedder=embedder,
        )
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"metadata", "vector"}:
                    return None
                raw_metadata = archive["metadata"]
                vector = archive["vector"]
                if raw_metadata.shape != () or raw_metadata.dtype.kind not in {"U", "S"}:
                    return None
                metadata = json.loads(str(raw_metadata.item()))
                if not isinstance(metadata, dict) or metadata != expected_metadata:
                    return None
                if not _is_valid_unit_vector(vector, embedder.dimension):
                    return None
                return cast(Float32Array, np.array(vector, dtype=np.float32, copy=True))
        except (OSError, EOFError, KeyError, TypeError, ValueError, zipfile.BadZipFile):
            return None

    def store(
        self,
        content_sha256: str,
        embedder: Embedder,
        vector: Float32Array,
    ) -> None:
        key = embedding_cache_key(
            content_sha256,
            embedder.model_identity,
            embedder.preprocessing_version,
        )
        if not _is_valid_unit_vector(vector, embedder.dimension):
            raise ValueError("only finite float32 unit vectors can be cached")
        path = self.path_for_key(key)
        metadata = _expected_cache_metadata(
            key=key,
            content_sha256=content_sha256,
            embedder=embedder,
        )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{key}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                np.savez_compressed(
                    temporary,
                    metadata=np.asarray(canonical_json(metadata)),
                    vector=vector,
                )
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, path)
        except OSError as exc:
            raise EmbeddingCacheError("embedding cache entry could not be written") from exc
        finally:
            if "temporary_path" in locals():
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass


def _safe_root(dataset_root: str | os.PathLike[str]) -> Path:
    try:
        root = Path(dataset_root).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise EmbeddingInputError("dataset root could not be resolved") from exc
    if not root.is_dir():
        raise EmbeddingInputError("dataset root must be a directory")
    return root


def _read_record_bytes(root: Path, record: ImageRecord) -> bytes:
    candidate = root.joinpath(*PurePosixPath(record.path).parts)
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise EmbeddingInputError(f"record {record.id} is not an accessible file") from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise EmbeddingInputError(f"record {record.id} is outside the dataset root or not a file")
    try:
        contents = resolved.read_bytes()
    except OSError as exc:
        raise EmbeddingInputError(f"record {record.id} could not be read") from exc
    if hashlib.sha256(contents).hexdigest() != record.byte_sha256:
        raise EmbeddingInputError(f"record {record.id} content changed after validation")
    return contents


def _decode_rgb(contents: bytes, record_id: str) -> Image.Image:
    try:
        with Image.open(BytesIO(contents)) as image:
            oriented = ImageOps.exif_transpose(image)
            try:
                rgb = oriented.convert("RGB")
                rgb.load()
                return rgb
            finally:
                if oriented is not image:
                    oriented.close()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise EmbeddingInputError(f"record {record_id} could not be decoded") from exc


def embed_records(
    dataset_root: str | os.PathLike[str],
    records: Iterable[ImageRecord],
    *,
    embedder: Embedder,
    cache_dir: str | os.PathLike[str],
    batch_size: int = 32,
) -> EmbeddingResult:
    """Embed a stable record snapshot using only local files and cache storage."""

    started = time.perf_counter()
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if embedder.dimension <= 0:
        raise ValueError("embedder dimension must be positive")

    ordered_records = tuple(sorted(records, key=lambda record: record.id))
    record_ids = tuple(record.id for record in ordered_records)
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("records must have unique IDs")

    root = _safe_root(dataset_root)
    cache = EmbeddingCache(cache_dir)
    key_by_record_id: dict[str, str] = {}
    vectors_by_key: dict[str, Float32Array] = {}
    pending_contents: dict[str, tuple[str, bytes]] = {}
    cache_hits = 0
    deduplicated_records = 0

    for record in ordered_records:
        contents = _read_record_bytes(root, record)
        key = embedding_cache_key(
            record.byte_sha256,
            embedder.model_identity,
            embedder.preprocessing_version,
        )
        key_by_record_id[record.id] = key
        if key in vectors_by_key:
            deduplicated_records += 1
            continue
        if key in pending_contents:
            deduplicated_records += 1
            continue
        cached = cache.load(record.byte_sha256, embedder)
        if cached is not None:
            vectors_by_key[key] = cached
            cache_hits += 1
            continue
        pending_contents[key] = (record.byte_sha256, contents)

    pending_items = tuple(pending_contents.items())
    for offset in range(0, len(pending_items), batch_size):
        batch = pending_items[offset : offset + batch_size]
        images = [_decode_rgb(contents, key) for key, (_, contents) in batch]
        try:
            raw_matrix = embedder.embed(images)
            if not isinstance(raw_matrix, np.ndarray) or raw_matrix.dtype != np.dtype(np.float32):
                raise ValueError("embedder must return a float32 NumPy array")
            matrix = normalize_embeddings(
                raw_matrix,
                expected_rows=len(batch),
                expected_dimension=embedder.dimension,
            )
        finally:
            for image in images:
                image.close()
        for row_index, (key, (content_sha256, _)) in enumerate(batch):
            vector = cast(Float32Array, np.array(matrix[row_index], dtype=np.float32, copy=True))
            cache.store(content_sha256, embedder, vector)
            vectors_by_key[key] = vector

    if ordered_records:
        output = np.stack(
            [vectors_by_key[key_by_record_id[record.id]] for record in ordered_records]
        ).astype(np.float32, copy=False)
    else:
        output = np.empty((0, embedder.dimension), dtype=np.float32)
    output = normalize_embeddings(
        output,
        expected_rows=len(ordered_records),
        expected_dimension=embedder.dimension,
    )
    output.setflags(write=False)
    return EmbeddingResult(
        record_ids=record_ids,
        vectors=output,
        model_identity=embedder.model_identity,
        preprocessing_version=embedder.preprocessing_version,
        dimension=embedder.dimension,
        cache_hits=cache_hits,
        cache_misses=len(pending_contents),
        deduplicated_records=deduplicated_records,
        duration_seconds=time.perf_counter() - started,
    )


__all__ = [
    "EmbeddingCache",
    "EmbeddingCacheError",
    "EmbeddingError",
    "EmbeddingInputError",
    "EmbeddingResult",
    "embed_records",
    "embedding_cache_key",
]
