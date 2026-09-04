"""Deterministic CPU FAISS indexes and exact-versus-approximate evaluation."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from time import perf_counter
from typing import Annotated, Literal, Self

import faiss
import numpy as np
from pydantic import Field, model_validator

from splitguard.schemas import StableId, StrictFrozenModel

NeighborIndexKind = Literal["flat_ip", "hnsw"]
NeighborIndexRequestKind = Literal["flat", "flat_ip", "hnsw"]

_STABLE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[0-9a-f]{16,64}$")
_MAX_FAISS_THREADS = 1024


class NeighborCandidate(StrictFrozenModel):
    """One directed query-to-index neighbor with cosine-similarity evidence."""

    left_id: StableId
    right_id: StableId
    cosine_similarity: Annotated[float, Field(ge=-1.0, le=1.0)]

    @model_validator(mode="after")
    def excludes_self(self) -> Self:
        if self.left_id == self.right_id:
            raise ValueError("a neighbor candidate cannot reference itself")
        return self


class NeighborBenchmarkResult(StrictFrozenModel):
    """Measured HNSW recall and latency against an exact FlatIP reference.

    ``recall_at_k`` is empirical for this dataset and configuration. It does
    not assert that approximate HNSW search is equivalent to exact search.
    """

    comparison: Literal["empirical_recall_against_flat_ip"] = (
        "empirical_recall_against_flat_ip"
    )
    reference_index: Literal["IndexFlatIP"] = "IndexFlatIP"
    approximate_index: Literal["IndexHNSWFlat"] = "IndexHNSWFlat"
    dataset_size: Annotated[int, Field(gt=0)]
    dimension: Annotated[int, Field(gt=0)]
    query_count: Annotated[int, Field(gt=0)]
    k: Annotated[int, Field(gt=0)]
    threads: Annotated[int, Field(gt=0)]
    hnsw_m: Annotated[int, Field(ge=2)]
    hnsw_ef_construction: Annotated[int, Field(ge=2)]
    hnsw_ef_search: Annotated[int, Field(ge=1)]
    flat_build_seconds: Annotated[float, Field(ge=0.0)]
    flat_query_seconds: Annotated[float, Field(ge=0.0)]
    hnsw_build_seconds: Annotated[float, Field(ge=0.0)]
    hnsw_query_seconds: Annotated[float, Field(ge=0.0)]
    recall_at_k: Annotated[float, Field(ge=0.0, le=1.0)]


def _validate_positive_integer(value: int, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _set_faiss_threads(threads: int) -> int:
    threads = _validate_positive_integer(threads, "threads")
    if threads > _MAX_FAISS_THREADS:
        raise ValueError(f"threads cannot exceed {_MAX_FAISS_THREADS}")
    faiss.omp_set_num_threads(threads)
    return threads


def _validate_threshold(cosine_threshold: float) -> float:
    if isinstance(cosine_threshold, bool) or not isinstance(
        cosine_threshold, (int, float)
    ):
        raise TypeError("cosine_threshold must be a number")
    threshold = float(cosine_threshold)
    if not math.isfinite(threshold):
        raise ValueError("cosine_threshold must be finite")
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("cosine_threshold must be between -1 and 1")
    return threshold


def _validate_ids(record_ids: Iterable[str], *, allow_empty: bool) -> tuple[str, ...]:
    if isinstance(record_ids, (str, bytes)):
        raise TypeError("record_ids must be an iterable of canonical IDs")
    ids = tuple(record_ids)
    if not allow_empty and not ids:
        raise ValueError("at least one record_id is required")
    for record_id in ids:
        if not isinstance(record_id, str):
            raise TypeError("every record_id must be a string")
        if _STABLE_ID_RE.fullmatch(record_id) is None:
            raise ValueError("every record_id must be a canonical stable identifier")
    if len(set(ids)) != len(ids):
        raise ValueError("record_ids must be unique")
    return ids


def _normalize_embeddings(
    embeddings: np.ndarray,
    *,
    expected_rows: int,
    expected_dimension: int | None = None,
    allow_empty: bool = False,
) -> np.ndarray:
    if not isinstance(embeddings, np.ndarray):
        raise TypeError("embeddings must be a NumPy array")
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional matrix")
    if embeddings.shape[0] != expected_rows:
        raise ValueError("embedding row count must match record_ids")
    if embeddings.shape[1] == 0:
        raise ValueError("embedding dimension must be positive")
    if expected_dimension is not None and embeddings.shape[1] != expected_dimension:
        raise ValueError("query embedding dimension does not match the index")
    if not allow_empty and embeddings.shape[0] == 0:
        raise ValueError("at least one embedding row is required")
    if not np.issubdtype(embeddings.dtype, np.floating):
        raise TypeError("embeddings must have a floating-point dtype")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings must contain only finite values")
    if embeddings.shape[0] == 0:
        return np.ascontiguousarray(embeddings, dtype=np.float32)

    working = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(working, axis=1)
    if not np.isfinite(norms).all():
        raise ValueError("embedding norms must be finite")
    if np.any(norms == 0.0):
        raise ValueError("embedding rows must be nonzero")
    normalized = working / norms[:, None]
    matrix = np.ascontiguousarray(normalized, dtype=np.float32)

    # Normalize once more after the float32 conversion so FAISS always receives
    # contiguous unit vectors in the exact dtype it indexes.
    float32_norms = np.linalg.norm(matrix, axis=1)
    if np.any(float32_norms == 0.0) or not np.isfinite(float32_norms).all():
        raise ValueError("embeddings cannot be represented as finite float32 unit vectors")
    matrix /= float32_norms[:, None]
    return np.ascontiguousarray(matrix, dtype=np.float32)


def _canonical_rows(
    record_ids: Iterable[str],
    embeddings: np.ndarray,
    *,
    expected_dimension: int | None = None,
    allow_empty: bool = False,
) -> tuple[tuple[str, ...], np.ndarray]:
    ids = _validate_ids(record_ids, allow_empty=allow_empty)
    matrix = _normalize_embeddings(
        embeddings,
        expected_rows=len(ids),
        expected_dimension=expected_dimension,
        allow_empty=allow_empty,
    )
    order = sorted(range(len(ids)), key=ids.__getitem__)
    canonical_ids = tuple(ids[index] for index in order)
    return canonical_ids, np.ascontiguousarray(matrix[order], dtype=np.float32)


class FaissNeighborIndex:
    """A CPU FAISS index with stable canonical ID-to-row mapping."""

    def __init__(
        self,
        *,
        index: faiss.Index,
        record_ids: tuple[str, ...],
        kind: NeighborIndexKind,
        dimension: int,
        threads: int,
        hnsw_m: int | None,
        hnsw_ef_construction: int | None,
        hnsw_ef_search: int | None,
    ) -> None:
        self._index = index
        self._record_ids = record_ids
        self._kind = kind
        self._dimension = dimension
        self._threads = threads
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construction = hnsw_ef_construction
        self._hnsw_ef_search = hnsw_ef_search

    @property
    def record_ids(self) -> tuple[str, ...]:
        return self._record_ids

    @property
    def kind(self) -> NeighborIndexKind:
        return self._kind

    @property
    def backend(self) -> Literal["cpu"]:
        return "cpu"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def size(self) -> int:
        return len(self._record_ids)

    @property
    def threads(self) -> int:
        return self._threads

    def search(
        self,
        query_ids: Iterable[str],
        query_embeddings: np.ndarray,
        *,
        k: int,
        cosine_threshold: float = -1.0,
    ) -> tuple[NeighborCandidate, ...]:
        """Return at most ``k`` non-self neighbors for each query.

        Results are ordered by query ID, descending cosine similarity, and
        neighbor ID. Ties within the retrieved candidate set are therefore
        stable even when FAISS returns equal scores in another order.
        """

        k = _validate_positive_integer(k, "k")
        threshold = _validate_threshold(cosine_threshold)
        canonical_query_ids, canonical_queries = _canonical_rows(
            query_ids,
            query_embeddings,
            expected_dimension=self.dimension,
            allow_empty=True,
        )
        if not canonical_query_ids:
            return ()

        _set_faiss_threads(self.threads)
        # The extra slot compensates for an indexed query finding itself. When
        # k exceeds the index size, one extra slot also exercises FAISS's -1
        # sentinel, which is filtered below without unbounded allocation.
        search_k = min(k + 1, self.size + 1)
        scores, row_indices = self._index.search(canonical_queries, search_k)
        while search_k < self.size and self._has_unresolved_cutoff_tie(
            canonical_query_ids,
            scores,
            row_indices,
            k,
        ):
            # Equal similarities at the requested cutoff can otherwise let
            # FAISS's internal tie order choose a non-canonical ID. Expansion
            # is limited to actual cutoff ties and eventually retrieves all
            # rows when that is the only way to resolve them deterministically.
            search_k = min(self.size, max(search_k + 1, search_k * 2))
            scores, row_indices = self._index.search(canonical_queries, search_k)

        output: list[NeighborCandidate] = []
        for query_row, query_id in enumerate(canonical_query_ids):
            candidates_by_id: dict[str, NeighborCandidate] = {}
            for raw_score, raw_row_index in zip(
                scores[query_row], row_indices[query_row], strict=True
            ):
                row_index = int(raw_row_index)
                if row_index == -1:
                    continue
                if not 0 <= row_index < self.size:
                    raise RuntimeError("FAISS returned an invalid row index")
                right_id = self.record_ids[row_index]
                if right_id == query_id:
                    continue
                score = float(raw_score)
                if not math.isfinite(score):
                    raise RuntimeError("FAISS returned a non-finite similarity")
                score = min(1.0, max(-1.0, score))
                if score < threshold:
                    continue
                candidate = NeighborCandidate(
                    left_id=query_id,
                    right_id=right_id,
                    cosine_similarity=score,
                )
                previous = candidates_by_id.get(right_id)
                if previous is None or score > previous.cosine_similarity:
                    candidates_by_id[right_id] = candidate

            ranked = sorted(
                candidates_by_id.values(),
                key=lambda candidate: (-candidate.cosine_similarity, candidate.right_id),
            )
            output.extend(ranked[:k])

        return tuple(
            sorted(
                output,
                key=lambda candidate: (
                    candidate.left_id,
                    -candidate.cosine_similarity,
                    candidate.right_id,
                ),
            )
        )

    def _has_unresolved_cutoff_tie(
        self,
        query_ids: tuple[str, ...],
        scores: np.ndarray,
        row_indices: np.ndarray,
        k: int,
    ) -> bool:
        for query_row, query_id in enumerate(query_ids):
            retrieved_scores: list[float] = []
            for raw_score, raw_row_index in zip(
                scores[query_row], row_indices[query_row], strict=True
            ):
                row_index = int(raw_row_index)
                if row_index == -1:
                    continue
                if not 0 <= row_index < self.size:
                    raise RuntimeError("FAISS returned an invalid row index")
                if self.record_ids[row_index] != query_id:
                    retrieved_scores.append(float(raw_score))
            retrieved_scores.sort(reverse=True)
            if len(retrieved_scores) < k:
                return True
            if retrieved_scores[k - 1] == retrieved_scores[-1]:
                return True
        return False


def build_neighbor_index(
    record_ids: Iterable[str],
    embeddings: np.ndarray,
    *,
    kind: NeighborIndexRequestKind = "flat_ip",
    threads: int = 1,
    hnsw_m: int = 32,
    hnsw_ef_construction: int = 200,
    hnsw_ef_search: int = 64,
) -> FaissNeighborIndex:
    """Build a deterministic CPU FlatIP or HNSW inner-product index."""

    if kind not in {"flat", "flat_ip", "hnsw"}:
        raise ValueError("kind must be 'flat', 'flat_ip', or 'hnsw'")
    canonical_kind: NeighborIndexKind = "flat_ip" if kind == "flat" else kind
    threads = _set_faiss_threads(threads)
    canonical_ids, matrix = _canonical_rows(record_ids, embeddings)
    dimension = int(matrix.shape[1])

    if canonical_kind == "flat_ip":
        index: faiss.Index = faiss.IndexFlatIP(dimension)
        configured_m: int | None = None
        configured_ef_construction: int | None = None
        configured_ef_search: int | None = None
    else:
        hnsw_m = _validate_positive_integer(hnsw_m, "hnsw_m", minimum=2)
        hnsw_ef_construction = _validate_positive_integer(
            hnsw_ef_construction,
            "hnsw_ef_construction",
            minimum=2,
        )
        hnsw_ef_search = _validate_positive_integer(hnsw_ef_search, "hnsw_ef_search")
        if hnsw_ef_construction < hnsw_m:
            raise ValueError("hnsw_ef_construction must be at least hnsw_m")
        hnsw_index = faiss.IndexHNSWFlat(dimension, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        hnsw_index.hnsw.efConstruction = hnsw_ef_construction
        hnsw_index.hnsw.efSearch = hnsw_ef_search
        index = hnsw_index
        configured_m = hnsw_m
        configured_ef_construction = hnsw_ef_construction
        configured_ef_search = hnsw_ef_search

    index.add(matrix)
    return FaissNeighborIndex(
        index=index,
        record_ids=canonical_ids,
        kind=canonical_kind,
        dimension=dimension,
        threads=threads,
        hnsw_m=configured_m,
        hnsw_ef_construction=configured_ef_construction,
        hnsw_ef_search=configured_ef_search,
    )


def calculate_recall_at_k(
    exact_candidates: Iterable[NeighborCandidate],
    approximate_candidates: Iterable[NeighborCandidate],
) -> float:
    """Return micro recall of approximate directed neighbors against exact ones."""

    def candidate_sets(
        candidates: Iterable[NeighborCandidate],
    ) -> dict[str, set[str]]:
        by_query: dict[str, set[str]] = {}
        seen_pairs: set[tuple[str, str]] = set()
        for candidate in candidates:
            pair = (candidate.left_id, candidate.right_id)
            if pair in seen_pairs:
                raise ValueError("neighbor candidates must contain unique directed pairs")
            seen_pairs.add(pair)
            by_query.setdefault(candidate.left_id, set()).add(candidate.right_id)
        return by_query

    exact_by_query = candidate_sets(exact_candidates)
    approximate_by_query = candidate_sets(approximate_candidates)
    expected_count = sum(len(neighbor_ids) for neighbor_ids in exact_by_query.values())
    if expected_count == 0:
        return 1.0
    recovered_count = sum(
        len(exact_ids & approximate_by_query.get(query_id, set()))
        for query_id, exact_ids in exact_by_query.items()
    )
    return recovered_count / expected_count


def benchmark_exact_vs_hnsw(
    record_ids: Iterable[str],
    embeddings: np.ndarray,
    *,
    k: int,
    threads: int = 1,
    hnsw_m: int = 32,
    hnsw_ef_construction: int = 200,
    hnsw_ef_search: int = 64,
) -> NeighborBenchmarkResult:
    """Measure self-query HNSW recall and latency against exact FlatIP."""

    ids = tuple(record_ids)
    k = _validate_positive_integer(k, "k")

    started = perf_counter()
    flat = build_neighbor_index(ids, embeddings, kind="flat_ip", threads=threads)
    flat_build_seconds = perf_counter() - started

    started = perf_counter()
    exact_candidates = flat.search(ids, embeddings, k=k)
    flat_query_seconds = perf_counter() - started

    started = perf_counter()
    hnsw = build_neighbor_index(
        ids,
        embeddings,
        kind="hnsw",
        threads=threads,
        hnsw_m=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
        hnsw_ef_search=hnsw_ef_search,
    )
    hnsw_build_seconds = perf_counter() - started

    started = perf_counter()
    approximate_candidates = hnsw.search(ids, embeddings, k=k)
    hnsw_query_seconds = perf_counter() - started

    return NeighborBenchmarkResult(
        dataset_size=flat.size,
        dimension=flat.dimension,
        query_count=len(ids),
        k=k,
        threads=threads,
        hnsw_m=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
        hnsw_ef_search=hnsw_ef_search,
        flat_build_seconds=flat_build_seconds,
        flat_query_seconds=flat_query_seconds,
        hnsw_build_seconds=hnsw_build_seconds,
        hnsw_query_seconds=hnsw_query_seconds,
        recall_at_k=calculate_recall_at_k(exact_candidates, approximate_candidates),
    )


__all__ = [
    "FaissNeighborIndex",
    "NeighborBenchmarkResult",
    "NeighborCandidate",
    "NeighborIndexKind",
    "NeighborIndexRequestKind",
    "benchmark_exact_vs_hnsw",
    "build_neighbor_index",
    "calculate_recall_at_k",
]
