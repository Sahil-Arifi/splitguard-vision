"""Offline tests for exact and approximate CPU FAISS retrieval."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from splitguard.neighbors import (
    NeighborCandidate,
    benchmark_exact_vs_hnsw,
    build_neighbor_index,
    calculate_recall_at_k,
)
from splitguard.schemas import stable_id


def image_id(name: str) -> str:
    return stable_id("img", name)


def candidate(query: str, neighbor: str, score: float) -> NeighborCandidate:
    return NeighborCandidate(
        left_id=image_id(query),
        right_id=image_id(neighbor),
        cosine_similarity=score,
    )


def test_flat_ip_search_preserves_mapping_and_filters_self() -> None:
    ids = (image_id("c"), image_id("a"), image_id("b"))
    vectors_by_id = {
        image_id("a"): np.array([1.0, 0.0], dtype=np.float32),
        image_id("b"): np.array([0.8, 0.6], dtype=np.float32),
        image_id("c"): np.array([0.0, 1.0], dtype=np.float32),
    }
    embeddings = np.stack([vectors_by_id[record_id] for record_id in ids])
    index = build_neighbor_index(ids, embeddings, kind="flat_ip", threads=1)

    results = index.search(ids, embeddings, k=2, cosine_threshold=0.1)
    by_query = {
        query_id: [item for item in results if item.left_id == query_id]
        for query_id in ids
    }

    assert index.backend == "cpu"
    assert index.kind == "flat_ip"
    assert index.record_ids == tuple(sorted(ids))
    assert index.dimension == 2
    assert index.size == 3
    assert all(item.left_id != item.right_id for item in results)
    assert [item.right_id for item in by_query[image_id("a")]] == [image_id("b")]
    assert by_query[image_id("a")][0].cosine_similarity == pytest.approx(0.8)
    assert [item.right_id for item in by_query[image_id("b")]] == [
        image_id("a"),
        image_id("c"),
    ]
    assert by_query[image_id("b")][1].cosine_similarity == pytest.approx(0.6)


def test_search_normalizes_float_inputs_and_stably_orders_ties() -> None:
    tied_ids = tuple(image_id(name) for name in ("first", "second", "third", "fourth"))
    ids = tuple(reversed(tied_ids))
    base = np.array(
        [[0.0, scale] for scale in (5.0, 10.0, 15.0, 20.0)],
        dtype=np.float64,
    )
    noncontiguous = base[:, ::-1]
    index = build_neighbor_index(ids, noncontiguous)
    query_id = image_id("external-query")

    results = index.search(
        (query_id,),
        np.array([[3.0, 0.0]], dtype=np.float64),
        k=1,
        cosine_threshold=1.0,
    )

    assert [item.right_id for item in results] == [min(tied_ids)]
    assert all(item.cosine_similarity == 1.0 for item in results)


def test_flat_alias_builds_canonical_flat_ip_index() -> None:
    index = build_neighbor_index(
        (image_id("only"),),
        np.array([[1.0, 0.0]], dtype=np.float32),
        kind="flat",
    )

    assert index.kind == "flat_ip"


def test_k_larger_than_index_filters_faiss_sentinel_rows() -> None:
    ids = tuple(image_id(name) for name in ("a", "b", "c"))
    embeddings = np.eye(3, dtype=np.float32)
    index = build_neighbor_index(ids, embeddings)

    results = index.search(ids, embeddings, k=50)

    assert len(results) == 6
    assert all(item.right_id in ids for item in results)
    assert all(item.left_id != item.right_id for item in results)


def test_empty_query_batch_returns_no_candidates() -> None:
    index = build_neighbor_index((image_id("a"),), np.array([[1.0, 0.0]], dtype=np.float32))

    assert index.search((), np.empty((0, 2), dtype=np.float32), k=1) == ()


def test_hnsw_api_returns_canonical_immutable_candidates() -> None:
    rng = np.random.default_rng(711)
    embeddings = rng.normal(size=(96, 16)).astype(np.float32)
    ids = tuple(image_id(f"item-{index:03}") for index in range(len(embeddings)))
    index = build_neighbor_index(
        reversed(ids),
        embeddings[::-1],
        kind="hnsw",
        threads=1,
        hnsw_m=16,
        hnsw_ef_construction=80,
        hnsw_ef_search=64,
    )

    results = index.search(ids[:8], embeddings[:8], k=5)

    assert index.kind == "hnsw"
    assert index.backend == "cpu"
    assert len(results) == 8 * 5
    assert results == tuple(
        sorted(
            results,
            key=lambda item: (item.left_id, -item.cosine_similarity, item.right_id),
        )
    )
    with pytest.raises(ValidationError, match="frozen"):
        results[0].cosine_similarity = 0.0


def test_recall_at_k_is_micro_recall_and_rejects_duplicate_pairs() -> None:
    exact = (
        candidate("q1", "a", 0.9),
        candidate("q1", "b", 0.8),
        candidate("q2", "c", 0.9),
        candidate("q2", "d", 0.8),
    )
    approximate = (
        candidate("q1", "a", 0.91),
        candidate("q1", "x", 0.7),
        candidate("q2", "c", 0.91),
        candidate("q2", "d", 0.79),
    )

    assert calculate_recall_at_k(exact, approximate) == pytest.approx(0.75)
    assert calculate_recall_at_k((), ()) == 1.0
    with pytest.raises(ValueError, match="unique directed pairs"):
        calculate_recall_at_k(exact, (*approximate, approximate[0]))


def test_exact_vs_hnsw_benchmark_captures_parameters_and_recall() -> None:
    rng = np.random.default_rng(991)
    embeddings = rng.normal(size=(128, 24)).astype(np.float32)
    ids = tuple(image_id(f"benchmark-{index:03}") for index in range(len(embeddings)))

    result = benchmark_exact_vs_hnsw(
        ids,
        embeddings,
        k=8,
        threads=1,
        hnsw_m=16,
        hnsw_ef_construction=80,
        hnsw_ef_search=64,
    )

    assert result.comparison == "empirical_recall_against_flat_ip"
    assert result.reference_index == "IndexFlatIP"
    assert result.approximate_index == "IndexHNSWFlat"
    assert (result.dataset_size, result.dimension, result.query_count) == (128, 24, 128)
    assert (result.k, result.threads) == (8, 1)
    assert (result.hnsw_m, result.hnsw_ef_construction, result.hnsw_ef_search) == (
        16,
        80,
        64,
    )
    assert 0.0 <= result.recall_at_k <= 1.0
    assert result.recall_at_k >= 0.95
    assert result.flat_build_seconds >= 0.0
    assert result.flat_query_seconds >= 0.0
    assert result.hnsw_build_seconds >= 0.0
    assert result.hnsw_query_seconds >= 0.0


@pytest.mark.parametrize(
    "ids,embeddings,error_type,message",
    [
        ((image_id("a"),), np.ones(2, dtype=np.float32), ValueError, "two-dimensional"),
        (
            (image_id("a"), image_id("b")),
            np.ones((1, 2), dtype=np.float32),
            ValueError,
            "row count",
        ),
        ((image_id("a"),), np.empty((1, 0), dtype=np.float32), ValueError, "dimension"),
        ((image_id("a"),), np.array([[0.0, 0.0]]), ValueError, "nonzero"),
        ((image_id("a"),), np.array([[math.nan, 0.0]]), ValueError, "finite"),
        ((image_id("a"),), np.array([[math.inf, 0.0]]), ValueError, "finite"),
        ((image_id("a"),), np.array([[1, 0]], dtype=np.int64), TypeError, "floating"),
        ((), np.empty((0, 2), dtype=np.float32), ValueError, "at least one"),
    ],
)
def test_build_rejects_invalid_embedding_inputs(
    ids: tuple[str, ...],
    embeddings: np.ndarray,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        build_neighbor_index(ids, embeddings)


def test_build_rejects_invalid_ids_and_hnsw_settings() -> None:
    valid_id = image_id("valid")
    matrix = np.array([[1.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="unique"):
        build_neighbor_index((valid_id, valid_id), np.vstack((matrix, matrix)))
    with pytest.raises(ValueError, match="canonical"):
        build_neighbor_index(("invalid",), matrix)
    with pytest.raises(ValueError, match="kind"):
        build_neighbor_index((valid_id,), matrix, kind="ivf")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least hnsw_m"):
        build_neighbor_index(
            (valid_id,),
            matrix,
            kind="hnsw",
            hnsw_m=8,
            hnsw_ef_construction=4,
        )
    with pytest.raises(TypeError, match="threads must be an integer"):
        build_neighbor_index((valid_id,), matrix, threads=True)


def test_search_rejects_invalid_query_controls_and_dimensions() -> None:
    record_id = image_id("indexed")
    index = build_neighbor_index((record_id,), np.array([[1.0, 0.0]], dtype=np.float32))

    with pytest.raises(ValueError, match="dimension does not match"):
        index.search((image_id("query"),), np.array([[1.0, 0.0, 0.0]]), k=1)
    with pytest.raises(ValueError, match="k must be at least"):
        index.search((image_id("query"),), np.array([[1.0, 0.0]]), k=0)
    with pytest.raises(TypeError, match="k must be an integer"):
        index.search((image_id("query"),), np.array([[1.0, 0.0]]), k=True)
    for threshold in (-1.1, 1.1, math.nan):
        with pytest.raises(ValueError):
            index.search(
                (image_id("query"),),
                np.array([[1.0, 0.0]]),
                k=1,
                cosine_threshold=threshold,
            )
