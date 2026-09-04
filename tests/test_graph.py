from __future__ import annotations

from collections.abc import Iterable

import pytest

from splitguard.config import PolicyConfig
from splitguard.graph import (
    EvidenceGraph,
    EvidencePolicy,
    GraphInputError,
    build_evidence_graph,
)
from splitguard.hashing import (
    ExactDuplicateCluster,
    PhashCandidatePair,
    group_exact_duplicates,
)
from splitguard.neighbors import NeighborCandidate
from splitguard.schemas import (
    DuplicateClassification,
    EdgeDecision,
    ImageRecord,
    Split,
    stable_id,
)


def record(path: str, *, sha: str | None = None) -> ImageRecord:
    content_sha = sha if sha is not None else stable_id("sha", path).split("_", maxsplit=1)[1]
    content_sha = (content_sha * 4)[:64]
    return ImageRecord(
        id=stable_id("img", path),
        path=path,
        split=Split.TRAIN,
        label="cat",
        byte_sha256=content_sha,
        byte_size=10,
        width=4,
        height=4,
        format="png",
    )


def exact_clusters(records: Iterable[ImageRecord]) -> tuple[ExactDuplicateCluster, ...]:
    return group_exact_duplicates(records)


def pair(left: ImageRecord, right: ImageRecord, distance: int) -> PhashCandidatePair:
    left_id, right_id = sorted((left.id, right.id))
    return PhashCandidatePair(left_id=left_id, right_id=right_id, distance=distance)


def neighbor(left: ImageRecord, right: ImageRecord, score: float) -> NeighborCandidate:
    return NeighborCandidate(
        left_id=left.id,
        right_id=right.id,
        cosine_similarity=score,
    )


def build(
    records: Iterable[ImageRecord],
    *,
    exact: Iterable[ExactDuplicateCluster] = (),
    phash: Iterable[PhashCandidatePair] = (),
    neighbors: Iterable[NeighborCandidate] = (),
    policy: EvidencePolicy | PolicyConfig | None = None,
    phash_threshold: int = 8,
    cosine_threshold: float = 0.9,
) -> EvidenceGraph:
    return build_evidence_graph(
        records,
        exact,
        phash,
        neighbors,
        phash_threshold=phash_threshold,
        cosine_threshold=cosine_threshold,
        policy=policy,
    )


def test_merges_bidirectional_evidence_with_exact_precedence() -> None:
    duplicate_sha = "a" * 64
    left = record("train/cat/a.png", sha=duplicate_sha)
    right = record("test/cat/b.png", sha=duplicate_sha)

    graph = build(
        (right, left),
        exact=exact_clusters((left, right)),
        phash=(pair(left, right, 7), pair(left, right, 3)),
        neighbors=(neighbor(left, right, 0.91), neighbor(right, left, 0.97)),
    )

    assert len(graph.edges) == 1
    edge = graph.edges[0]
    assert edge.evidence.exact_match is True
    assert edge.evidence.phash_distance == 3
    assert edge.evidence.cosine_similarity == pytest.approx(0.97)
    assert edge.classification is DuplicateClassification.EXACT
    assert edge.decision is EdgeDecision.DEFINITE
    assert edge.confidence == 1.0
    assert graph.review_edges == ()
    assert graph.families[0].edge_count == 1


def test_thresholds_are_inclusive_and_rejected_evidence_is_not_preserved() -> None:
    first = record("train/cat/a.png")
    second = record("val/cat/b.png")
    third = record("test/cat/c.png")

    graph = build(
        (first, second, third),
        phash=(pair(first, second, 8), pair(first, third, 9)),
        neighbors=(
            neighbor(second, first, 0.9),
            neighbor(first, third, 0.89),
        ),
    )

    assert len(graph.edges) == 1
    edge = graph.edges[0]
    assert {edge.left_id, edge.right_id} == {first.id, second.id}
    assert edge.evidence.phash_distance == 8
    assert edge.evidence.cosine_similarity == 0.9
    assert edge.classification is DuplicateClassification.TRANSFORMED_DUPLICATE
    assert edge.confidence == pytest.approx(1.0 - 8 / 64)


def test_semantic_only_edges_are_review_only_and_do_not_form_families() -> None:
    left = record("train/cat/a.png")
    right = record("test/cat/b.png")

    graph = build(
        (left, right),
        neighbors=(neighbor(left, right, 0.96), neighbor(right, left, 0.95)),
    )

    assert len(graph.edges) == 1
    edge = graph.edges[0]
    assert edge.classification is DuplicateClassification.SEMANTIC_CANDIDATE
    assert edge.decision is EdgeDecision.REVIEW
    assert edge.confidence == pytest.approx((0.96 + 1.0) / 2.0)
    assert graph.review_edges == (edge,)
    assert graph.families == ()


def test_definite_edges_build_transitive_family_without_inventing_edge() -> None:
    first = record("train/cat/a.png")
    second = record("val/cat/b.png")
    third = record("test/cat/c.png")

    graph = build(
        (third, first, second),
        phash=(pair(first, second, 4), pair(second, third, 5)),
    )

    assert {(edge.left_id, edge.right_id) for edge in graph.edges} == {
        tuple(sorted((first.id, second.id))),
        tuple(sorted((second.id, third.id))),
    }
    assert len(graph.families) == 1
    assert graph.families[0].member_ids == tuple(sorted((first.id, second.id, third.id)))
    assert graph.families[0].edge_count == 2


def test_input_permutations_produce_identical_graph() -> None:
    records = (
        record("train/cat/a.png"),
        record("val/cat/b.png"),
        record("test/cat/c.png"),
    )
    phash = (pair(records[0], records[1], 4), pair(records[1], records[2], 5))
    neighbors = (
        neighbor(records[0], records[2], 0.94),
        neighbor(records[2], records[0], 0.93),
    )

    forward = build(records, phash=phash, neighbors=neighbors)
    reversed_graph = build(
        reversed(records),
        phash=reversed(phash),
        neighbors=reversed(neighbors),
    )

    assert forward == reversed_graph


@pytest.mark.parametrize("source", ["phash", "neighbor"])
def test_candidates_referencing_unknown_ids_are_rejected(source: str) -> None:
    known = record("train/cat/a.png")
    unknown = record("test/cat/missing.png")

    with pytest.raises(GraphInputError, match="unknown record ID"):
        if source == "phash":
            build((known,), phash=(pair(known, unknown, 2),))
        else:
            build((known,), neighbors=(neighbor(known, unknown, 0.99),))


def test_exact_cluster_membership_must_match_record_content_hashes() -> None:
    duplicate_sha = "b" * 64
    first = record("train/cat/a.png", sha=duplicate_sha)
    second = record("test/cat/b.png", sha=duplicate_sha)
    cluster = exact_clusters((first, second))[0]
    changed_second = second.model_copy(update={"byte_sha256": "c" * 64})

    with pytest.raises(GraphInputError, match="membership"):
        build((first, changed_second), exact=(cluster,))


def test_candidate_between_non_anchor_exact_members_retains_exact_evidence() -> None:
    duplicate_sha = "d" * 64
    records = tuple(
        record(path, sha=duplicate_sha)
        for path in ("train/cat/a.png", "val/cat/b.png", "test/cat/c.png")
    )
    ordered = tuple(sorted(records, key=lambda item: item.id))

    graph = build(
        records,
        exact=exact_clusters(records),
        neighbors=(neighbor(ordered[1], ordered[2], 0.99),),
    )

    matching = next(
        edge
        for edge in graph.edges
        if {edge.left_id, edge.right_id} == {ordered[1].id, ordered[2].id}
    )
    assert matching.evidence.exact_match is True
    assert matching.evidence.cosine_similarity == 0.99
    assert matching.classification is DuplicateClassification.EXACT
    assert len(graph.edges) == 3


def test_policy_toggles_control_decisions_and_hard_grouping() -> None:
    exact_sha = "e" * 64
    exact_left = record("train/cat/exact-a.png", sha=exact_sha)
    exact_right = record("test/cat/exact-b.png", sha=exact_sha)
    phash_left = record("train/cat/phash-a.png")
    phash_right = record("test/cat/phash-b.png")
    semantic_left = record("train/cat/semantic-a.png")
    semantic_right = record("test/cat/semantic-b.png")
    records = (
        exact_left,
        exact_right,
        phash_left,
        phash_right,
        semantic_left,
        semantic_right,
    )
    policy = EvidencePolicy(
        exact_is_duplicate=False,
        phash_is_duplicate=False,
        embedding_only_requires_review=False,
    )

    graph = build(
        records,
        exact=exact_clusters((exact_left, exact_right)),
        phash=(pair(phash_left, phash_right, 1),),
        neighbors=(neighbor(semantic_left, semantic_right, 0.99),),
        policy=policy,
    )

    decisions = {edge.classification: edge.decision for edge in graph.edges}
    assert decisions == {
        DuplicateClassification.EXACT: EdgeDecision.REVIEW,
        DuplicateClassification.TRANSFORMED_DUPLICATE: EdgeDecision.REVIEW,
        DuplicateClassification.SEMANTIC_CANDIDATE: EdgeDecision.DEFINITE,
    }
    assert len(graph.review_edges) == 2
    assert len(graph.families) == 1
    assert graph.families[0].member_ids == tuple(sorted((semantic_left.id, semantic_right.id)))


def test_policy_config_is_accepted_without_mutating_decision_semantics() -> None:
    left = record("train/cat/a.png")
    right = record("test/cat/b.png")

    graph = build(
        (left, right),
        phash=(pair(left, right, 3),),
        policy=PolicyConfig(phash_is_duplicate=False),
    )

    assert graph.policy == EvidencePolicy(phash_is_duplicate=False)
    assert graph.edges[0].decision is EdgeDecision.REVIEW


@pytest.mark.parametrize(
    "phash_threshold,cosine_threshold,error",
    [
        (True, 0.9, TypeError),
        (65, 0.9, ValueError),
        (8, float("nan"), ValueError),
        (8, 1.1, ValueError),
    ],
)
def test_thresholds_are_validated_strictly(
    phash_threshold: int,
    cosine_threshold: float,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        build(
            (),
            phash_threshold=phash_threshold,
            cosine_threshold=cosine_threshold,
        )
