"""Tests for definite and semantic cross-label conflict analysis."""

from __future__ import annotations

import hashlib

from splitguard.conflicts import analyze_conflicts
from splitguard.schemas import (
    ConflictKind,
    DuplicateClassification,
    DuplicateEdge,
    DuplicateEvidence,
    DuplicateFamily,
    EdgeDecision,
    ImageRecord,
    Split,
    family_id_for,
    stable_id,
)


def make_record(name: str, label: str | None) -> ImageRecord:
    return ImageRecord(
        id=stable_id("image", name),
        path=f"train/{name}.png",
        split=Split.TRAIN,
        label=label,
        byte_sha256=hashlib.sha256(name.encode()).hexdigest(),
        byte_size=10,
        width=4,
        height=4,
        format="png",
    )


def make_family(records: tuple[ImageRecord, ...], edge_count: int) -> DuplicateFamily:
    member_ids = tuple(sorted(record.id for record in records))
    return DuplicateFamily(
        family_id=family_id_for(member_ids),
        member_ids=member_ids,
        edge_count=edge_count,
    )


def make_edge(
    left: ImageRecord,
    right: ImageRecord,
    classification: DuplicateClassification,
    *,
    decision: EdgeDecision = EdgeDecision.DEFINITE,
    similarity: float = 0.98,
    semantic_with_phash: bool = False,
) -> DuplicateEdge:
    left_id, right_id = sorted((left.id, right.id))
    if classification is DuplicateClassification.EXACT:
        evidence = DuplicateEvidence(
            exact_match=True,
            phash_distance=0,
            cosine_similarity=1.0,
        )
    elif classification is DuplicateClassification.TRANSFORMED_DUPLICATE:
        evidence = DuplicateEvidence(phash_distance=3, cosine_similarity=similarity)
    else:
        evidence = DuplicateEvidence(
            phash_distance=12 if semantic_with_phash else None,
            cosine_similarity=similarity,
        )
    return DuplicateEdge(
        left_id=left_id,
        right_id=right_id,
        evidence=evidence,
        classification=classification,
        decision=decision,
        confidence=similarity,
    )


def test_exact_near_and_semantic_cross_label_conflicts_are_distinguished() -> None:
    exact_cat = make_record("exact-cat", "cat")
    exact_dog = make_record("exact-dog", "dog")
    near_cat = make_record("near-cat", "cat")
    near_dog = make_record("near-dog", "dog")
    semantic_cat = make_record("semantic-cat", "cat")
    semantic_dog = make_record("semantic-dog", "dog")
    exact_family = make_family((exact_cat, exact_dog), edge_count=1)
    near_family = make_family((near_cat, near_dog), edge_count=1)
    exact_edge = make_edge(exact_cat, exact_dog, DuplicateClassification.EXACT)
    near_edge = make_edge(
        near_cat,
        near_dog,
        DuplicateClassification.TRANSFORMED_DUPLICATE,
    )
    semantic_edge = make_edge(
        semantic_cat,
        semantic_dog,
        DuplicateClassification.SEMANTIC_CANDIDATE,
        decision=EdgeDecision.REVIEW,
    )

    result = analyze_conflicts(
        (
            semantic_dog,
            exact_cat,
            near_dog,
            semantic_cat,
            exact_dog,
            near_cat,
        ),
        (near_family, exact_family),
        (near_edge, exact_edge),
        review_edges=(semantic_edge,),
    )

    assert tuple(conflict.kind for conflict in result.conflicts) == (
        ConflictKind.EXACT_DUPLICATE,
        ConflictKind.NEAR_DUPLICATE,
        ConflictKind.SEMANTIC_CANDIDATE,
    )
    assert result.cross_label_conflict_count == 3
    assert result.definite_conflict_count == 2
    assert result.exact_conflict_count == 1
    assert result.near_conflict_count == 1
    assert result.semantic_review_conflict_count == 1


def test_family_strength_uses_direct_cross_label_evidence_only() -> None:
    cat_a = make_record("cat-a", "cat")
    cat_b = make_record("cat-b", "cat")
    dog = make_record("dog", "dog")
    family = make_family((cat_a, cat_b, dog), edge_count=2)
    same_label_exact = make_edge(cat_a, cat_b, DuplicateClassification.EXACT)
    cross_label_near = make_edge(
        cat_b,
        dog,
        DuplicateClassification.TRANSFORMED_DUPLICATE,
    )

    result = analyze_conflicts(
        (dog, cat_a, cat_b),
        (family,),
        (same_label_exact, cross_label_near),
    )

    assert len(result.conflicts) == 1
    assert result.conflicts[0].kind is ConflictKind.NEAR_DUPLICATE
    assert result.conflicts[0].member_ids == family.member_ids
    assert result.conflicts[0].labels == ("cat", "dog")


def test_exact_direct_cross_label_evidence_takes_precedence() -> None:
    cat = make_record("priority-cat", "cat")
    dog_a = make_record("priority-dog-a", "dog")
    dog_b = make_record("priority-dog-b", "dog")
    family = make_family((cat, dog_a, dog_b), edge_count=2)
    transformed = make_edge(
        cat,
        dog_a,
        DuplicateClassification.TRANSFORMED_DUPLICATE,
    )
    exact = make_edge(cat, dog_b, DuplicateClassification.EXACT)

    result = analyze_conflicts((dog_b, cat, dog_a), (family,), (transformed, exact))

    assert result.conflicts[0].kind is ConflictKind.EXACT_DUPLICATE
    assert result.exact_conflict_count == 1
    assert result.near_conflict_count == 0


def test_semantic_pair_already_represented_by_definite_family_is_not_repeated() -> None:
    cat = make_record("represented-cat", "cat")
    dog = make_record("represented-dog", "dog")
    family = make_family((cat, dog), edge_count=1)
    near = make_edge(cat, dog, DuplicateClassification.TRANSFORMED_DUPLICATE)
    semantic = make_edge(
        cat,
        dog,
        DuplicateClassification.SEMANTIC_CANDIDATE,
        decision=EdgeDecision.REVIEW,
    )

    result = analyze_conflicts((cat, dog), (family,), (near,), review_edges=(semantic,))

    assert len(result.conflicts) == 1
    assert result.conflicts[0].kind is ConflictKind.NEAR_DUPLICATE
    assert result.semantic_review_conflict_count == 0


def test_semantic_pair_with_direct_definite_evidence_is_not_repeated_without_family() -> None:
    cat = make_record("direct-cat", "cat")
    dog = make_record("direct-dog", "dog")
    exact = make_edge(cat, dog, DuplicateClassification.EXACT)
    semantic = make_edge(
        cat,
        dog,
        DuplicateClassification.SEMANTIC_CANDIDATE,
        decision=EdgeDecision.REVIEW,
    )

    result = analyze_conflicts((dog, cat), (), (semantic, exact))

    assert result.conflicts == ()
    assert result.semantic_review_conflict_count == 0


def test_semantic_review_requires_different_known_labels_and_semantic_only_evidence() -> None:
    cat_a = make_record("same-cat-a", "cat")
    cat_b = make_record("same-cat-b", "cat")
    dog = make_record("phash-dog", "dog")
    unlabeled = make_record("unlabeled", None)
    same_label = make_edge(
        cat_a,
        cat_b,
        DuplicateClassification.SEMANTIC_CANDIDATE,
        decision=EdgeDecision.REVIEW,
    )
    unknown_label = make_edge(
        dog,
        unlabeled,
        DuplicateClassification.SEMANTIC_CANDIDATE,
        decision=EdgeDecision.REVIEW,
    )
    has_phash = make_edge(
        cat_a,
        dog,
        DuplicateClassification.SEMANTIC_CANDIDATE,
        decision=EdgeDecision.REVIEW,
        semantic_with_phash=True,
    )

    result = analyze_conflicts(
        (unlabeled, dog, cat_b, cat_a),
        (),
        (),
        review_edges=(same_label, unknown_label, has_phash),
    )

    assert result.conflicts == ()
    assert result.cross_label_conflict_count == 0


def test_semantic_reviews_are_deduplicated_and_input_order_independent() -> None:
    cat = make_record("dedupe-cat", "cat")
    dog = make_record("dedupe-dog", "dog")
    lower = make_edge(
        cat,
        dog,
        DuplicateClassification.SEMANTIC_CANDIDATE,
        decision=EdgeDecision.REVIEW,
        similarity=0.90,
    )
    higher = make_edge(
        cat,
        dog,
        DuplicateClassification.SEMANTIC_CANDIDATE,
        decision=EdgeDecision.REVIEW,
        similarity=0.96,
    )

    forward = analyze_conflicts(
        (cat, dog),
        (),
        (lower,),
        review_edges=(higher, lower),
    )
    reverse = analyze_conflicts(
        (dog, cat),
        (),
        (higher,),
        review_edges=(lower, higher),
    )

    assert forward == reverse
    assert forward.semantic_review_conflict_count == 1
    assert forward.conflicts[0].kind is ConflictKind.SEMANTIC_CANDIDATE
    assert forward.conflicts[0].member_ids == tuple(sorted((cat.id, dog.id)))


def test_multilabel_family_without_direct_cross_label_edge_is_not_misclassified() -> None:
    cat = make_record("bridge-cat", "cat")
    bridge = make_record("bridge-unlabeled", None)
    dog = make_record("bridge-dog", "dog")
    family = make_family((cat, bridge, dog), edge_count=2)
    edges = (
        make_edge(cat, bridge, DuplicateClassification.EXACT),
        make_edge(bridge, dog, DuplicateClassification.EXACT),
    )

    result = analyze_conflicts((cat, bridge, dog), (family,), edges)

    assert result.conflicts == ()
    assert result.definite_conflict_count == 0
