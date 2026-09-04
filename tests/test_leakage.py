"""Tests for definite cross-split leakage and semantic review policy."""

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from splitguard.leakage import (
    PROTECTED_EVALUATION_BOUNDARIES,
    LeakageAnalysisResult,
    analyze_leakage,
)
from splitguard.schemas import (
    DuplicateClassification,
    DuplicateEdge,
    DuplicateEvidence,
    DuplicateFamily,
    EdgeDecision,
    ImageRecord,
    Split,
    SplitBoundary,
    family_id_for,
    stable_id,
)


def make_record(name: str, split: Split, label: str | None = "cat") -> ImageRecord:
    return ImageRecord(
        id=stable_id("image", name),
        path=f"{split.value}/{name}.png",
        split=split,
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
) -> DuplicateEdge:
    left_id, right_id = sorted((left.id, right.id))
    if classification is DuplicateClassification.EXACT:
        evidence = DuplicateEvidence(
            exact_match=True,
            phash_distance=0,
            cosine_similarity=1.0,
        )
    elif classification is DuplicateClassification.TRANSFORMED_DUPLICATE:
        evidence = DuplicateEvidence(phash_distance=4, cosine_similarity=similarity)
    else:
        evidence = DuplicateEvidence(cosine_similarity=similarity)
    return DuplicateEdge(
        left_id=left_id,
        right_id=right_id,
        evidence=evidence,
        classification=classification,
        decision=decision,
        confidence=similarity,
    )


@pytest.mark.parametrize(
    ("left_split", "right_split", "expected_boundary"),
    (
        (Split.TRAIN, Split.VAL, SplitBoundary(left=Split.TRAIN, right=Split.VAL)),
        (Split.TRAIN, Split.TEST, SplitBoundary(left=Split.TRAIN, right=Split.TEST)),
        (Split.VAL, Split.TEST, SplitBoundary(left=Split.VAL, right=Split.TEST)),
    ),
)
def test_each_protected_boundary_is_reported_separately(
    left_split: Split,
    right_split: Split,
    expected_boundary: SplitBoundary,
) -> None:
    left = make_record("left", left_split)
    right = make_record("right", right_split)
    family = make_family((left, right), edge_count=1)

    result = analyze_leakage(
        (right, left),
        (family,),
        (make_edge(left, right, DuplicateClassification.EXACT),),
    )

    assert result.leakage_group_count == 1
    assert result.leakage_groups[0].boundaries == (expected_boundary,)
    counts = {
        item.boundary: item.leakage_group_count
        for item in result.boundary_group_counts
    }
    assert tuple(counts) == PROTECTED_EVALUATION_BOUNDARIES
    assert counts[expected_boundary] == 1
    assert sum(counts.values()) == 1


def test_multisplit_family_is_one_group_with_three_boundaries_and_unique_images() -> None:
    train = make_record("train-family", Split.TRAIN, "cat")
    val = make_record("val-family", Split.VAL, "cat")
    test = make_record("test-family", Split.TEST, "dog")
    clean_val = make_record("clean-val", Split.VAL)
    clean_test = make_record("clean-test", Split.TEST)
    family = make_family((train, val, test), edge_count=2)
    edges = (
        make_edge(train, val, DuplicateClassification.EXACT),
        make_edge(val, test, DuplicateClassification.TRANSFORMED_DUPLICATE),
    )

    result = analyze_leakage(
        (clean_test, val, train, clean_val, test),
        (family,),
        tuple(reversed(edges)),
    )

    group = result.leakage_groups[0]
    assert result.leakage_group_count == 1
    assert group.boundaries == PROTECTED_EVALUATION_BOUNDARIES
    assert group.evidence_types == (
        DuplicateClassification.EXACT,
        DuplicateClassification.TRANSFORMED_DUPLICATE,
    )
    assert group.strongest_evidence is DuplicateClassification.EXACT
    assert group.labels == ("cat", "dog")
    assert group.label_conflict is True
    assert [item.leakage_group_count for item in result.boundary_group_counts] == [1, 1, 1]
    assert result.contaminated_image_count == 3
    assert result.evaluation_image_count == 4
    assert result.contaminated_evaluation_image_count == 2
    assert result.contaminated_evaluation_fraction == pytest.approx(0.5)
    assert result.exact_leakage_group_count == 1
    assert result.perceptual_leakage_group_count == 0


def test_same_split_and_custom_only_boundaries_are_not_leakage() -> None:
    train_a = make_record("train-a", Split.TRAIN)
    train_b = make_record("train-b", Split.TRAIN)
    train_c = make_record("train-c", Split.TRAIN)
    custom = make_record("custom", Split.CUSTOM)
    same_split_family = make_family((train_a, train_b), edge_count=1)
    custom_family = make_family((train_c, custom), edge_count=1)
    edges = (
        make_edge(train_a, train_b, DuplicateClassification.EXACT),
        make_edge(train_c, custom, DuplicateClassification.EXACT),
    )

    result = analyze_leakage(
        (custom, train_b, train_c, train_a),
        (custom_family, same_split_family),
        edges,
    )

    assert result.leakage_groups == ()
    assert result.leakage_group_count == 0
    assert result.contaminated_image_ids == ()
    assert all(item.leakage_group_count == 0 for item in result.boundary_group_counts)
    assert result.contaminated_evaluation_fraction == 0.0


def test_perceptual_leakage_and_semantic_review_are_separate_and_deduplicated() -> None:
    train_near = make_record("near-train", Split.TRAIN)
    test_near = make_record("near-test", Split.TEST)
    train_review = make_record("review-train", Split.TRAIN)
    val_review = make_record("review-val", Split.VAL)
    val_same_split = make_record("review-val-two", Split.VAL)
    near_family = make_family((train_near, test_near), edge_count=1)
    near_edge = make_edge(
        train_near,
        test_near,
        DuplicateClassification.TRANSFORMED_DUPLICATE,
    )
    semantic_cross_split = make_edge(
        train_review,
        val_review,
        DuplicateClassification.SEMANTIC_CANDIDATE,
        decision=EdgeDecision.REVIEW,
        similarity=0.91,
    )
    semantic_same_split = make_edge(
        val_review,
        val_same_split,
        DuplicateClassification.SEMANTIC_CANDIDATE,
        decision=EdgeDecision.REVIEW,
        similarity=0.94,
    )

    result = analyze_leakage(
        (val_review, test_near, train_near, val_same_split, train_review),
        (near_family,),
        (near_edge, semantic_cross_split),
        review_edges=(semantic_same_split, semantic_cross_split),
    )

    assert result.leakage_group_count == 1
    assert result.exact_leakage_group_count == 0
    assert result.perceptual_leakage_group_count == 1
    assert result.embedding_only_review_count == 1
    assert result.semantic_review_edges == (semantic_cross_split,)
    assert train_review.id not in result.contaminated_image_ids
    assert val_review.id not in result.contaminated_image_ids


def test_semantic_only_family_is_review_not_definite_leakage() -> None:
    train = make_record("semantic-train", Split.TRAIN)
    test = make_record("semantic-test", Split.TEST)
    family = make_family((train, test), edge_count=0)
    semantic = make_edge(
        train,
        test,
        DuplicateClassification.SEMANTIC_CANDIDATE,
        decision=EdgeDecision.REVIEW,
    )

    result = analyze_leakage((test, train), (family,), (semantic,))

    assert result.leakage_groups == ()
    assert result.contaminated_image_ids == ()
    assert result.semantic_review_edges == (semantic,)
    assert result.embedding_only_review_count == 1


def test_analysis_result_is_immutable_and_checks_derived_counts() -> None:
    result = analyze_leakage((), (), ())

    with pytest.raises(ValidationError):
        result.leakage_group_count = 1
    with pytest.raises(ValidationError, match="does not match"):
        LeakageAnalysisResult(
            **{
                **result.model_dump(),
                "evaluation_image_count": 1,
            }
        )
