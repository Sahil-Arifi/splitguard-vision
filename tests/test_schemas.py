from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from splitguard.schemas import (
    AccuracyMetric,
    BinaryMetrics,
    DuplicateClassification,
    DuplicateEdge,
    DuplicateEvidence,
    DuplicateFamily,
    EdgeDecision,
    ImageRecord,
    PackageVersion,
    RepairArtifact,
    RepairAssignment,
    RepairSummary,
    RunMetadata,
    Split,
    SplitRatio,
    SplitStatistics,
    canonical_json,
    canonical_sha256,
    family_id_for,
    stable_id,
)

ZERO_SHA256 = "0" * 64
ONE_SHA256 = "1" * 64


def image_id(path: str) -> str:
    return stable_id("img", path)


def metadata() -> RunMetadata:
    return RunMetadata(
        timestamp=datetime(2026, 9, 3, 12, 0, tzinfo=UTC),
        git_commit_sha="a" * 40,
        git_dirty=False,
        python_version="3.11.9",
        os="test-os",
        cpu="test-cpu",
        cuda_available=False,
        package_versions=(PackageVersion(name="numpy", version="2.0.0"),),
        configuration_sha256=ZERO_SHA256,
        dataset_manifest_sha256=ONE_SHA256,
        random_seeds=(7, 11),
    )


def test_stable_id_uses_unambiguous_parts() -> None:
    first = stable_id("img", "ab", "c")
    second = stable_id("img", "a", "bc")

    assert first == stable_id("img", "ab", "c")
    assert first != second


@pytest.mark.parametrize("digest_length", [0, 15, 65])
def test_stable_id_rejects_invalid_digest_lengths(digest_length: int) -> None:
    with pytest.raises(ValueError, match="digest_length"):
        stable_id("img", "train/cat/a.png", digest_length=digest_length)


def test_stable_id_rejects_invalid_namespaces_and_parts() -> None:
    with pytest.raises(ValueError, match="namespace"):
        stable_id("Bad-Namespace", "value")
    with pytest.raises(ValueError, match="at least one"):
        stable_id("img")
    with pytest.raises(TypeError, match="must be strings"):
        stable_id("img", 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="NUL"):
        stable_id("img", "unsafe\x00part")


def test_family_id_requires_sorted_unique_members() -> None:
    first, second = sorted((image_id("a.png"), image_id("b.png")))

    with pytest.raises(ValueError, match="at least one member"):
        family_id_for(())
    with pytest.raises(ValueError, match="sorted and unique"):
        family_id_for((second, first))


def test_image_record_is_frozen_strict_and_canonical() -> None:
    record = ImageRecord(
        id=image_id("train/cat/a.png"),
        path="train/cat/a.png",
        split=Split.TRAIN,
        label="cat",
        byte_sha256=ZERO_SHA256,
        byte_size=12,
        width=8,
        height=8,
        format=".PNG",
        phash=0,
    )

    assert record.format == "png"
    with pytest.raises(ValidationError, match="frozen"):
        record.width = 9  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ImageRecord.model_validate(
            {
                **record.model_dump(),
                "width": "8",
                "unexpected": True,
            }
        )


@pytest.mark.parametrize(
    "path",
    [
        "C:/private/image.png",
        "/private/image.png",
        "../image.png",
        "train/../image.png",
        "train\\image.png",
        "train//image.png",
    ],
)
def test_image_record_rejects_nonportable_or_unsafe_paths(path: str) -> None:
    with pytest.raises(ValidationError):
        ImageRecord(
            id=image_id("safe.png"),
            path=path,
            split=Split.TRAIN,
            byte_sha256=ZERO_SHA256,
            byte_size=1,
            width=1,
            height=1,
            format="png",
        )


def test_image_record_rejects_invalid_hash_and_phash_range() -> None:
    with pytest.raises(ValidationError):
        ImageRecord(
            id=image_id("a.png"),
            path="a.png",
            split=Split.TRAIN,
            byte_sha256="not-a-sha",
            byte_size=1,
            width=1,
            height=1,
            format="png",
            phash=1 << 64,
        )


def test_duplicate_evidence_requires_a_signal() -> None:
    with pytest.raises(ValidationError, match="at least one evidence"):
        DuplicateEvidence()


def test_duplicate_edge_requires_canonical_pair_order() -> None:
    left = image_id("a.png")
    right = image_id("b.png")
    left, right = sorted((left, right))
    edge = DuplicateEdge(
        left_id=left,
        right_id=right,
        evidence=DuplicateEvidence(exact_match=True, phash_distance=0),
        classification=DuplicateClassification.EXACT,
        decision=EdgeDecision.DEFINITE,
        confidence=1.0,
    )

    assert edge.left_id < edge.right_id
    with pytest.raises(ValidationError, match="left_id < right_id"):
        DuplicateEdge(
            left_id=right,
            right_id=left,
            evidence=edge.evidence,
            classification=edge.classification,
            decision=edge.decision,
            confidence=edge.confidence,
        )


def test_duplicate_edge_classification_must_match_evidence() -> None:
    left, right = sorted((image_id("a.png"), image_id("b.png")))
    with pytest.raises(ValidationError, match="exact classification"):
        DuplicateEdge(
            left_id=left,
            right_id=right,
            evidence=DuplicateEvidence(phash_distance=2),
            classification=DuplicateClassification.EXACT,
            decision=EdgeDecision.DEFINITE,
            confidence=0.9,
        )


def test_family_members_and_identifier_are_canonical() -> None:
    members = tuple(sorted((image_id("a.png"), image_id("b.png"))))
    family = DuplicateFamily(
        family_id=family_id_for(members),
        member_ids=members,
        edge_count=1,
    )

    assert family.family_id == family_id_for(members)
    with pytest.raises(ValidationError, match="sorted and unique"):
        DuplicateFamily(
            family_id=family.family_id,
            member_ids=tuple(reversed(members)),
            edge_count=1,
        )
    with pytest.raises(ValidationError, match="family_id"):
        DuplicateFamily(
            family_id=stable_id("family", "wrong"),
            member_ids=members,
            edge_count=1,
        )


def test_canonical_json_is_stable_and_hashable() -> None:
    first = {"z": [3, 2, 1], "a": {"value": 1.25}}
    second = {"a": {"value": 1.25}, "z": [3, 2, 1]}

    assert canonical_json(first) == canonical_json(second)
    assert canonical_sha256(first) == canonical_sha256(second)
    assert '"cpu":"test-cpu"' in canonical_json(metadata())


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_canonical_json_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json({"unsafe": [value]})


def test_pydantic_float_fields_reject_non_finite_values() -> None:
    with pytest.raises(ValidationError):
        DuplicateEdge(
            left_id=image_id("a.png"),
            right_id=image_id("z.png"),
            evidence=DuplicateEvidence(cosine_similarity=math.nan),
            classification=DuplicateClassification.SEMANTIC_CANDIDATE,
            decision=EdgeDecision.REVIEW,
            confidence=0.5,
        )


def test_binary_metrics_are_derived_from_counts() -> None:
    metrics = BinaryMetrics.from_counts(8, 2, 4)

    assert metrics.precision == pytest.approx(0.8)
    assert metrics.recall == pytest.approx(8 / 12)
    assert metrics.f1 == pytest.approx(8 / 11)
    with pytest.raises(ValidationError, match="precision"):
        BinaryMetrics(
            true_positives=8,
            false_positives=2,
            false_negatives=4,
            precision=0.7,
            recall=8 / 12,
            f1=8 / 11,
        )


def test_accuracy_metric_must_match_counts() -> None:
    assert AccuracyMetric(correct=3, total=4, accuracy=0.75).accuracy == 0.75
    with pytest.raises(ValidationError, match="does not match"):
        AccuracyMetric(correct=3, total=4, accuracy=0.8)


def test_repair_artifact_enforces_ratio_sum_and_assignment_order() -> None:
    first, second = sorted((image_id("a.png"), image_id("b.png")))
    first_family = family_id_for((first,))
    second_family = family_id_for((second,))
    assignments = (
        RepairAssignment(
            record_id=first,
            family_id=first_family,
            original_split=Split.TRAIN,
            repaired_split=Split.TRAIN,
        ),
        RepairAssignment(
            record_id=second,
            family_id=second_family,
            original_split=Split.TEST,
            repaired_split=Split.TEST,
        ),
    )
    summary = RepairSummary(
        objective_value=0.0,
        split_size_error_before=0.0,
        split_size_error_after=0.0,
        class_divergence_before=0.0,
        class_divergence_after=0.0,
        definite_leakage_groups_before=0,
        definite_leakage_groups_after=0,
        moved_image_count=0,
        hard_group_invariant_satisfied=True,
    )
    stats = (
        SplitStatistics(split=Split.TRAIN, image_count=1),
        SplitStatistics(split=Split.TEST, image_count=1),
    )

    artifact = RepairArtifact(
        metadata=metadata(),
        requested_ratios=(
            SplitRatio(split=Split.TRAIN, ratio=0.5),
            SplitRatio(split=Split.VAL, ratio=0.0),
            SplitRatio(split=Split.TEST, ratio=0.5),
        ),
        integer_targets=((Split.TRAIN, 1), (Split.VAL, 0), (Split.TEST, 1)),
        assignments=assignments,
        split_size_weight=1.0,
        class_balance_weight=1.0,
        random_seed=42,
        local_improvement_iterations=250,
        repaired_manifest_sha256="a" * 64,
        before_split_statistics=stats,
        after_split_statistics=stats,
        summary=summary,
    )
    assert artifact.schema_version == "1.0"

    with pytest.raises(ValidationError, match="sum to one"):
        RepairArtifact(
            metadata=metadata(),
            requested_ratios=(
                SplitRatio(split=Split.TRAIN, ratio=0.8),
                SplitRatio(split=Split.VAL, ratio=0.1),
                SplitRatio(split=Split.TEST, ratio=0.05),
            ),
            integer_targets=((Split.TRAIN, 1), (Split.VAL, 0), (Split.TEST, 1)),
            assignments=assignments,
            split_size_weight=1.0,
            class_balance_weight=1.0,
            random_seed=42,
            local_improvement_iterations=250,
            repaired_manifest_sha256="a" * 64,
            before_split_statistics=stats,
            after_split_statistics=stats,
            summary=summary,
        )
