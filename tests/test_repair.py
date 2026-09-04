from __future__ import annotations

import hashlib
import os
import shutil
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from splitguard.manifest import parse_csv_manifest
from splitguard.repair import (
    MaterializationError,
    RepairInputError,
    RepairPlan,
    jensen_shannon_divergence,
    materialize_repaired_manifest,
    repair_splits,
    write_repaired_manifest,
)
from splitguard.schemas import (
    DuplicateFamily,
    ImageRecord,
    RepairAssignment,
    Split,
    SplitRatio,
    family_id_for,
    stable_id,
)

RATIOS = {Split.TRAIN: 0.5, Split.VAL: 0.25, Split.TEST: 0.25}
_TARGET_SPLITS_FOR_TEST = (Split.TRAIN, Split.VAL, Split.TEST)


def record(index: int, split: Split, label: str | None = "cat") -> ImageRecord:
    path = f"{split.value}/{label or 'unlabeled'}/{index:03d}.png"
    return ImageRecord(
        id=stable_id("img", path),
        path=path,
        split=split,
        label=label,
        byte_sha256=f"{index:064x}",
        byte_size=10,
        width=4,
        height=4,
        format="png",
    )


def family(*records: ImageRecord) -> DuplicateFamily:
    member_ids = tuple(sorted(item.id for item in records))
    return DuplicateFamily(
        family_id=family_id_for(member_ids),
        member_ids=member_ids,
        edge_count=max(0, len(member_ids) - 1),
    )


def assignment(record: ImageRecord, repaired_split: Split) -> RepairAssignment:
    return RepairAssignment(
        record_id=record.id,
        family_id=family_id_for((record.id,)),
        original_split=record.split,
        repaired_split=repaired_split,
    )


def repaired_splits(plan: RepairPlan) -> dict[str, Split]:
    return {item.record_id: item.repaired_split for item in plan.assignments}


def test_cross_split_family_is_indivisible_and_removes_leakage() -> None:
    left = record(1, Split.TRAIN)
    middle = record(2, Split.VAL)
    right = record(3, Split.TEST)
    singleton = record(4, Split.TRAIN, "dog")

    plan = repair_splits(
        (left, middle, right, singleton),
        (family(left, middle, right),),
        target_ratios=RATIOS,
    )

    assigned = repaired_splits(plan)
    assert len({assigned[left.id], assigned[middle.id], assigned[right.id]}) == 1
    assert plan.definite_leakage_groups_before == 1
    assert plan.definite_leakage_groups_after == 0
    assert plan.hard_group_invariant_satisfied is True


def test_input_permutations_and_iterable_forms_are_deterministic() -> None:
    records = tuple(
        record(index, tuple(_split for _split in Split)[index % 3], "cat" if index % 2 else "dog")
        for index in range(1, 9)
    )
    families = (family(records[0], records[1]), family(records[4], records[5]))

    forward = repair_splits(records, families, target_ratios=RATIOS, seed=7)
    reverse = repair_splits(
        reversed(records),
        reversed(families),
        target_ratios=(0.5, 0.25, 0.25),
        seed=7,
    )

    assert forward == reverse


def test_largest_remainder_targets_and_size_only_balance() -> None:
    records = tuple(record(index, Split.TRAIN) for index in range(1, 11))

    plan = repair_splits(
        records,
        target_ratios=(0.6, 0.2, 0.2),
        class_balance_weight=0.0,
    )

    assert plan.integer_targets == (
        (Split.TRAIN, 6),
        (Split.VAL, 2),
        (Split.TEST, 2),
    )
    assert Counter(item.repaired_split for item in plan.assignments) == {
        Split.TRAIN: 6,
        Split.VAL: 2,
        Split.TEST: 2,
    }
    assert plan.split_size_error_before == pytest.approx(0.4)
    assert plan.split_size_error_after == 0.0


def test_largest_remainder_ties_follow_split_order() -> None:
    records = (record(1, Split.TRAIN), record(2, Split.TRAIN))

    plan = repair_splits(
        records,
        target_ratios=(1 / 3, 1 / 3, 1 / 3),
        class_balance_weight=0.0,
    )

    assert plan.integer_targets == (
        (Split.TRAIN, 1),
        (Split.VAL, 1),
        (Split.TEST, 0),
    )


def test_class_balance_objective_improves_skewed_original_splits() -> None:
    records = tuple(
        [record(index, Split.TRAIN, "cat") for index in range(1, 7)]
        + [record(index, Split.TEST, "dog") for index in range(7, 13)]
    )

    plan = repair_splits(
        records,
        target_ratios=(0.5, 0.25, 0.25),
        split_size_weight=1.0,
        class_balance_weight=3.0,
        local_iterations=100,
    )

    assert plan.class_jsd_after < plan.class_jsd_before
    assert plan.split_size_error_after == 0.0
    assert plan.objective_value == pytest.approx(
        plan.split_size_weight * plan.split_size_error_after
        + plan.class_balance_weight * plan.class_jsd_after
    )


def test_class_jsd_is_target_ratio_weighted_against_global_distribution() -> None:
    records = (
        record(1, Split.TRAIN, "cat"),
        record(2, Split.VAL, "dog"),
    )

    plan = repair_splits(
        records,
        target_ratios=(0.5, 0.25, 0.25),
        local_iterations=0,
    )

    point_to_balanced = jensen_shannon_divergence(
        {"cat": 1},
        {"cat": 1, "dog": 1},
    )
    expected_before = 0.5 * point_to_balanced + 0.25 * point_to_balanced + 0.25
    assert plan.class_jsd_before == pytest.approx(expected_before)


def test_local_best_improvement_can_move_a_whole_group() -> None:
    records = tuple(
        record(index + 1, _TARGET_SPLITS_FOR_TEST[index % 3], label)
        for index, label in enumerate(("a", "a", "a", "a", "a", "b"))
    )

    greedy = repair_splits(
        records,
        target_ratios=(0.6, 0.2, 0.2),
        seed=0,
        local_iterations=0,
    )
    improved = repair_splits(
        records,
        target_ratios=(0.6, 0.2, 0.2),
        seed=0,
        local_iterations=1,
    )

    changed = [
        record_id
        for record_id, split in repaired_splits(greedy).items()
        if repaired_splits(improved)[record_id] is not split
    ]
    assert len(changed) == 1
    assert improved.objective_value < greedy.objective_value


def test_local_best_improvement_can_swap_whole_groups() -> None:
    labels = ("a", "a", "a", "c")
    records = tuple(
        record(index + 1, _TARGET_SPLITS_FOR_TEST[index % 3], label)
        for index, label in enumerate(labels)
    )

    greedy = repair_splits(
        records,
        target_ratios=(0.5, 0.25, 0.25),
        seed=7,
        local_iterations=0,
    )
    improved = repair_splits(
        records,
        target_ratios=(0.5, 0.25, 0.25),
        seed=7,
        local_iterations=1,
    )

    assert Counter(item.repaired_split for item in greedy.assignments) == Counter(
        item.repaired_split for item in improved.assignments
    )
    assert sum(
        left.repaired_split is not right.repaired_split
        for left, right in zip(greedy.assignments, improved.assignments, strict=True)
    ) == 2
    assert improved.objective_value < greedy.objective_value
    assert "local improvement reached its iteration limit" in improved.warnings


def test_oversized_family_reports_impossible_exact_balance_without_splitting() -> None:
    grouped = tuple(record(index, Split.TRAIN) for index in range(1, 5))
    other = record(5, Split.TEST, "dog")

    plan = repair_splits(
        (*grouped, other),
        (family(*grouped),),
        target_ratios=(0.4, 0.3, 0.3),
    )

    grouped_splits = {
        item.repaired_split
        for item in plan.assignments
        if item.record_id in {record.id for record in grouped}
    }
    assert len(grouped_splits) == 1
    assert plan.split_size_error_after > 0.0
    assert any("exceed every integer split target" in warning for warning in plan.warnings)
    assert plan.hard_group_invariant_satisfied is True


def test_transitive_family_is_treated_as_one_group_without_pairwise_edges() -> None:
    first = record(1, Split.TRAIN)
    second = record(2, Split.VAL)
    third = record(3, Split.TEST)
    transitive_family = family(first, second, third)
    transitive_family = transitive_family.model_copy(update={"edge_count": 2})

    plan = repair_splits(
        (first, second, third),
        (transitive_family,),
        target_ratios=(0.8, 0.1, 0.1),
    )

    assert len({item.repaired_split for item in plan.assignments}) == 1


def test_mixed_label_family_uses_full_count_vector_and_conserves_labels() -> None:
    cat = record(1, Split.TRAIN, "cat")
    dog = record(2, Split.TEST, "dog")
    bird = record(3, Split.VAL, "bird")
    plan = repair_splits(
        (cat, dog, bird),
        (family(cat, dog),),
        target_ratios=(1 / 3, 1 / 3, 1 / 3),
    )

    before_labels = sum(
        (Counter(dict(stat.class_counts)) for stat in plan.before_split_statistics),
        start=Counter(),
    )
    after_labels = sum(
        (Counter(dict(stat.class_counts)) for stat in plan.after_split_statistics),
        start=Counter(),
    )
    assert before_labels == after_labels == Counter(cat=1, dog=1, bird=1)
    assigned = repaired_splits(plan)
    assert assigned[cat.id] is assigned[dog.id]


def test_unlabeled_records_affect_size_but_not_class_divergence() -> None:
    records = (
        record(1, Split.TRAIN, None),
        record(2, Split.TEST, None),
        record(3, Split.VAL, None),
    )

    plan = repair_splits(records, target_ratios=(1 / 3, 1 / 3, 1 / 3))

    assert plan.class_jsd_before == 0.0
    assert plan.class_jsd_after == 0.0
    assert any("unlabeled" in warning for warning in plan.warnings)


def test_custom_split_is_included_in_before_error_then_eliminated() -> None:
    custom = record(1, Split.CUSTOM)

    plan = repair_splits(
        (custom,),
        target_ratios=(1.0, 0.0, 0.0),
        class_balance_weight=0.0,
    )

    assert tuple(stat.split for stat in plan.before_split_statistics) == tuple(Split)
    assert plan.split_size_error_before == 1.0
    assert plan.split_size_error_after == 0.0
    assert plan.assignments[0].repaired_split is Split.TRAIN


def test_empty_input_has_zero_metrics_and_deterministic_warning() -> None:
    plan = repair_splits((), target_ratios=RATIOS)

    assert plan.assignments == ()
    assert plan.integer_targets == (
        (Split.TRAIN, 0),
        (Split.VAL, 0),
        (Split.TEST, 0),
    )
    assert plan.objective_value == 0.0
    assert plan.hard_group_invariant_satisfied is True
    assert plan.warnings == ("input dataset is empty; no assignments were produced",)


def test_zero_objective_weights_are_supported_with_deficit_tie_breaking() -> None:
    records = tuple(record(index, Split.TRAIN) for index in range(1, 5))

    plan = repair_splits(
        records,
        target_ratios=(0.5, 0.25, 0.25),
        split_size_weight=0.0,
        class_balance_weight=0.0,
    )

    assert plan.objective_value == 0.0
    assert Counter(item.repaired_split for item in plan.assignments) == {
        Split.TRAIN: 2,
        Split.VAL: 1,
        Split.TEST: 1,
    }
    assert "class-balance objective weight is zero" in plan.warnings
    assert "split-size objective weight is zero" in plan.warnings


@pytest.mark.parametrize(
    "ratios,match",
    [
        ({Split.TRAIN: 0.8, Split.VAL: 0.2}, "exactly train"),
        (
            {Split.TRAIN: 0.7, Split.VAL: 0.2, Split.TEST: 0.2},
            "sum to one",
        ),
        (
            {Split.TRAIN: 0.8, Split.VAL: -0.1, Split.TEST: 0.3},
            "cannot be negative",
        ),
        (
            {Split.TRAIN: 0.8, Split.VAL: 0.1, Split.TEST: float("nan")},
            "finite",
        ),
    ],
)
def test_invalid_ratios_are_rejected(ratios: object, match: str) -> None:
    with pytest.raises((RepairInputError, ValueError), match=match):
        repair_splits((), target_ratios=ratios)  # type: ignore[arg-type]


def test_split_ratio_contract_input_is_supported() -> None:
    ratios = tuple(
        SplitRatio(split=split, ratio=ratio)
        for split, ratio in zip(_TARGET_SPLITS_FOR_TEST, (0.5, 0.25, 0.25), strict=True)
    )

    plan = repair_splits((), target_ratios=ratios)

    assert plan.requested_ratios == ratios


def test_near_one_ratio_sum_is_normalized_for_canonical_output() -> None:
    plan = repair_splits(
        (),
        target_ratios=(0.8, 0.1, 0.1000000005),
    )

    assert sum(item.ratio for item in plan.requested_ratios) == pytest.approx(1.0)
    assert all(0.0 <= item.ratio <= 1.0 for item in plan.requested_ratios)


def test_unknown_and_overlapping_family_members_are_rejected() -> None:
    known = record(1, Split.TRAIN)
    other = record(2, Split.TEST)
    unknown = record(3, Split.VAL)

    with pytest.raises(RepairInputError, match="unknown"):
        repair_splits(
            (known, other),
            (family(known, unknown),),
            target_ratios=RATIOS,
        )
    with pytest.raises(RepairInputError, match="disjoint"):
        repair_splits(
            (known, other),
            (family(known), family(known, other)),
            target_ratios=RATIOS,
        )


def test_duplicate_record_ids_are_rejected() -> None:
    item = record(1, Split.TRAIN)

    with pytest.raises(RepairInputError, match="unique IDs"):
        repair_splits((item, item), target_ratios=RATIOS)


def test_every_record_is_assigned_once_and_statistics_conserve_images() -> None:
    records = tuple(
        record(index, Split.TEST if index % 3 == 0 else Split.TRAIN, "cat")
        for index in range(1, 8)
    )

    plan = repair_splits(records, target_ratios=RATIOS)

    assert tuple(item.record_id for item in plan.assignments) == tuple(
        sorted(record.id for record in records)
    )
    assert sum(stat.image_count for stat in plan.before_split_statistics) == len(records)
    assert sum(stat.image_count for stat in plan.after_split_statistics) == len(records)
    assert plan.moved_image_count == sum(
        item.original_split is not item.repaired_split for item in plan.assignments
    )
    assert plan.summary.objective_value == plan.objective_value


def test_jensen_shannon_divergence_has_bounded_known_cases() -> None:
    assert jensen_shannon_divergence({}, {}) == 0.0
    assert jensen_shannon_divergence({}, {"cat": 1}) == 1.0
    assert jensen_shannon_divergence({"cat": 2}, {"cat": 10}) == 0.0
    assert jensen_shannon_divergence({"cat": 1}, {"dog": 1}) == 1.0
    assert jensen_shannon_divergence(
        {"cat": 1, "dog": 1},
        {"cat": 1},
    ) == pytest.approx(0.3112781244591328)


@pytest.mark.parametrize(
    "left,error",
    [
        ({"cat": -1}, ValueError),
        ({"cat": float("inf")}, ValueError),
        ({"cat": True}, TypeError),
        ({"": 1}, TypeError),
    ],
)
def test_jensen_shannon_inputs_are_strict(
    left: dict[str, object],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        jensen_shannon_divergence(left, {"cat": 1})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"split_size_weight": -1.0}, ValueError),
        ({"class_balance_weight": float("inf")}, ValueError),
        ({"seed": True}, TypeError),
        ({"local_iterations": -1}, ValueError),
    ],
)
def test_optimizer_settings_are_strict(kwargs: dict[str, object], error: type[Exception]) -> None:
    with pytest.raises(error):
        repair_splits((), target_ratios=RATIOS, **kwargs)  # type: ignore[arg-type]


def test_repair_plan_is_frozen() -> None:
    plan = repair_splits((), target_ratios=RATIOS)

    with pytest.raises(ValidationError, match="frozen"):
        plan.objective_value = 1.0


def test_repaired_manifest_is_deterministic_utf8_lf_and_atomic(tmp_path: Path) -> None:
    first = record(1, Split.TRAIN, "café")
    second = record(2, Split.TEST, "dog,large")
    output = tmp_path / "nested" / "repaired.csv"

    first_sha = write_repaired_manifest(
        output,
        (second, first),
        (assignment(second, Split.VAL), assignment(first, Split.TEST)),
    )
    first_bytes = output.read_bytes()
    second_sha = write_repaired_manifest(
        output,
        (first, second),
        (assignment(first, Split.TEST), assignment(second, Split.VAL)),
    )

    assert first_bytes == output.read_bytes()
    assert first_sha == second_sha == hashlib.sha256(first_bytes).hexdigest()
    assert first_bytes.startswith(b"path,split,label\n")
    assert b"\r\n" not in first_bytes
    assert "café" in first_bytes.decode("utf-8")
    parsed = parse_csv_manifest(output)
    assert [(entry.path, entry.split, entry.label) for entry in parsed.entries] == [
        (second.path, Split.VAL, "dog,large"),
        (first.path, Split.TEST, "café"),
    ]


def test_repaired_manifest_requires_exact_consistent_assignment_coverage(
    tmp_path: Path,
) -> None:
    first = record(1, Split.TRAIN)
    second = record(2, Split.TEST)
    output = tmp_path / "repaired.csv"

    with pytest.raises(RepairInputError, match=r"1 missing, 0 unknown"):
        write_repaired_manifest(output, (first, second), (assignment(first, Split.TRAIN),))
    with pytest.raises(RepairInputError, match="unique record IDs"):
        write_repaired_manifest(
            output,
            (first,),
            (assignment(first, Split.TRAIN), assignment(first, Split.TEST)),
        )
    mismatched = assignment(first, Split.TEST).model_copy(
        update={"original_split": Split.VAL}
    )
    with pytest.raises(RepairInputError, match="original split"):
        write_repaired_manifest(output, (first,), (mismatched,))


@pytest.mark.parametrize(
    "unsafe_path",
    ["../escape.png", "train//cat.png", "train/./cat.png", "C:/private.png", "bad\\path.png"],
)
def test_repaired_manifest_rejects_unsafe_unvalidated_record_paths(
    tmp_path: Path,
    unsafe_path: str,
) -> None:
    safe = record(1, Split.TRAIN)
    unsafe = safe.model_copy(update={"path": unsafe_path})

    with pytest.raises(RepairInputError, match="manifest paths"):
        write_repaired_manifest(
            tmp_path / "repaired.csv",
            (unsafe,),
            (assignment(safe, Split.TEST),),
        )


def test_repaired_manifest_preserves_existing_file_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = record(1, Split.TRAIN)
    output = tmp_path / "repaired.csv"
    output.write_bytes(b"original")

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("simulated failure with a private absolute path")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(RepairInputError, match="atomically") as error:
        write_repaired_manifest(output, (item,), (assignment(item, Split.TEST),))

    assert output.read_bytes() == b"original"
    assert list(tmp_path.glob(".repaired.csv.*.tmp")) == []
    assert str(tmp_path) not in str(error.value)


def _write_materialization_fixture(tmp_path: Path) -> tuple[Path, Path, tuple[ImageRecord, ...]]:
    dataset_root = tmp_path / "dataset"
    first = record(1, Split.TRAIN, "cat")
    second = record(2, Split.TEST, "dog")
    for item, content in ((first, b"first image"), (second, b"second image")):
        source = dataset_root.joinpath(*item.path.split("/"))
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(content)
    manifest = tmp_path / "repaired.csv"
    write_repaired_manifest(
        manifest,
        (first, second),
        (assignment(first, Split.TEST), assignment(second, Split.TRAIN)),
    )
    return dataset_root, manifest, (first, second)


def test_materialize_copy_uses_stable_flat_names_and_rewritten_manifest(
    tmp_path: Path,
) -> None:
    dataset_root, manifest, records = _write_materialization_fixture(tmp_path)
    output = tmp_path / "materialized"
    source_before = {
        item.id: dataset_root.joinpath(*item.path.split("/")).read_bytes()
        for item in records
    }

    result = materialize_repaired_manifest(manifest, dataset_root, output)

    assert result.mode == "copy"
    assert result.record_count == result.verified_file_count == 2
    assert result.manifest_sha256 == hashlib.sha256(
        (output / "manifest.csv").read_bytes()
    ).hexdigest()
    expected = {
        records[0].id: ("png", Split.TEST, "cat"),
        records[1].id: ("png", Split.TRAIN, "dog"),
    }
    materialized = parse_csv_manifest(output / "manifest.csv", dataset_root=output)
    assert len(materialized.entries) == 2
    for entry in materialized.entries:
        original_id = Path(entry.path).stem
        extension, split, label = expected[original_id]
        assert entry.path == f"images/{original_id}.{extension}"
        assert entry.split is split
        assert entry.label == label
        assert (output / entry.path).read_bytes() == source_before[original_id]
    assert {
        item.id: dataset_root.joinpath(*item.path.split("/")).read_bytes()
        for item in records
    } == source_before


def test_materialize_symlink_requests_relative_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root, manifest, _ = _write_materialization_fixture(tmp_path)
    output = tmp_path / "linked"
    requested_targets: list[str] = []

    def relative_link(target: str, link_name: str | os.PathLike[str]) -> None:
        requested_targets.append(target)
        assert not os.path.isabs(target)
        link_path = Path(link_name)
        source_path = (link_path.parent / target).resolve()
        shutil.copyfile(source_path, link_path)

    monkeypatch.setattr(os, "symlink", relative_link)
    result = materialize_repaired_manifest(
        manifest,
        dataset_root,
        output,
        mode="symlink",
    )

    assert result.mode == "symlink"
    assert len(requested_targets) == 2
    assert all(not os.path.isabs(target) for target in requested_targets)


def test_materialize_rejects_existing_and_overlapping_destinations(tmp_path: Path) -> None:
    dataset_root, manifest, _ = _write_materialization_fixture(tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()

    with pytest.raises(MaterializationError, match="must not already exist"):
        materialize_repaired_manifest(manifest, dataset_root, existing)
    overlapping = dataset_root / "derived"
    with pytest.raises(MaterializationError, match="must not overlap"):
        materialize_repaired_manifest(manifest, dataset_root, overlapping)
    assert not overlapping.exists()


@pytest.mark.parametrize("unsafe_source", ["missing/image.png", "images/no-extension"])
def test_materialize_preflight_rejects_missing_or_unsafe_sources_without_staging(
    tmp_path: Path,
    unsafe_source: str,
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    manifest = tmp_path / "repaired.csv"
    manifest.write_text(
        f"path,split,label\n{unsafe_source},train,cat\n",
        encoding="utf-8",
        newline="",
    )
    output = tmp_path / "materialized"

    with pytest.raises(MaterializationError):
        materialize_repaired_manifest(manifest, dataset_root, output)

    assert not output.exists()
    assert list(tmp_path.glob(".materialized.stage-*")) == []


def test_materialize_hash_failure_cleans_stage_and_does_not_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root, manifest, records = _write_materialization_fixture(tmp_path)
    output = tmp_path / "materialized"
    original_copy = shutil.copyfile

    def corrupt_copy(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> str:
        original_copy(source, destination)
        Path(destination).write_bytes(b"corrupt")
        return os.fspath(destination)

    monkeypatch.setattr(shutil, "copyfile", corrupt_copy)
    with pytest.raises(MaterializationError, match="verification failed"):
        materialize_repaired_manifest(manifest, dataset_root, output)

    assert not output.exists()
    assert list(tmp_path.glob(".materialized.stage-*")) == []
    assert dataset_root.joinpath(*records[0].path.split("/")).read_bytes() == b"first image"


def test_materialize_publish_failure_is_sanitized_and_cleans_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root, manifest, records = _write_materialization_fixture(tmp_path)
    output = tmp_path / "materialized"

    def fail_rename(_source: object, _destination: object) -> None:
        raise OSError(f"private path: {tmp_path}")

    monkeypatch.setattr(os, "rename", fail_rename)
    with pytest.raises(MaterializationError, match="no partial output") as error:
        materialize_repaired_manifest(manifest, dataset_root, output)

    assert str(tmp_path) not in str(error.value)
    assert not output.exists()
    assert list(tmp_path.glob(".materialized.stage-*")) == []
    assert dataset_root.joinpath(*records[0].path.split("/")).read_bytes() == b"first image"


def test_materialize_rejects_invalid_mode_before_creating_output(tmp_path: Path) -> None:
    dataset_root, manifest, _ = _write_materialization_fixture(tmp_path)
    output = tmp_path / "materialized"

    with pytest.raises(MaterializationError, match="mode"):
        materialize_repaired_manifest(
            manifest,
            dataset_root,
            output,
            mode="move",  # type: ignore[arg-type]
        )
    assert not output.exists()
