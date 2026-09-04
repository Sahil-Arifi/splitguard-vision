"""Deterministic group-aware repair for contaminated dataset splits.

The repair objective is a weighted sum of two bounded terms:

* split-size error is total-variation distance, ``0.5 * sum(abs(p_s - t_s))``;
* class error is the target-ratio-weighted, base-2 Jensen-Shannon divergence
  between each target split's labeled distribution and the global labeled
  distribution.

Unlabeled records contribute to split size but not class divergence. Duplicate
families are indivisible throughout greedy assignment and local improvement.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import re
import shutil
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import Field, field_validator, model_validator

from splitguard.manifest import ManifestError, parse_csv_manifest
from splitguard.schemas import (
    DuplicateFamily,
    ImageRecord,
    RepairAssignment,
    RepairSummary,
    Split,
    SplitRatio,
    SplitStatistics,
    StrictFrozenModel,
    family_id_for,
)

_TARGET_SPLITS = (Split.TRAIN, Split.VAL, Split.TEST)
_TARGET_SPLIT_SET = frozenset(_TARGET_SPLITS)
_SPLIT_ORDER = {split: index for index, split in enumerate(Split)}
_IMPROVEMENT_EPSILON = 1e-12
_MAX_SWAP_CANDIDATES = 4096
_SAFE_EXTENSION_RE = re.compile(r"^[A-Za-z0-9]{1,16}$")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")

RatioInput: TypeAlias = (
    Mapping[Split, float]
    | Mapping[str, float]
    | Iterable[SplitRatio]
    | Sequence[float]
)


class RepairInputError(ValueError):
    """Raised when records, families, ratios, or optimizer settings are invalid."""


class MaterializationError(RuntimeError):
    """Raised when a repaired manifest cannot be safely materialized."""


class MaterializationResult(StrictFrozenModel):
    """Privacy-safe summary of an explicit materialization run."""

    mode: Literal["copy", "symlink"]
    record_count: Annotated[int, Field(ge=0)]
    verified_file_count: Annotated[int, Field(ge=0)]
    manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def all_files_verified(self) -> Self:
        if self.verified_file_count != self.record_count:
            raise ValueError("every materialized file must be hash verified")
        return self


class RepairPlan(StrictFrozenModel):
    """A complete immutable and internally consistent split-repair result."""

    requested_ratios: Annotated[tuple[SplitRatio, ...], Field(min_length=3, max_length=3)]
    integer_targets: Annotated[
        tuple[tuple[Split, Annotated[int, Field(ge=0)]], ...],
        Field(min_length=3, max_length=3),
    ]
    assignments: tuple[RepairAssignment, ...]
    before_split_statistics: tuple[SplitStatistics, ...]
    after_split_statistics: tuple[SplitStatistics, ...]
    split_size_weight: Annotated[float, Field(ge=0.0)]
    class_balance_weight: Annotated[float, Field(ge=0.0)]
    seed: Annotated[int, Field(ge=0)]
    local_iterations: Annotated[int, Field(ge=0)]
    split_size_error_before: Annotated[float, Field(ge=0.0, le=1.0)]
    split_size_error_after: Annotated[float, Field(ge=0.0, le=1.0)]
    class_jsd_before: Annotated[float, Field(ge=0.0, le=1.0)]
    class_jsd_after: Annotated[float, Field(ge=0.0, le=1.0)]
    objective_value: Annotated[float, Field(ge=0.0)]
    moved_image_count: Annotated[int, Field(ge=0)]
    hard_group_invariant_satisfied: bool
    definite_leakage_groups_before: Annotated[int, Field(ge=0)]
    definite_leakage_groups_after: Annotated[int, Field(ge=0)]
    infeasibility_warnings: tuple[str, ...] = ()

    @field_validator("infeasibility_warnings")
    @classmethod
    def canonical_warnings(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value for value in values):
            raise ValueError("warnings cannot contain empty strings")
        if values != tuple(sorted(set(values))):
            raise ValueError("warnings must be sorted and unique")
        return values

    @model_validator(mode="after")
    def consistent_plan(self) -> Self:
        ratio_splits = tuple(item.split for item in self.requested_ratios)
        if ratio_splits != _TARGET_SPLITS:
            raise ValueError("requested_ratios must contain train, val, and test in order")
        if not math.isclose(
            sum(item.ratio for item in self.requested_ratios),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("requested_ratios must sum to one")

        target_splits = tuple(split for split, _ in self.integer_targets)
        if target_splits != _TARGET_SPLITS:
            raise ValueError("integer_targets must contain train, val, and test in order")
        if sum(count for _, count in self.integer_targets) != len(self.assignments):
            raise ValueError("integer target counts must sum to the assignment count")

        assignment_ids = tuple(item.record_id for item in self.assignments)
        if assignment_ids != tuple(sorted(set(assignment_ids))):
            raise ValueError("assignments must have unique record IDs in canonical order")
        before_splits = tuple(item.split for item in self.before_split_statistics)
        after_splits = tuple(item.split for item in self.after_split_statistics)
        expected_splits = tuple(sorted(set(before_splits), key=_SPLIT_ORDER.__getitem__))
        if before_splits != expected_splits or after_splits != before_splits:
            raise ValueError("before and after statistics must use the same canonical splits")

        before_counts = Counter(item.original_split for item in self.assignments)
        after_counts = Counter(item.repaired_split for item in self.assignments)
        if any(
            statistic.image_count != before_counts[statistic.split]
            for statistic in self.before_split_statistics
        ):
            raise ValueError("before statistics do not match assignments")
        if any(
            statistic.image_count != after_counts[statistic.split]
            for statistic in self.after_split_statistics
        ):
            raise ValueError("after statistics do not match assignments")
        if any(
            sum(count for _, count in statistic.class_counts) > statistic.image_count
            for statistic in (*self.before_split_statistics, *self.after_split_statistics)
        ):
            raise ValueError("class counts cannot exceed image counts")

        expected_moved = sum(
            item.original_split is not item.repaired_split for item in self.assignments
        )
        if self.moved_image_count != expected_moved:
            raise ValueError("moved_image_count does not match assignments")

        original_splits_by_family: dict[str, set[Split]] = defaultdict(set)
        repaired_splits_by_family: dict[str, set[Split]] = defaultdict(set)
        family_sizes: Counter[str] = Counter()
        for item in self.assignments:
            original_splits_by_family[item.family_id].add(item.original_split)
            repaired_splits_by_family[item.family_id].add(item.repaired_split)
            family_sizes[item.family_id] += 1
        expected_before_leakage = sum(
            family_sizes[family_id] > 1
            and len(splits & _TARGET_SPLIT_SET) > 1
            for family_id, splits in original_splits_by_family.items()
        )
        expected_after_leakage = sum(
            family_sizes[family_id] > 1
            and len(splits & _TARGET_SPLIT_SET) > 1
            for family_id, splits in repaired_splits_by_family.items()
        )
        if self.definite_leakage_groups_before != expected_before_leakage:
            raise ValueError("before leakage count does not match assignments")
        if self.definite_leakage_groups_after != expected_after_leakage:
            raise ValueError("after leakage count does not match assignments")
        expected_invariant = all(
            len(splits) <= 1 for splits in repaired_splits_by_family.values()
        )
        if self.hard_group_invariant_satisfied is not expected_invariant:
            raise ValueError("hard-group invariant flag does not match assignments")

        expected_objective = (
            self.split_size_weight * self.split_size_error_after
            + self.class_balance_weight * self.class_jsd_after
        )
        if not math.isclose(
            self.objective_value,
            expected_objective,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("objective_value does not match its weighted components")
        return self

    @property
    def class_divergence_before(self) -> float:
        """Compatibility name used by the serialized repair summary."""

        return self.class_jsd_before

    @property
    def class_divergence_after(self) -> float:
        """Compatibility name used by the serialized repair summary."""

        return self.class_jsd_after

    @property
    def warnings(self) -> tuple[str, ...]:
        """Concise compatibility alias for infeasibility warnings."""

        return self.infeasibility_warnings

    @property
    def summary(self) -> RepairSummary:
        """Return the schema-level summary used by ``RepairArtifact``."""

        return RepairSummary(
            objective_value=self.objective_value,
            split_size_error_before=self.split_size_error_before,
            split_size_error_after=self.split_size_error_after,
            class_divergence_before=self.class_jsd_before,
            class_divergence_after=self.class_jsd_after,
            definite_leakage_groups_before=self.definite_leakage_groups_before,
            definite_leakage_groups_after=self.definite_leakage_groups_after,
            moved_image_count=self.moved_image_count,
            hard_group_invariant_satisfied=self.hard_group_invariant_satisfied,
        )


@dataclass(frozen=True, slots=True)
class _Group:
    family_id: str
    member_ids: tuple[str, ...]
    label_counts: tuple[tuple[str, int], ...]
    original_split_counts: tuple[tuple[Split, int], ...]
    entropy: float
    seeded_digest: str

    @property
    def size(self) -> int:
        return len(self.member_ids)


@dataclass(slots=True)
class _AssignmentState:
    group_splits: dict[str, Split]
    split_counts: dict[Split, int]
    label_counts: dict[Split, Counter[str]]


def _validated_nonnegative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    if converted < 0.0:
        raise ValueError(f"{name} cannot be negative")
    return converted


def _validated_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value


def _normalize_split(value: Split | str) -> Split:
    if isinstance(value, Split):
        return value
    if not isinstance(value, str):
        raise TypeError("target ratio split keys must be Split values or strings")
    try:
        return Split(value)
    except ValueError as exc:
        raise RepairInputError(f"unsupported target split {value!r}") from exc


def _normalize_ratios(target_ratios: RatioInput) -> tuple[SplitRatio, ...]:
    raw: dict[Split, float] = {}
    if isinstance(target_ratios, Mapping):
        for key, value in target_ratios.items():
            split = _normalize_split(key)
            if split in raw:
                raise RepairInputError(f"target ratio for {split.value!r} is duplicated")
            raw[split] = _validated_nonnegative_float(value, f"{split.value} ratio")
    else:
        items: tuple[object, ...] = tuple(target_ratios)
        numeric_triplet = len(items) == 3 and all(
            isinstance(item, (int, float)) and not isinstance(item, bool) for item in items
        )
        if numeric_triplet:
            for split, value in zip(_TARGET_SPLITS, items, strict=True):
                raw[split] = _validated_nonnegative_float(value, f"{split.value} ratio")
        else:
            for item in items:
                if not isinstance(item, SplitRatio):
                    raise TypeError(
                        "target_ratios must be a mapping, three numeric values, "
                        "or SplitRatio values"
                    )
                if item.split in raw:
                    raise RepairInputError(
                        f"target ratio for {item.split.value!r} is duplicated"
                    )
                raw[item.split] = item.ratio

    if set(raw) != _TARGET_SPLIT_SET:
        missing = sorted(split.value for split in _TARGET_SPLIT_SET - set(raw))
        extras = sorted(split.value for split in set(raw) - _TARGET_SPLIT_SET)
        details: list[str] = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extras:
            details.append(f"unsupported {', '.join(extras)}")
        raise RepairInputError(
            "target_ratios must contain exactly train, val, and test"
            + (f" ({'; '.join(details)})" if details else "")
        )

    ratio_sum = sum(raw.values())
    if not math.isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise RepairInputError(f"target ratios must sum to one; got {ratio_sum!r}")
    normalized = {split: raw[split] / ratio_sum for split in _TARGET_SPLITS}
    return tuple(SplitRatio(split=split, ratio=normalized[split]) for split in _TARGET_SPLITS)


def _normalized_distribution(
    values: Mapping[str, int | float],
    name: str,
) -> dict[str, float]:
    normalized_values: dict[str, float] = {}
    for label, value in values.items():
        if not isinstance(label, str) or not label:
            raise TypeError(f"{name} labels must be non-empty strings")
        normalized_values[label] = _validated_nonnegative_float(value, f"{name}[{label!r}]")
    total = sum(normalized_values.values())
    if total == 0.0:
        return {}
    return {
        label: value / total
        for label, value in normalized_values.items()
        if value > 0.0
    }


def _jsd_from_probabilities(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if not left and not right:
        return 0.0
    if not left or not right:
        return 1.0
    divergence = 0.0
    for label in set(left) | set(right):
        left_probability = left.get(label, 0.0)
        right_probability = right.get(label, 0.0)
        midpoint = (left_probability + right_probability) / 2.0
        if left_probability > 0.0:
            divergence += 0.5 * left_probability * math.log2(left_probability / midpoint)
        if right_probability > 0.0:
            divergence += 0.5 * right_probability * math.log2(right_probability / midpoint)
    return min(1.0, max(0.0, divergence))


def jensen_shannon_divergence(
    left: Mapping[str, int | float],
    right: Mapping[str, int | float],
) -> float:
    """Return bounded base-2 JSD for two nonnegative label-count mappings.

    Two empty distributions have zero divergence. Exactly one empty
    distribution receives the maximum penalty of one.
    """

    return _jsd_from_probabilities(
        _normalized_distribution(left, "left"),
        _normalized_distribution(right, "right"),
    )


def _largest_remainder_targets(
    total_count: int,
    ratios: Mapping[Split, float],
) -> dict[Split, int]:
    raw_targets = {split: total_count * ratios[split] for split in _TARGET_SPLITS}
    targets = {split: math.floor(raw_targets[split]) for split in _TARGET_SPLITS}
    remaining = total_count - sum(targets.values())
    remainder_order = sorted(
        _TARGET_SPLITS,
        key=lambda split: (-(raw_targets[split] - targets[split]), _SPLIT_ORDER[split]),
    )
    for split in remainder_order[:remaining]:
        targets[split] += 1
    return targets


def _entropy(label_counts: Mapping[str, int]) -> float:
    total = sum(label_counts.values())
    if total == 0:
        return 0.0
    return -sum(
        (count / total) * math.log2(count / total)
        for count in label_counts.values()
        if count > 0
    )


def _seeded_digest(seed: int, family_id: str) -> str:
    payload = f"{seed}\0{family_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def _make_group(
    family_id: str,
    member_ids: tuple[str, ...],
    records_by_id: Mapping[str, ImageRecord],
    seed: int,
) -> _Group:
    label_counts: Counter[str] = Counter()
    original_counts: Counter[Split] = Counter()
    for member_id in member_ids:
        record = records_by_id[member_id]
        original_counts[record.split] += 1
        if record.label is not None:
            label_counts[record.label] += 1
    return _Group(
        family_id=family_id,
        member_ids=member_ids,
        label_counts=tuple(sorted(label_counts.items())),
        original_split_counts=tuple(
            sorted(original_counts.items(), key=lambda item: _SPLIT_ORDER[item[0]])
        ),
        entropy=_entropy(label_counts),
        seeded_digest=_seeded_digest(seed, family_id),
    )


def _build_groups(
    records_by_id: Mapping[str, ImageRecord],
    families: Iterable[DuplicateFamily],
    seed: int,
) -> tuple[tuple[_Group, ...], tuple[DuplicateFamily, ...]]:
    ordered_families = tuple(sorted(families, key=lambda family: family.family_id))
    family_ids = tuple(family.family_id for family in ordered_families)
    if family_ids != tuple(sorted(set(family_ids))):
        raise RepairInputError("families must have unique IDs")

    owner_by_member: dict[str, str] = {}
    groups: list[_Group] = []
    for family in ordered_families:
        for member_id in family.member_ids:
            if member_id not in records_by_id:
                raise RepairInputError("a duplicate family references an unknown record ID")
            if member_id in owner_by_member:
                raise RepairInputError("duplicate families must be disjoint")
            owner_by_member[member_id] = family.family_id
        groups.append(
            _make_group(
                family.family_id,
                family.member_ids,
                records_by_id,
                seed,
            )
        )

    for record_id in sorted(set(records_by_id) - set(owner_by_member)):
        singleton_members = (record_id,)
        groups.append(
            _make_group(
                family_id_for(singleton_members),
                singleton_members,
                records_by_id,
                seed,
            )
        )
    return tuple(groups), ordered_families


def _split_size_error(
    split_counts: Mapping[Split, int],
    total_count: int,
    ratios: Mapping[Split, float],
) -> float:
    if total_count == 0:
        return 0.0
    return 0.5 * sum(
        abs(split_counts.get(split, 0) / total_count - ratios.get(split, 0.0))
        for split in Split
    )


def _class_jsd(
    label_counts: Mapping[Split, Mapping[str, int]],
    global_label_counts: Mapping[str, int],
    ratios: Mapping[Split, float],
) -> float:
    global_distribution = _normalized_distribution(global_label_counts, "global labels")
    if not global_distribution:
        return 0.0
    divergence = 0.0
    for split in _TARGET_SPLITS:
        split_distribution = _normalized_distribution(
            label_counts.get(split, {}),
            f"{split.value} labels",
        )
        divergence += ratios[split] * _jsd_from_probabilities(
            split_distribution,
            global_distribution,
        )
    return min(1.0, max(0.0, divergence))


def _objective(
    split_counts: Mapping[Split, int],
    label_counts: Mapping[Split, Mapping[str, int]],
    *,
    total_count: int,
    global_label_counts: Mapping[str, int],
    ratios: Mapping[Split, float],
    split_size_weight: float,
    class_balance_weight: float,
) -> tuple[float, float, float]:
    size_error = _split_size_error(split_counts, total_count, ratios)
    class_error = _class_jsd(label_counts, global_label_counts, ratios)
    return (
        split_size_weight * size_error + class_balance_weight * class_error,
        size_error,
        class_error,
    )


def _apply_group(
    state: _AssignmentState,
    group: _Group,
    split: Split,
    multiplier: int,
) -> None:
    state.split_counts[split] += multiplier * group.size
    for label, count in group.label_counts:
        state.label_counts[split][label] += multiplier * count


def _group_original_count(group: _Group, split: Split) -> int:
    return dict(group.original_split_counts).get(split, 0)


def _choose_greedy_split(
    state: _AssignmentState,
    group: _Group,
    *,
    global_label_counts: Mapping[str, int],
    ratios: Mapping[Split, float],
    integer_targets: Mapping[Split, int],
    split_size_weight: float,
    class_balance_weight: float,
) -> Split:
    best_split = _TARGET_SPLITS[0]
    best_score = math.inf
    best_tie: tuple[int, int, int] | None = None
    for split in _TARGET_SPLITS:
        deficit = integer_targets[split] - state.split_counts[split]
        plurality = _group_original_count(group, split)
        tie = (-deficit, -plurality, _SPLIT_ORDER[split])
        _apply_group(state, group, split, 1)
        score, _, _ = _objective(
            state.split_counts,
            state.label_counts,
            total_count=sum(state.split_counts.values()),
            global_label_counts=global_label_counts,
            ratios=ratios,
            split_size_weight=split_size_weight,
            class_balance_weight=class_balance_weight,
        )
        _apply_group(state, group, split, -1)
        if score < best_score - _IMPROVEMENT_EPSILON or (
            math.isclose(score, best_score, rel_tol=0.0, abs_tol=_IMPROVEMENT_EPSILON)
            and (best_tie is None or tie < best_tie)
        ):
            best_split = split
            best_score = score
            best_tie = tie
    return best_split


def _evaluate_move(
    state: _AssignmentState,
    group: _Group,
    destination: Split,
    *,
    total_count: int,
    global_label_counts: Mapping[str, int],
    ratios: Mapping[Split, float],
    split_size_weight: float,
    class_balance_weight: float,
) -> float:
    source = state.group_splits[group.family_id]
    _apply_group(state, group, source, -1)
    _apply_group(state, group, destination, 1)
    score, _, _ = _objective(
        state.split_counts,
        state.label_counts,
        total_count=total_count,
        global_label_counts=global_label_counts,
        ratios=ratios,
        split_size_weight=split_size_weight,
        class_balance_weight=class_balance_weight,
    )
    _apply_group(state, group, destination, -1)
    _apply_group(state, group, source, 1)
    return score


def _evaluate_swap(
    state: _AssignmentState,
    left: _Group,
    right: _Group,
    *,
    total_count: int,
    global_label_counts: Mapping[str, int],
    ratios: Mapping[Split, float],
    split_size_weight: float,
    class_balance_weight: float,
) -> float:
    left_split = state.group_splits[left.family_id]
    right_split = state.group_splits[right.family_id]
    _apply_group(state, left, left_split, -1)
    _apply_group(state, right, right_split, -1)
    _apply_group(state, left, right_split, 1)
    _apply_group(state, right, left_split, 1)
    score, _, _ = _objective(
        state.split_counts,
        state.label_counts,
        total_count=total_count,
        global_label_counts=global_label_counts,
        ratios=ratios,
        split_size_weight=split_size_weight,
        class_balance_weight=class_balance_weight,
    )
    _apply_group(state, left, right_split, -1)
    _apply_group(state, right, left_split, -1)
    _apply_group(state, left, left_split, 1)
    _apply_group(state, right, right_split, 1)
    return score


def _local_improvement(
    state: _AssignmentState,
    groups: tuple[_Group, ...],
    *,
    total_count: int,
    global_label_counts: Mapping[str, int],
    ratios: Mapping[Split, float],
    split_size_weight: float,
    class_balance_weight: float,
    local_iterations: int,
) -> tuple[bool, bool]:
    ordered_groups = tuple(sorted(groups, key=lambda group: group.family_id))
    current_score, _, _ = _objective(
        state.split_counts,
        state.label_counts,
        total_count=total_count,
        global_label_counts=global_label_counts,
        ratios=ratios,
        split_size_weight=split_size_weight,
        class_balance_weight=class_balance_weight,
    )
    swap_search_truncated = False
    reached_iteration_limit = False
    for iteration in range(local_iterations):
        best_move: tuple[float, str, Split, _Group] | None = None
        for group in ordered_groups:
            source = state.group_splits[group.family_id]
            for destination in _TARGET_SPLITS:
                if destination is source:
                    continue
                score = _evaluate_move(
                    state,
                    group,
                    destination,
                    total_count=total_count,
                    global_label_counts=global_label_counts,
                    ratios=ratios,
                    split_size_weight=split_size_weight,
                    class_balance_weight=class_balance_weight,
                )
                move_candidate = (score, group.family_id, destination, group)
                if score < current_score - _IMPROVEMENT_EPSILON and (
                    best_move is None or move_candidate[:3] < best_move[:3]
                ):
                    best_move = move_candidate
        if best_move is not None:
            score, _, destination, group = best_move
            source = state.group_splits[group.family_id]
            _apply_group(state, group, source, -1)
            _apply_group(state, group, destination, 1)
            state.group_splits[group.family_id] = destination
            current_score = score
            if iteration == local_iterations - 1:
                reached_iteration_limit = True
            continue

        best_swap: tuple[float, str, str, _Group, _Group] | None = None
        comparisons = 0
        stop_search = False
        for left_index, left in enumerate(ordered_groups):
            for right in ordered_groups[left_index + 1 :]:
                if state.group_splits[left.family_id] is state.group_splits[right.family_id]:
                    continue
                if comparisons >= _MAX_SWAP_CANDIDATES:
                    swap_search_truncated = True
                    stop_search = True
                    break
                comparisons += 1
                score = _evaluate_swap(
                    state,
                    left,
                    right,
                    total_count=total_count,
                    global_label_counts=global_label_counts,
                    ratios=ratios,
                    split_size_weight=split_size_weight,
                    class_balance_weight=class_balance_weight,
                )
                swap_candidate = (score, left.family_id, right.family_id, left, right)
                if score < current_score - _IMPROVEMENT_EPSILON and (
                    best_swap is None or swap_candidate[:3] < best_swap[:3]
                ):
                    best_swap = swap_candidate
            if stop_search:
                break
        if best_swap is None:
            break
        score, _, _, left, right = best_swap
        left_split = state.group_splits[left.family_id]
        right_split = state.group_splits[right.family_id]
        _apply_group(state, left, left_split, -1)
        _apply_group(state, right, right_split, -1)
        _apply_group(state, left, right_split, 1)
        _apply_group(state, right, left_split, 1)
        state.group_splits[left.family_id] = right_split
        state.group_splits[right.family_id] = left_split
        current_score = score
        if iteration == local_iterations - 1:
            reached_iteration_limit = True
    return swap_search_truncated, reached_iteration_limit


def _statistics(
    records: Iterable[ImageRecord],
    splits: tuple[Split, ...],
    repaired_split_by_id: Mapping[str, Split] | None = None,
) -> tuple[SplitStatistics, ...]:
    image_counts: Counter[Split] = Counter()
    class_counts: dict[Split, Counter[str]] = {split: Counter() for split in splits}
    for record in records:
        split = (
            record.split
            if repaired_split_by_id is None
            else repaired_split_by_id[record.id]
        )
        image_counts[split] += 1
        if record.label is not None:
            class_counts[split][record.label] += 1
    return tuple(
        SplitStatistics(
            split=split,
            image_count=image_counts[split],
            class_counts=tuple(sorted(class_counts[split].items())),
        )
        for split in splits
    )


def repair_splits(
    records: Iterable[ImageRecord],
    families: Iterable[DuplicateFamily] = (),
    *,
    target_ratios: RatioInput,
    split_size_weight: float = 1.0,
    class_balance_weight: float = 1.0,
    seed: int = 42,
    local_iterations: int = 250,
) -> RepairPlan:
    """Return a deterministic repair plan without moving or modifying source files.

    Groups are greedily assigned in descending size and label-entropy order,
    followed by a seeded SHA tie-break. Candidate split ties prefer integer
    target deficit, then the group's original split plurality, then
    train/validation/test order. Strictly improving whole-group moves and a
    bounded deterministic swap search refine the result.
    """

    ratios_contract = _normalize_ratios(target_ratios)
    ratios = {item.split: item.ratio for item in ratios_contract}
    accepted_size_weight = _validated_nonnegative_float(
        split_size_weight,
        "split_size_weight",
    )
    accepted_class_weight = _validated_nonnegative_float(
        class_balance_weight,
        "class_balance_weight",
    )
    accepted_seed = _validated_nonnegative_int(seed, "seed")
    accepted_iterations = _validated_nonnegative_int(local_iterations, "local_iterations")

    record_snapshot = tuple(records)
    records_by_id = {record.id: record for record in record_snapshot}
    if len(records_by_id) != len(record_snapshot):
        raise RepairInputError("records must contain unique IDs")

    groups, definite_families = _build_groups(records_by_id, families, accepted_seed)
    total_count = len(record_snapshot)
    integer_targets = _largest_remainder_targets(total_count, ratios)
    ordered_groups = tuple(
        sorted(
            groups,
            key=lambda group: (
                -group.size,
                -group.entropy,
                group.seeded_digest,
                group.family_id,
            ),
        )
    )

    global_label_counts: Counter[str] = Counter()
    before_split_counts: Counter[Split] = Counter()
    before_label_counts: dict[Split, Counter[str]] = {
        split: Counter() for split in Split
    }
    for record in record_snapshot:
        before_split_counts[record.split] += 1
        if record.label is not None:
            global_label_counts[record.label] += 1
            before_label_counts[record.split][record.label] += 1

    state = _AssignmentState(
        group_splits={},
        split_counts={split: 0 for split in Split},
        label_counts={split: Counter() for split in Split},
    )
    for group in ordered_groups:
        destination = _choose_greedy_split(
            state,
            group,
            global_label_counts=global_label_counts,
            ratios=ratios,
            integer_targets=integer_targets,
            split_size_weight=accepted_size_weight,
            class_balance_weight=accepted_class_weight,
        )
        state.group_splits[group.family_id] = destination
        _apply_group(state, group, destination, 1)

    swap_truncated, iteration_limit_reached = _local_improvement(
        state,
        groups,
        total_count=total_count,
        global_label_counts=global_label_counts,
        ratios=ratios,
        split_size_weight=accepted_size_weight,
        class_balance_weight=accepted_class_weight,
        local_iterations=accepted_iterations,
    )

    repaired_split_by_id: dict[str, Split] = {}
    family_id_by_record: dict[str, str] = {}
    for group in groups:
        repaired_split = state.group_splits[group.family_id]
        for member_id in group.member_ids:
            repaired_split_by_id[member_id] = repaired_split
            family_id_by_record[member_id] = group.family_id

    assignments = tuple(
        RepairAssignment(
            record_id=record_id,
            family_id=family_id_by_record[record_id],
            original_split=records_by_id[record_id].split,
            repaired_split=repaired_split_by_id[record_id],
        )
        for record_id in sorted(records_by_id)
    )
    metric_splits = tuple(
        sorted(
            _TARGET_SPLIT_SET | {record.split for record in record_snapshot},
            key=_SPLIT_ORDER.__getitem__,
        )
    )
    before_statistics = _statistics(record_snapshot, metric_splits)
    after_statistics = _statistics(
        record_snapshot,
        metric_splits,
        repaired_split_by_id,
    )
    _, size_error_before, class_jsd_before = _objective(
        before_split_counts,
        before_label_counts,
        total_count=total_count,
        global_label_counts=global_label_counts,
        ratios=ratios,
        split_size_weight=accepted_size_weight,
        class_balance_weight=accepted_class_weight,
    )
    objective_value, size_error_after, class_jsd_after = _objective(
        state.split_counts,
        state.label_counts,
        total_count=total_count,
        global_label_counts=global_label_counts,
        ratios=ratios,
        split_size_weight=accepted_size_weight,
        class_balance_weight=accepted_class_weight,
    )

    original_split_by_id = {record.id: record.split for record in record_snapshot}
    leakage_before = sum(
        len(
            {
                original_split_by_id[member_id]
                for member_id in family.member_ids
                if original_split_by_id[member_id] in _TARGET_SPLIT_SET
            }
        )
        > 1
        for family in definite_families
    )
    leakage_after = sum(
        len(
            {
                repaired_split_by_id[member_id]
                for member_id in family.member_ids
                if repaired_split_by_id[member_id] in _TARGET_SPLIT_SET
            }
        )
        > 1
        for family in definite_families
    )
    invariant_satisfied = (
        len(repaired_split_by_id) == total_count
        and all(
            len({repaired_split_by_id[member_id] for member_id in family.member_ids}) <= 1
            for family in definite_families
        )
    )

    warnings: set[str] = set()
    if total_count == 0:
        warnings.add("input dataset is empty; no assignments were produced")
    unlabeled_count = sum(record.label is None for record in record_snapshot)
    if unlabeled_count:
        warnings.add(
            f"{unlabeled_count} unlabeled record(s) were excluded from class divergence"
        )
    max_integer_target = max(integer_targets.values(), default=0)
    oversized_groups = sum(group.size > max_integer_target for group in groups)
    if oversized_groups:
        warnings.add(
            f"{oversized_groups} group(s) exceed every integer split target; "
            "exact size balance is impossible"
        )
    if any(state.split_counts[split] != integer_targets[split] for split in _TARGET_SPLITS):
        warnings.add(
            "integer split targets were not reached because groups are indivisible "
            "or bounded optimization found no exact assignment"
        )
    if accepted_size_weight == 0.0:
        warnings.add("split-size objective weight is zero")
    if accepted_class_weight == 0.0:
        warnings.add("class-balance objective weight is zero")
    if swap_truncated:
        warnings.add(
            f"local swap search was limited to {_MAX_SWAP_CANDIDATES} candidate pairs"
        )
    if iteration_limit_reached:
        warnings.add("local improvement reached its iteration limit")

    return RepairPlan(
        requested_ratios=ratios_contract,
        integer_targets=tuple(
            (split, integer_targets[split]) for split in _TARGET_SPLITS
        ),
        assignments=assignments,
        before_split_statistics=before_statistics,
        after_split_statistics=after_statistics,
        split_size_weight=accepted_size_weight,
        class_balance_weight=accepted_class_weight,
        seed=accepted_seed,
        local_iterations=accepted_iterations,
        split_size_error_before=size_error_before,
        split_size_error_after=size_error_after,
        class_jsd_before=class_jsd_before,
        class_jsd_after=class_jsd_after,
        objective_value=objective_value,
        moved_image_count=sum(
            assignment.original_split is not assignment.repaired_split
            for assignment in assignments
        ),
        hard_group_invariant_satisfied=invariant_satisfied,
        definite_leakage_groups_before=leakage_before,
        definite_leakage_groups_after=leakage_after,
        infeasibility_warnings=tuple(sorted(warnings)),
    )


def _validated_relative_posix_path(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("manifest paths must be strings")
    normalized = unicodedata.normalize("NFC", value)
    if normalized != value:
        raise RepairInputError("manifest paths must use canonical Unicode normalization")
    if not value or "\x00" in value or "\\" in value:
        raise RepairInputError("manifest paths must be non-empty NUL-free POSIX paths")
    if value.startswith("/") or _WINDOWS_DRIVE_RE.match(value):
        raise RepairInputError("manifest paths must be relative")
    segments = value.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise RepairInputError("manifest paths cannot contain empty, current, or parent segments")
    if PurePosixPath(value).is_absolute():
        raise RepairInputError("manifest paths must be relative")
    return value


def _render_manifest_bytes(
    rows: Iterable[tuple[str, Split, str | None]],
) -> bytes:
    canonical_rows = tuple(
        sorted(
            rows,
            key=lambda row: (
                row[0].casefold(),
                row[0],
                row[1].value,
                "" if row[2] is None else row[2],
            ),
        )
    )
    logical_paths: set[str] = set()
    for path, _, _ in canonical_rows:
        _validated_relative_posix_path(path)
        collision_key = path.casefold()
        if collision_key in logical_paths:
            raise RepairInputError("manifest rows contain duplicate logical paths")
        logical_paths.add(collision_key)

    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("path", "split", "label"))
    for path, split, label in canonical_rows:
        writer.writerow((path, split.value, "" if label is None else label))
    return stream.getvalue().encode()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.is_dir():
        raise OSError("output path is a directory")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


def _manifest_rows_for_assignments(
    records: Iterable[ImageRecord],
    assignments: Iterable[RepairAssignment],
) -> tuple[tuple[str, Split, str | None], ...]:
    record_snapshot = tuple(records)
    assignment_snapshot = tuple(assignments)
    records_by_id = {record.id: record for record in record_snapshot}
    if len(records_by_id) != len(record_snapshot):
        raise RepairInputError("records must contain unique IDs")
    assignments_by_id = {
        assignment.record_id: assignment for assignment in assignment_snapshot
    }
    if len(assignments_by_id) != len(assignment_snapshot):
        raise RepairInputError("assignments must contain unique record IDs")

    record_ids = set(records_by_id)
    assignment_ids = set(assignments_by_id)
    if record_ids != assignment_ids:
        missing_count = len(record_ids - assignment_ids)
        unknown_count = len(assignment_ids - record_ids)
        raise RepairInputError(
            "assignments must cover every record exactly once "
            f"({missing_count} missing, {unknown_count} unknown)"
        )

    rows: list[tuple[str, Split, str | None]] = []
    for record_id in sorted(records_by_id):
        record = records_by_id[record_id]
        assignment = assignments_by_id[record_id]
        if assignment.original_split is not record.split:
            raise RepairInputError("an assignment original split does not match its record")
        if assignment.repaired_split not in _TARGET_SPLIT_SET:
            raise RepairInputError("repaired assignments must target train, val, or test")
        rows.append(
            (
                _validated_relative_posix_path(record.path),
                assignment.repaired_split,
                record.label,
            )
        )
    return tuple(rows)


def write_repaired_manifest(
    path: str | os.PathLike[str],
    records: Iterable[ImageRecord],
    assignments: Iterable[RepairAssignment],
) -> str:
    """Atomically write canonical ``path,split,label`` CSV and return its SHA-256.

    The assignment collection must cover the record snapshot exactly once.
    Existing regular files are atomically replaced; source records and images
    are never modified.
    """

    payload = _render_manifest_bytes(_manifest_rows_for_assignments(records, assignments))
    output_path = Path(os.path.abspath(Path(path)))
    try:
        _atomic_write_bytes(output_path, payload)
    except OSError as exc:
        raise RepairInputError("could not write repaired manifest atomically") from exc
    return hashlib.sha256(payload).hexdigest()


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _prospective_resolve(path: Path) -> Path:
    missing_parts: list[str] = []
    cursor = path
    while not _path_lexists(cursor):
        missing_parts.append(cursor.name)
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent
    resolved = cursor.resolve(strict=True)
    for part in reversed(missing_parts):
        resolved /= part
    return resolved


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _sha256_file(path: Path, logical_path: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise MaterializationError(
            f"could not read source or staged file {logical_path!r}"
        ) from exc
    return digest.hexdigest()


def _safe_extension(path: str) -> str:
    suffix = PurePosixPath(path).suffix
    extension = suffix.removeprefix(".")
    if not extension or _SAFE_EXTENSION_RE.fullmatch(extension) is None:
        raise MaterializationError(
            f"manifest path {path!r} does not have a safe file extension"
        )
    return extension.casefold()


@dataclass(frozen=True, slots=True)
class _MaterializationItem:
    source_path: Path
    source_logical_path: str
    destination_relative_path: str
    split: Split
    label: str | None
    source_sha256: str


def _preflight_materialization(
    manifest_path: str | os.PathLike[str],
    dataset_root: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
) -> tuple[Path, tuple[_MaterializationItem, ...]]:
    root_lexical = Path(os.path.abspath(Path(dataset_root)))
    output_lexical = Path(os.path.abspath(Path(output_dir)))
    if not root_lexical.exists() or not root_lexical.is_dir():
        raise MaterializationError("dataset root must be an existing directory")
    if _path_lexists(output_lexical):
        raise MaterializationError("output directory must not already exist")

    root_resolved = root_lexical.resolve(strict=True)
    try:
        output_resolved = _prospective_resolve(output_lexical)
    except OSError as exc:
        raise MaterializationError("output directory parent cannot be resolved") from exc
    if _paths_overlap(root_resolved, output_resolved):
        raise MaterializationError("output directory must not overlap the dataset root")

    try:
        manifest = parse_csv_manifest(manifest_path, dataset_root=root_lexical)
    except ManifestError as exc:
        raise MaterializationError(f"invalid repaired manifest: {exc}") from exc

    items: list[_MaterializationItem] = []
    destination_paths: set[str] = set()
    for entry in manifest.entries:
        try:
            source_logical_path = _validated_relative_posix_path(entry.path)
        except (RepairInputError, TypeError) as exc:
            raise MaterializationError(
                "repaired manifest contains an invalid relative path"
            ) from exc
        if entry.split not in _TARGET_SPLIT_SET:
            raise MaterializationError(
                "repaired manifest destinations must be train, val, or test"
            )
        extension = _safe_extension(source_logical_path)
        destination_relative_path = f"images/{entry.id}.{extension}"
        collision_key = destination_relative_path.casefold()
        if collision_key in destination_paths:
            raise MaterializationError("materialized image names are not unique")
        destination_paths.add(collision_key)

        source_lexical = root_lexical.joinpath(*PurePosixPath(source_logical_path).parts)
        try:
            source_resolved = source_lexical.resolve(strict=True)
        except OSError as exc:
            raise MaterializationError(
                f"source file is missing or inaccessible: {source_logical_path!r}"
            ) from exc
        if not source_resolved.is_relative_to(root_resolved):
            raise MaterializationError(
                f"source path escapes the dataset root: {source_logical_path!r}"
            )
        if not source_resolved.is_file():
            raise MaterializationError(
                f"source path is not a file: {source_logical_path!r}"
            )
        items.append(
            _MaterializationItem(
                source_path=source_resolved,
                source_logical_path=source_logical_path,
                destination_relative_path=destination_relative_path,
                split=entry.split,
                label=entry.label,
                source_sha256=_sha256_file(source_resolved, source_logical_path),
            )
        )
    return output_lexical, tuple(items)


def _remove_stage(stage: Path | None) -> None:
    if stage is not None:
        shutil.rmtree(stage, ignore_errors=True)


def materialize_repaired_manifest(
    manifest_path: str | os.PathLike[str],
    dataset_root: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    mode: Literal["copy", "symlink"] = "copy",
) -> MaterializationResult:
    """Transactionally materialize a collision-safe flat image collection.

    Files are staged beside the destination, content-hash verified, and only
    then renamed into place. Symlink targets are relative to each staged link.
    The destination contains ``images/<stable-id>.<safe-ext>`` and a manifest
    whose paths reference those materialized files.
    """

    if mode not in {"copy", "symlink"}:
        raise MaterializationError("materialization mode must be 'copy' or 'symlink'")
    output_path, items = _preflight_materialization(
        manifest_path,
        dataset_root,
        output_dir,
    )
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise MaterializationError("output directory parent cannot be created") from exc

    stage: Path | None = None
    try:
        stage = Path(
            tempfile.mkdtemp(
                prefix=f".{output_path.name}.stage-",
                dir=output_path.parent,
            )
        )
        (stage / "images").mkdir()
        materialized_rows: list[tuple[str, Split, str | None]] = []
        verified_count = 0
        for item in items:
            destination = stage.joinpath(
                *PurePosixPath(item.destination_relative_path).parts
            )
            if mode == "copy":
                shutil.copyfile(item.source_path, destination)
            else:
                relative_target = os.path.relpath(item.source_path, start=destination.parent)
                os.symlink(relative_target, destination)
            if _sha256_file(destination, item.destination_relative_path) != item.source_sha256:
                raise MaterializationError(
                    f"content verification failed for {item.destination_relative_path!r}"
                )
            verified_count += 1
            materialized_rows.append(
                (item.destination_relative_path, item.split, item.label)
            )

        manifest_payload = _render_manifest_bytes(materialized_rows)
        _atomic_write_bytes(stage / "manifest.csv", manifest_payload)
        result = MaterializationResult(
            mode=mode,
            record_count=len(items),
            verified_file_count=verified_count,
            manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        )
        if _path_lexists(output_path):
            raise MaterializationError("output directory appeared during materialization")
        os.rename(stage, output_path)
        stage = None
        return result
    except MaterializationError:
        raise
    except OSError as exc:
        raise MaterializationError(
            "materialization failed; no partial output directory was published"
        ) from exc
    except Exception as exc:
        raise MaterializationError(
            "materialization failed safely before publishing the output directory"
        ) from exc
    finally:
        _remove_stage(stage)



__all__ = [
    "MaterializationError",
    "MaterializationResult",
    "RatioInput",
    "RepairInputError",
    "RepairPlan",
    "jensen_shannon_divergence",
    "materialize_repaired_manifest",
    "repair_splits",
    "write_repaired_manifest",
]
