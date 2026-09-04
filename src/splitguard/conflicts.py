"""Cross-label integrity conflicts for definite families and review pairs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Self

from pydantic import Field, model_validator

from splitguard.schemas import (
    ConflictKind,
    DuplicateClassification,
    DuplicateEdge,
    DuplicateFamily,
    EdgeDecision,
    ImageRecord,
    LabelConflict,
    StrictFrozenModel,
    family_id_for,
)

_CONFLICT_ORDER = {
    ConflictKind.EXACT_DUPLICATE: 0,
    ConflictKind.NEAR_DUPLICATE: 1,
    ConflictKind.SEMANTIC_CANDIDATE: 2,
}


class ConflictAnalysisResult(StrictFrozenModel):
    """Immutable cross-label conflicts and counts by evidence strength."""

    conflicts: tuple[LabelConflict, ...]
    cross_label_conflict_count: Annotated[int, Field(ge=0)]
    definite_conflict_count: Annotated[int, Field(ge=0)]
    exact_conflict_count: Annotated[int, Field(ge=0)]
    near_conflict_count: Annotated[int, Field(ge=0)]
    semantic_review_conflict_count: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def derived_fields_are_consistent(self) -> Self:
        keys = tuple(
            (_CONFLICT_ORDER[conflict.kind], conflict.family_id, conflict.member_ids)
            for conflict in self.conflicts
        )
        if keys != tuple(sorted(set(keys))):
            raise ValueError("conflicts must be unique and in canonical evidence order")

        exact_count = sum(
            conflict.kind is ConflictKind.EXACT_DUPLICATE
            for conflict in self.conflicts
        )
        near_count = sum(
            conflict.kind is ConflictKind.NEAR_DUPLICATE
            for conflict in self.conflicts
        )
        semantic_count = sum(
            conflict.kind is ConflictKind.SEMANTIC_CANDIDATE
            for conflict in self.conflicts
        )
        expected = {
            "cross_label_conflict_count": len(self.conflicts),
            "definite_conflict_count": exact_count + near_count,
            "exact_conflict_count": exact_count,
            "near_conflict_count": near_count,
            "semantic_review_conflict_count": semantic_count,
        }
        for field_name, count in expected.items():
            if getattr(self, field_name) != count:
                raise ValueError(f"{field_name} does not match conflicts")
        return self


def analyze_conflicts(
    records: Sequence[ImageRecord],
    families: Sequence[DuplicateFamily],
    edges: Sequence[DuplicateEdge],
    *,
    review_edges: Sequence[DuplicateEdge] = (),
) -> ConflictAnalysisResult:
    """Classify definite family conflicts and separate semantic review pairs.

    Definite families use only direct, definite, cross-label evidence for their
    strength: exact evidence takes precedence over transformed-duplicate
    evidence.  A semantic-only pair is emitted as a review conflict only when
    both endpoints have different known labels and the pair is not already
    represented by the same definite conflict family.
    """

    records_by_id = _index_records(records)
    canonical_families = _validate_families(families, records_by_id)
    all_edges = tuple(edges) + tuple(review_edges)
    _validate_edge_records(all_edges, records_by_id)

    definite_edges = tuple(
        edge
        for edge in edges
        if edge.decision is EdgeDecision.DEFINITE
        and edge.classification
        in {
            DuplicateClassification.EXACT,
            DuplicateClassification.TRANSFORMED_DUPLICATE,
        }
    )
    conflicts: list[LabelConflict] = []
    definite_conflict_by_member: dict[str, str] = {}
    definite_cross_label_pairs = {
        (edge.left_id, edge.right_id)
        for edge in definite_edges
        if _has_different_known_labels(edge, records_by_id)
    }

    for family in canonical_families:
        labels = _known_labels(family.member_ids, records_by_id)
        if len(labels) < 2:
            continue

        member_set = set(family.member_ids)
        cross_label_edges = tuple(
            edge
            for edge in definite_edges
            if edge.left_id in member_set
            and edge.right_id in member_set
            and _has_different_known_labels(edge, records_by_id)
        )
        if not cross_label_edges:
            # Without a direct cross-label edge, evidence strength is ambiguous.
            # Do not silently promote unrelated same-label evidence.
            continue

        kind = (
            ConflictKind.EXACT_DUPLICATE
            if any(
                edge.classification is DuplicateClassification.EXACT
                for edge in cross_label_edges
            )
            else ConflictKind.NEAR_DUPLICATE
        )
        conflict = LabelConflict(
            family_id=family.family_id,
            member_ids=family.member_ids,
            labels=labels,
            kind=kind,
        )
        conflicts.append(conflict)
        for member_id in family.member_ids:
            definite_conflict_by_member[member_id] = family.family_id

    semantic_edges = _canonical_semantic_reviews(all_edges, records_by_id)
    for edge in semantic_edges:
        pair = (edge.left_id, edge.right_id)
        if pair in definite_cross_label_pairs:
            continue
        left_family = definite_conflict_by_member.get(edge.left_id)
        right_family = definite_conflict_by_member.get(edge.right_id)
        if left_family is not None and left_family == right_family:
            continue
        left_label = records_by_id[edge.left_id].label
        right_label = records_by_id[edge.right_id].label
        if left_label is None or right_label is None:  # Defensive narrowing.
            continue
        labels = tuple(sorted((left_label, right_label)))
        # _canonical_semantic_reviews already requires two different known labels.
        conflicts.append(
            LabelConflict(
                family_id=family_id_for(pair),
                member_ids=pair,
                labels=labels,
                kind=ConflictKind.SEMANTIC_CANDIDATE,
            )
        )

    canonical_conflicts = tuple(
        sorted(
            conflicts,
            key=lambda conflict: (
                _CONFLICT_ORDER[conflict.kind],
                conflict.family_id,
                conflict.member_ids,
            ),
        )
    )
    exact_count = sum(
        conflict.kind is ConflictKind.EXACT_DUPLICATE
        for conflict in canonical_conflicts
    )
    near_count = sum(
        conflict.kind is ConflictKind.NEAR_DUPLICATE
        for conflict in canonical_conflicts
    )
    semantic_count = sum(
        conflict.kind is ConflictKind.SEMANTIC_CANDIDATE
        for conflict in canonical_conflicts
    )
    return ConflictAnalysisResult(
        conflicts=canonical_conflicts,
        cross_label_conflict_count=len(canonical_conflicts),
        definite_conflict_count=exact_count + near_count,
        exact_conflict_count=exact_count,
        near_conflict_count=near_count,
        semantic_review_conflict_count=semantic_count,
    )


def _index_records(records: Sequence[ImageRecord]) -> dict[str, ImageRecord]:
    records_by_id: dict[str, ImageRecord] = {}
    for record in records:
        if record.id in records_by_id:
            raise ValueError(f"duplicate record ID: {record.id}")
        records_by_id[record.id] = record
    return records_by_id


def _validate_families(
    families: Sequence[DuplicateFamily],
    records_by_id: dict[str, ImageRecord],
) -> tuple[DuplicateFamily, ...]:
    seen_family_ids: set[str] = set()
    member_owner: dict[str, str] = {}
    for family in families:
        if family.family_id in seen_family_ids:
            raise ValueError(f"duplicate family ID: {family.family_id}")
        seen_family_ids.add(family.family_id)
        for member_id in family.member_ids:
            if member_id not in records_by_id:
                raise ValueError(f"family references unknown record ID: {member_id}")
            previous_owner = member_owner.get(member_id)
            if previous_owner is not None:
                raise ValueError(
                    f"record {member_id} belongs to multiple families: "
                    f"{previous_owner}, {family.family_id}"
                )
            member_owner[member_id] = family.family_id
    return tuple(sorted(families, key=lambda family: family.family_id))


def _validate_edge_records(
    edges: Sequence[DuplicateEdge], records_by_id: dict[str, ImageRecord]
) -> None:
    for edge in edges:
        for record_id in (edge.left_id, edge.right_id):
            if record_id not in records_by_id:
                raise ValueError(f"edge references unknown record ID: {record_id}")


def _has_different_known_labels(
    edge: DuplicateEdge, records_by_id: dict[str, ImageRecord]
) -> bool:
    left_label = records_by_id[edge.left_id].label
    right_label = records_by_id[edge.right_id].label
    return left_label is not None and right_label is not None and left_label != right_label


def _known_labels(
    member_ids: Sequence[str], records_by_id: dict[str, ImageRecord]
) -> tuple[str, ...]:
    labels: set[str] = set()
    for member_id in member_ids:
        label = records_by_id[member_id].label
        if label is not None:
            labels.add(label)
    return tuple(sorted(labels))


def _is_semantic_only_review(edge: DuplicateEdge) -> bool:
    return (
        edge.decision is EdgeDecision.REVIEW
        and edge.classification is DuplicateClassification.SEMANTIC_CANDIDATE
        and not edge.evidence.exact_match
        and edge.evidence.phash_distance is None
        and edge.evidence.cosine_similarity is not None
    )


def _canonical_semantic_reviews(
    edges: Sequence[DuplicateEdge], records_by_id: dict[str, ImageRecord]
) -> tuple[DuplicateEdge, ...]:
    selected: dict[tuple[str, str], DuplicateEdge] = {}
    for edge in edges:
        if not _is_semantic_only_review(edge) or not _has_different_known_labels(
            edge, records_by_id
        ):
            continue
        pair = (edge.left_id, edge.right_id)
        existing = selected.get(pair)
        if existing is None or _semantic_rank(edge) > _semantic_rank(existing):
            selected[pair] = edge
    return tuple(selected[pair] for pair in sorted(selected))


def _semantic_rank(edge: DuplicateEdge) -> tuple[float, float]:
    similarity = edge.evidence.cosine_similarity
    if similarity is None:  # Defensive; semantic-only callers have a cosine value.
        similarity = -1.0
    return (similarity, edge.confidence)


__all__ = ["ConflictAnalysisResult", "analyze_conflicts"]
