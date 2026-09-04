"""Cross-split leakage analysis for definite duplicate families.

The analyzer deliberately keeps semantic-only neighbor suggestions out of the
definite leakage totals.  Those edges remain available as review candidates so
callers can surface them without overstating what the detector established.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Annotated, Self

from pydantic import Field, model_validator

from splitguard.schemas import (
    DuplicateClassification,
    DuplicateEdge,
    DuplicateFamily,
    EdgeDecision,
    ImageRecord,
    LeakageGroup,
    Split,
    SplitBoundary,
    StrictFrozenModel,
)

_SPLIT_ORDER = {split: index for index, split in enumerate(Split)}
_EVIDENCE_ORDER = {
    DuplicateClassification.EXACT: 0,
    DuplicateClassification.TRANSFORMED_DUPLICATE: 1,
}

PROTECTED_EVALUATION_BOUNDARIES = (
    SplitBoundary(left=Split.TRAIN, right=Split.VAL),
    SplitBoundary(left=Split.TRAIN, right=Split.TEST),
    SplitBoundary(left=Split.VAL, right=Split.TEST),
)


class LeakageBoundaryCount(StrictFrozenModel):
    """Number of definite leakage families crossing one protected boundary."""

    boundary: SplitBoundary
    leakage_group_count: Annotated[int, Field(ge=0)]


class LeakageAnalysisResult(StrictFrozenModel):
    """Immutable leakage groups, review candidates, and aggregate metrics."""

    leakage_groups: tuple[LeakageGroup, ...]
    boundary_group_counts: tuple[LeakageBoundaryCount, ...]
    semantic_review_edges: tuple[DuplicateEdge, ...]
    contaminated_image_ids: tuple[str, ...]
    evaluation_image_ids: tuple[str, ...]
    contaminated_evaluation_image_ids: tuple[str, ...]
    leakage_group_count: Annotated[int, Field(ge=0)]
    contaminated_image_count: Annotated[int, Field(ge=0)]
    evaluation_image_count: Annotated[int, Field(ge=0)]
    contaminated_evaluation_image_count: Annotated[int, Field(ge=0)]
    contaminated_evaluation_fraction: Annotated[float, Field(ge=0.0, le=1.0)]
    exact_leakage_group_count: Annotated[int, Field(ge=0)]
    perceptual_leakage_group_count: Annotated[int, Field(ge=0)]
    embedding_only_review_count: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def derived_fields_are_consistent(self) -> Self:
        group_keys = tuple(group.family_id for group in self.leakage_groups)
        if group_keys != tuple(sorted(set(group_keys))):
            raise ValueError("leakage_groups must have unique family IDs in canonical order")

        expected_boundary_counts = tuple(
            LeakageBoundaryCount(
                boundary=boundary,
                leakage_group_count=sum(
                    boundary in group.boundaries for group in self.leakage_groups
                ),
            )
            for boundary in PROTECTED_EVALUATION_BOUNDARIES
        )
        if self.boundary_group_counts != expected_boundary_counts:
            raise ValueError("boundary_group_counts do not match leakage_groups")

        review_pairs = tuple(
            (edge.left_id, edge.right_id) for edge in self.semantic_review_edges
        )
        if review_pairs != tuple(sorted(set(review_pairs))):
            raise ValueError("semantic_review_edges must have unique pairs in canonical order")
        if any(not _is_semantic_only_review(edge) for edge in self.semantic_review_edges):
            raise ValueError("semantic_review_edges must contain semantic-only review edges")

        for field_name in (
            "contaminated_image_ids",
            "evaluation_image_ids",
            "contaminated_evaluation_image_ids",
        ):
            values = getattr(self, field_name)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{field_name} must be sorted and unique")
        if not set(self.contaminated_evaluation_image_ids).issubset(
            self.contaminated_image_ids
        ):
            raise ValueError("contaminated evaluation IDs must be contaminated image IDs")
        if not set(self.contaminated_evaluation_image_ids).issubset(
            self.evaluation_image_ids
        ):
            raise ValueError("contaminated evaluation IDs must be evaluation image IDs")

        expected_counts = {
            "leakage_group_count": len(self.leakage_groups),
            "contaminated_image_count": len(self.contaminated_image_ids),
            "evaluation_image_count": len(self.evaluation_image_ids),
            "contaminated_evaluation_image_count": len(
                self.contaminated_evaluation_image_ids
            ),
            "exact_leakage_group_count": sum(
                group.strongest_evidence is DuplicateClassification.EXACT
                for group in self.leakage_groups
            ),
            "perceptual_leakage_group_count": sum(
                group.strongest_evidence
                is DuplicateClassification.TRANSFORMED_DUPLICATE
                for group in self.leakage_groups
            ),
            "embedding_only_review_count": len(self.semantic_review_edges),
        }
        for field_name, expected in expected_counts.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"{field_name} does not match its source collection")

        expected_fraction = (
            self.contaminated_evaluation_image_count / self.evaluation_image_count
            if self.evaluation_image_count
            else 0.0
        )
        if not math.isclose(
            self.contaminated_evaluation_fraction,
            expected_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("contaminated_evaluation_fraction does not match the counts")
        return self


def analyze_leakage(
    records: Sequence[ImageRecord],
    families: Sequence[DuplicateFamily],
    edges: Sequence[DuplicateEdge],
    *,
    review_edges: Sequence[DuplicateEdge] = (),
) -> LeakageAnalysisResult:
    """Analyze definite family leakage and separate semantic review candidates.

    ``families`` are expected to be connected components formed from definite
    exact or transformed-duplicate edges.  A component is a definite leakage
    group only when it both crosses a protected boundary and has at least one
    supplied definite exact/transformed edge.  Semantic-only review edges never
    contribute members or counts to definite leakage.
    """

    records_by_id = _index_records(records)
    canonical_families = _validate_families(families, records_by_id)
    all_edges = tuple(edges) + tuple(review_edges)
    _validate_edge_records(all_edges, records_by_id)

    definite_edges = tuple(
        edge
        for edge in edges
        if edge.decision is EdgeDecision.DEFINITE
        and edge.classification in _EVIDENCE_ORDER
    )
    groups: list[LeakageGroup] = []
    for family in canonical_families:
        member_set = set(family.member_ids)
        member_records = tuple(records_by_id[member_id] for member_id in family.member_ids)
        splits = tuple(
            sorted({record.split for record in member_records}, key=_SPLIT_ORDER.__getitem__)
        )
        boundaries = _boundaries_for_splits(set(splits))
        if not boundaries:
            continue

        family_edges = tuple(
            edge
            for edge in definite_edges
            if edge.left_id in member_set and edge.right_id in member_set
        )
        if not family_edges:
            # A semantic-only component is review material, not definite leakage.
            continue
        evidence_types = tuple(
            sorted(
                {edge.classification for edge in family_edges},
                key=_EVIDENCE_ORDER.__getitem__,
            )
        )
        labels = tuple(
            sorted({record.label for record in member_records if record.label is not None})
        )
        groups.append(
            LeakageGroup(
                family_id=family.family_id,
                member_ids=family.member_ids,
                splits=splits,
                boundaries=boundaries,
                labels=labels,
                evidence_types=evidence_types,
                strongest_evidence=evidence_types[0],
                label_conflict=len(labels) > 1,
            )
        )

    leakage_groups = tuple(sorted(groups, key=lambda group: group.family_id))
    boundary_counts = tuple(
        LeakageBoundaryCount(
            boundary=boundary,
            leakage_group_count=sum(boundary in group.boundaries for group in leakage_groups),
        )
        for boundary in PROTECTED_EVALUATION_BOUNDARIES
    )

    definite_pairs = {
        (edge.left_id, edge.right_id)
        for edge in definite_edges
    }
    semantic_reviews = _canonical_semantic_reviews(
        all_edges,
        records_by_id,
        excluded_pairs=definite_pairs,
    )
    contaminated_ids = tuple(
        sorted({member_id for group in leakage_groups for member_id in group.member_ids})
    )
    evaluation_ids = tuple(
        sorted(
            record.id
            for record in records_by_id.values()
            if record.split in {Split.VAL, Split.TEST}
        )
    )
    evaluation_id_set = set(evaluation_ids)
    contaminated_evaluation_ids = tuple(
        member_id for member_id in contaminated_ids if member_id in evaluation_id_set
    )
    evaluation_count = len(evaluation_ids)
    contaminated_evaluation_count = len(contaminated_evaluation_ids)

    return LeakageAnalysisResult(
        leakage_groups=leakage_groups,
        boundary_group_counts=boundary_counts,
        semantic_review_edges=semantic_reviews,
        contaminated_image_ids=contaminated_ids,
        evaluation_image_ids=evaluation_ids,
        contaminated_evaluation_image_ids=contaminated_evaluation_ids,
        leakage_group_count=len(leakage_groups),
        contaminated_image_count=len(contaminated_ids),
        evaluation_image_count=evaluation_count,
        contaminated_evaluation_image_count=contaminated_evaluation_count,
        contaminated_evaluation_fraction=(
            contaminated_evaluation_count / evaluation_count if evaluation_count else 0.0
        ),
        exact_leakage_group_count=sum(
            group.strongest_evidence is DuplicateClassification.EXACT
            for group in leakage_groups
        ),
        perceptual_leakage_group_count=sum(
            group.strongest_evidence
            is DuplicateClassification.TRANSFORMED_DUPLICATE
            for group in leakage_groups
        ),
        embedding_only_review_count=len(semantic_reviews),
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


def _boundaries_for_splits(splits: set[Split]) -> tuple[SplitBoundary, ...]:
    return tuple(
        boundary
        for boundary in PROTECTED_EVALUATION_BOUNDARIES
        if boundary.left in splits and boundary.right in splits
    )


def _is_semantic_only_review(edge: DuplicateEdge) -> bool:
    return (
        edge.decision is EdgeDecision.REVIEW
        and edge.classification is DuplicateClassification.SEMANTIC_CANDIDATE
        and not edge.evidence.exact_match
        and edge.evidence.phash_distance is None
        and edge.evidence.cosine_similarity is not None
    )


def _canonical_semantic_reviews(
    edges: Sequence[DuplicateEdge],
    records_by_id: dict[str, ImageRecord],
    *,
    excluded_pairs: set[tuple[str, str]],
) -> tuple[DuplicateEdge, ...]:
    selected: dict[tuple[str, str], DuplicateEdge] = {}
    for edge in edges:
        pair = (edge.left_id, edge.right_id)
        if pair in excluded_pairs or not _is_semantic_only_review(edge):
            continue
        left_split = records_by_id[edge.left_id].split
        right_split = records_by_id[edge.right_id].split
        if not _boundaries_for_splits({left_split, right_split}):
            continue
        existing = selected.get(pair)
        if existing is None or _semantic_rank(edge) > _semantic_rank(existing):
            selected[pair] = edge
    return tuple(selected[pair] for pair in sorted(selected))


def _semantic_rank(edge: DuplicateEdge) -> tuple[float, float]:
    similarity = edge.evidence.cosine_similarity
    if similarity is None:  # Defensive; semantic-only callers have a cosine value.
        similarity = -1.0
    return (similarity, edge.confidence)


__all__ = [
    "PROTECTED_EVALUATION_BOUNDARIES",
    "LeakageAnalysisResult",
    "LeakageBoundaryCount",
    "analyze_leakage",
]
