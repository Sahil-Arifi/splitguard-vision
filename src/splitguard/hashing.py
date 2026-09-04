"""Deterministic image fingerprint grouping and indexed similarity search.

This phase implements exact binary duplicate grouping. Perceptual hashing and
its indexed Hamming-distance candidate search will extend this module without
changing the exact-duplicate contract.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Annotated, Self

from pydantic import Field, field_validator, model_validator

from splitguard.schemas import ImageRecord, Split, SplitBoundary, StableId, StrictFrozenModel

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_EXACT_CLUSTER_ID_PATTERN = r"^exact_[0-9a-f]{64}$"
_SPLIT_ORDER = {split: index for index, split in enumerate(Split)}
_EVALUATION_BOUNDARIES = (
    (Split.TRAIN, Split.VAL),
    (Split.TRAIN, Split.TEST),
    (Split.VAL, Split.TEST),
)


def _exact_cluster_id(byte_sha256: str) -> str:
    """Derive an exact-cluster identifier from the complete content digest."""

    return f"exact_{byte_sha256}"


def _boundaries_for(splits: Sequence[Split]) -> tuple[SplitBoundary, ...]:
    represented = set(splits)
    return tuple(
        SplitBoundary(left=left, right=right)
        for left, right in _EVALUATION_BOUNDARIES
        if left in represented and right in represented
    )


class ExactDuplicateCluster(StrictFrozenModel):
    """Canonical metadata for files with identical binary contents.

    ``total_duplicate_bytes`` is the sum of every member file's byte size,
    including the first member. It represents the total storage occupied by
    the cluster rather than an estimate of reclaimable storage.
    """

    cluster_id: Annotated[str, Field(pattern=_EXACT_CLUSTER_ID_PATTERN)]
    byte_sha256: Annotated[str, Field(pattern=_SHA256_PATTERN)]
    member_ids: Annotated[tuple[StableId, ...], Field(min_length=2)]
    member_count: Annotated[int, Field(ge=2)]
    splits: Annotated[tuple[Split, ...], Field(min_length=1)]
    labels: tuple[str, ...] = ()
    boundaries: tuple[SplitBoundary, ...] = ()
    total_duplicate_bytes: Annotated[int, Field(ge=0)]
    crosses_evaluation_boundary: bool

    @field_validator("member_ids")
    @classmethod
    def canonical_members(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if values != tuple(sorted(set(values))):
            raise ValueError("member_ids must be sorted and unique")
        return values

    @field_validator("splits")
    @classmethod
    def canonical_splits(cls, values: tuple[Split, ...]) -> tuple[Split, ...]:
        if values != tuple(sorted(set(values), key=_SPLIT_ORDER.__getitem__)):
            raise ValueError("splits must be sorted and unique")
        return values

    @field_validator("labels")
    @classmethod
    def canonical_labels(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if values != tuple(sorted(set(values))):
            raise ValueError("labels must be sorted and unique")
        return values

    @model_validator(mode="after")
    def consistent_metadata(self) -> Self:
        if self.cluster_id != _exact_cluster_id(self.byte_sha256):
            raise ValueError("cluster_id must contain the complete content SHA-256")
        if self.member_count != len(self.member_ids):
            raise ValueError("member_count must match member_ids")
        expected_boundaries = _boundaries_for(self.splits)
        if self.boundaries != expected_boundaries:
            raise ValueError("boundaries must match the represented protected splits")
        if self.crosses_evaluation_boundary != bool(expected_boundaries):
            raise ValueError("crosses_evaluation_boundary must agree with boundaries")
        return self


def group_exact_duplicates(records: Iterable[ImageRecord]) -> tuple[ExactDuplicateCluster, ...]:
    """Group records with the same SHA-256 digest into canonical clusters.

    Only groups containing at least two records are returned. Results and all
    nested collections are ordered deterministically, independent of input
    order. Evaluation boundaries are limited to train-validation, train-test,
    and validation-test; custom splits remain visible in ``splits`` but do not
    create protected-boundary metadata.
    """

    records_by_sha: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        records_by_sha[record.byte_sha256].append(record)

    clusters: list[ExactDuplicateCluster] = []
    for byte_sha256, matching_records in records_by_sha.items():
        if len(matching_records) < 2:
            continue

        member_ids = tuple(sorted(record.id for record in matching_records))
        splits = tuple(
            sorted({record.split for record in matching_records}, key=_SPLIT_ORDER.__getitem__)
        )
        labels = tuple(
            sorted({record.label for record in matching_records if record.label is not None})
        )
        boundaries = _boundaries_for(splits)
        clusters.append(
            ExactDuplicateCluster(
                cluster_id=_exact_cluster_id(byte_sha256),
                byte_sha256=byte_sha256,
                member_ids=member_ids,
                member_count=len(member_ids),
                splits=splits,
                labels=labels,
                boundaries=boundaries,
                total_duplicate_bytes=sum(record.byte_size for record in matching_records),
                crosses_evaluation_boundary=bool(boundaries),
            )
        )

    return tuple(sorted(clusters, key=lambda cluster: cluster.cluster_id))


__all__ = ["ExactDuplicateCluster", "group_exact_duplicates"]
