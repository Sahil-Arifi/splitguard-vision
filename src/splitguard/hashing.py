"""Deterministic exact and perceptual image fingerprint search."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Self

import numpy as np
from PIL import Image, ImageOps
from pydantic import Field, field_validator, model_validator
from scipy.fft import dctn

from splitguard.schemas import ImageRecord, Split, SplitBoundary, StableId, StrictFrozenModel

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_EXACT_CLUSTER_ID_PATTERN = r"^exact_[0-9a-f]{64}$"
_STABLE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[0-9a-f]{16,64}$")
_UINT64_MAX = (1 << 64) - 1
_PHASH_IMAGE_SIZE = 32
_PHASH_LOW_FREQUENCY_SIZE = 8
_PHASH_ALPHA_BACKGROUND = (255, 255, 255, 255)
PHASH_ALGORITHM_ID = "phash64-dct-v1"
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


class PhashMatch(StrictFrozenModel):
    """One deterministic BK-tree query result."""

    record_id: StableId
    phash: Annotated[int, Field(ge=0, le=_UINT64_MAX)]
    distance: Annotated[int, Field(ge=0, le=64)]


class PhashCandidatePair(StrictFrozenModel):
    """A canonical pair whose perceptual hashes are within a search radius."""

    left_id: StableId
    right_id: StableId
    distance: Annotated[int, Field(ge=0, le=64)]

    @model_validator(mode="after")
    def canonical_pair(self) -> Self:
        if self.left_id >= self.right_id:
            raise ValueError("candidate identifiers must satisfy left_id < right_id")
        return self


@dataclass(slots=True)
class _BKNode:
    phash: int
    record_ids: set[str] = field(default_factory=set)
    children: dict[int, _BKNode] = field(default_factory=dict)


def _validate_uint64(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= _UINT64_MAX:
        raise ValueError(f"{name} must be an unsigned 64-bit integer")
    return value


def _validate_radius(radius: int) -> int:
    if isinstance(radius, bool) or not isinstance(radius, int):
        raise TypeError("radius must be an integer")
    if not 0 <= radius <= 64:
        raise ValueError("radius must be between 0 and 64")
    return radius


def _validate_record_id(record_id: str) -> str:
    if not isinstance(record_id, str):
        raise TypeError("record_id must be a string")
    if _STABLE_ID_RE.fullmatch(record_id) is None:
        raise ValueError("record_id must be a canonical stable identifier")
    return record_id


def compute_phash(image: Image.Image) -> int:
    """Compute SplitGuard's documented 64-bit perceptual image hash.

    The image is EXIF-transposed and alpha-composited over opaque white before
    normalization through RGB and grayscale, then resized to 32 x 32 with
    Lanczos resampling. An orthonormal two-dimensional type-II DCT is computed
    in float64. The row-major top-left 8 x 8 block is packed MSB-first: the DC
    bit (the MSB) is always zero, while the 63 AC coefficients use a strict
    ``coefficient > AC median`` tie rule.
    """

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a Pillow Image")

    corrected = ImageOps.exif_transpose(image)
    has_transparency = "A" in corrected.getbands() or "transparency" in corrected.info
    if has_transparency:
        rgba = corrected.convert("RGBA")
        background = Image.new("RGBA", rgba.size, color=_PHASH_ALPHA_BACKGROUND)
        rgb = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        rgb = corrected.convert("RGB")
    grayscale = rgb.convert("L")
    resized = grayscale.resize(
        (_PHASH_IMAGE_SIZE, _PHASH_IMAGE_SIZE),
        resample=Image.Resampling.LANCZOS,
    )
    pixels = np.asarray(resized, dtype=np.float64)
    coefficients = dctn(pixels, type=2, axes=(0, 1), norm="ortho")
    low_frequency = coefficients[
        :_PHASH_LOW_FREQUENCY_SIZE, :_PHASH_LOW_FREQUENCY_SIZE
    ].reshape(-1)
    ac_median = float(np.median(low_frequency[1:]))

    fingerprint = 0
    for index, coefficient in enumerate(low_frequency):
        fingerprint <<= 1
        if index != 0 and coefficient > ac_median:
            fingerprint |= 1
    return fingerprint


def hamming_distance(left: int, right: int) -> int:
    """Return bitwise Hamming distance after strict uint64 validation."""

    left = _validate_uint64(left, "left")
    right = _validate_uint64(right, "right")
    return (left ^ right).bit_count()


class BKTree:
    """BK-tree over uint64 perceptual hashes with canonical record IDs.

    Multiple records may have the same hash and are retained at one tree node.
    Query output is always sorted by distance and record ID. Record IDs are
    globally unique within a tree, so every repeated ID is rejected.
    """

    def __init__(self) -> None:
        self._root: _BKNode | None = None
        self._hash_by_id: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._hash_by_id)

    def add(self, phash: int, record_id: str) -> None:
        """Insert a hash and ID without discarding equal hash values."""

        phash = _validate_uint64(phash, "phash")
        record_id = _validate_record_id(record_id)
        if record_id in self._hash_by_id:
            raise ValueError("record_id is already indexed")

        self._hash_by_id[record_id] = phash
        if self._root is None:
            self._root = _BKNode(phash=phash, record_ids={record_id})
            return

        node = self._root
        while True:
            distance = hamming_distance(phash, node.phash)
            if distance == 0:
                node.record_ids.add(record_id)
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _BKNode(phash=phash, record_ids={record_id})
                return
            node = child

    @classmethod
    def from_items(cls, items: Iterable[tuple[int, str]]) -> Self:
        """Build a tree with an input-order-independent insertion order."""

        normalized: list[tuple[int, str]] = []
        seen_ids: set[str] = set()
        for phash, record_id in items:
            phash = _validate_uint64(phash, "phash")
            record_id = _validate_record_id(record_id)
            if record_id in seen_ids:
                raise ValueError("record_id is duplicated in construction items")
            seen_ids.add(record_id)
            normalized.append((phash, record_id))

        tree = cls()
        for phash, record_id in sorted(normalized, key=lambda item: (item[0], item[1])):
            tree.add(phash, record_id)
        return tree

    def search(
        self,
        phash: int,
        radius: int,
        *,
        exclude_id: str | None = None,
    ) -> tuple[PhashMatch, ...]:
        """Find all stored hashes within ``radius`` of the query."""

        phash = _validate_uint64(phash, "phash")
        radius = _validate_radius(radius)
        if exclude_id is not None:
            exclude_id = _validate_record_id(exclude_id)
        if self._root is None:
            return ()

        matches: list[PhashMatch] = []
        pending = [self._root]
        while pending:
            node = pending.pop()
            distance = hamming_distance(phash, node.phash)
            if distance <= radius:
                matches.extend(
                    PhashMatch(record_id=record_id, phash=node.phash, distance=distance)
                    for record_id in sorted(node.record_ids)
                    if record_id != exclude_id
                )

            minimum_child_distance = max(0, distance - radius)
            maximum_child_distance = min(64, distance + radius)
            child_distances = sorted(node.children, reverse=True)
            pending.extend(
                node.children[child_distance]
                for child_distance in child_distances
                if minimum_child_distance <= child_distance <= maximum_child_distance
            )

        return tuple(sorted(matches, key=lambda match: (match.distance, match.record_id)))


def _phash_entries(records: Iterable[ImageRecord]) -> tuple[tuple[str, int], ...]:
    hashes_by_id: dict[str, int] = {}
    for record in records:
        if record.phash is None:
            continue
        if record.id in hashes_by_id:
            raise ValueError("record_id is duplicated in perceptual-hash records")
        hashes_by_id[record.id] = record.phash
    return tuple(sorted(hashes_by_id.items()))


def _candidate_pair(left_id: str, right_id: str, distance: int) -> PhashCandidatePair:
    canonical_left, canonical_right = sorted((left_id, right_id))
    return PhashCandidatePair(
        left_id=canonical_left,
        right_id=canonical_right,
        distance=distance,
    )


def indexed_phash_pairs(
    records: Iterable[ImageRecord], radius: int
) -> tuple[PhashCandidatePair, ...]:
    """Find pHash candidate pairs through a BK-tree rather than all-pairs search."""

    radius = _validate_radius(radius)
    entries = _phash_entries(records)
    tree = BKTree()
    candidates: list[PhashCandidatePair] = []

    # Hash-first insertion makes the index shape deterministic for the same
    # record set, while querying before insertion emits each pair exactly once.
    for record_id, phash in sorted(entries, key=lambda entry: (entry[1], entry[0])):
        candidates.extend(
            _candidate_pair(record_id, match.record_id, match.distance)
            for match in tree.search(phash, radius)
        )
        tree.add(phash, record_id)

    return tuple(
        sorted(candidates, key=lambda pair: (pair.left_id, pair.right_id, pair.distance))
    )


def brute_force_phash_pairs(
    records: Iterable[ImageRecord], radius: int
) -> tuple[PhashCandidatePair, ...]:
    """Reference all-pairs pHash search for tests and small benchmarks only."""

    radius = _validate_radius(radius)
    entries = _phash_entries(records)
    candidates: list[PhashCandidatePair] = []
    for left_index, (left_id, left_hash) in enumerate(entries):
        for right_id, right_hash in entries[left_index + 1 :]:
            distance = hamming_distance(left_hash, right_hash)
            if distance <= radius:
                candidates.append(_candidate_pair(left_id, right_id, distance))
    return tuple(candidates)


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


__all__ = [
    "PHASH_ALGORITHM_ID",
    "BKTree",
    "ExactDuplicateCluster",
    "PhashCandidatePair",
    "PhashMatch",
    "brute_force_phash_pairs",
    "compute_phash",
    "group_exact_duplicates",
    "hamming_distance",
    "indexed_phash_pairs",
]
