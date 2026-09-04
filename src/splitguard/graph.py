"""Deterministic duplicate-evidence merging and family construction.

Confidence values produced here are ranking scores, not calibrated
probabilities. Exact evidence scores 1.0, perceptual evidence scores
``1 - distance / 64``, and embedding evidence scores ``(cosine + 1) / 2``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated, Self

from pydantic import Field, model_validator

from splitguard.config import PolicyConfig
from splitguard.hashing import ExactDuplicateCluster, PhashCandidatePair
from splitguard.neighbors import NeighborCandidate
from splitguard.schemas import (
    DuplicateClassification,
    DuplicateEdge,
    DuplicateEvidence,
    DuplicateFamily,
    EdgeDecision,
    ImageRecord,
    StableId,
    StrictFrozenModel,
    family_id_for,
)


class GraphInputError(ValueError):
    """Raised when detector evidence does not match the record snapshot."""


class EvidencePolicy(StrictFrozenModel):
    """Strict, serializable policy used to turn evidence into decisions."""

    exact_is_duplicate: bool = True
    phash_is_duplicate: bool = True
    embedding_only_requires_review: bool = True

    @classmethod
    def from_config(cls, config: PolicyConfig) -> EvidencePolicy:
        return cls(
            exact_is_duplicate=config.exact_is_duplicate,
            phash_is_duplicate=config.phash_is_duplicate,
            embedding_only_requires_review=config.embedding_only_requires_review,
        )


class EvidenceGraph(StrictFrozenModel):
    """Canonical evidence edges and definite connected components."""

    record_ids: tuple[StableId, ...]
    phash_threshold: Annotated[int, Field(ge=0, le=64)]
    cosine_threshold: Annotated[float, Field(ge=-1.0, le=1.0)]
    policy: EvidencePolicy
    edges: tuple[DuplicateEdge, ...]
    families: tuple[DuplicateFamily, ...]
    review_edges: tuple[DuplicateEdge, ...]

    @property
    def definite_edges(self) -> tuple[DuplicateEdge, ...]:
        return tuple(edge for edge in self.edges if edge.decision is EdgeDecision.DEFINITE)

    @model_validator(mode="after")
    def canonical_graph(self) -> Self:
        if self.record_ids != tuple(sorted(set(self.record_ids))):
            raise ValueError("record_ids must be sorted and unique")
        edge_pairs = tuple((edge.left_id, edge.right_id) for edge in self.edges)
        if edge_pairs != tuple(sorted(set(edge_pairs))):
            raise ValueError("edges must have unique pairs in canonical order")
        expected_review = tuple(
            edge for edge in self.edges if edge.decision is EdgeDecision.REVIEW
        )
        if self.review_edges != expected_review:
            raise ValueError("review_edges must be the canonical review subset of edges")
        family_ids = tuple(family.family_id for family in self.families)
        if family_ids != tuple(sorted(set(family_ids))):
            raise ValueError("families must have unique IDs in canonical order")
        known_ids = set(self.record_ids)
        if any(not set(family.member_ids) <= known_ids for family in self.families):
            raise ValueError("family members must be present in record_ids")
        return self


@dataclass(slots=True)
class _EvidenceAccumulator:
    exact_match: bool = False
    phash_distance: int | None = None
    cosine_similarity: float | None = None


class _UnionFind:
    def __init__(self, record_ids: Iterable[str]) -> None:
        self._parent = {record_id: record_id for record_id in record_ids}
        self._size = {record_id: 1 for record_id in self._parent}

    def find(self, record_id: str) -> str:
        root = record_id
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[record_id] != record_id:
            parent = self._parent[record_id]
            self._parent[record_id] = root
            record_id = parent
        return root

    def union(self, left_id: str, right_id: str) -> None:
        left_root = self.find(left_id)
        right_root = self.find(right_id)
        if left_root == right_root:
            return
        left_size = self._size[left_root]
        right_size = self._size[right_root]
        if (left_size, left_root) < (right_size, right_root):
            left_root, right_root = right_root, left_root
        self._parent[right_root] = left_root
        self._size[left_root] += self._size[right_root]


def _validate_phash_threshold(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("phash_threshold must be an integer")
    if not 0 <= value <= 64:
        raise ValueError("phash_threshold must be between 0 and 64")
    return value


def _validate_cosine_threshold(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("cosine_threshold must be a number")
    threshold = float(value)
    if not math.isfinite(threshold):
        raise ValueError("cosine_threshold must be finite")
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("cosine_threshold must be between -1 and 1")
    return threshold


def _canonical_pair(left_id: str, right_id: str) -> tuple[str, str]:
    return (left_id, right_id) if left_id < right_id else (right_id, left_id)


def _require_known_pair(
    left_id: str,
    right_id: str,
    known_ids: set[str],
    source: str,
) -> tuple[str, str]:
    if left_id not in known_ids or right_id not in known_ids:
        raise GraphInputError(f"{source} references an unknown record ID")
    if left_id == right_id:
        raise GraphInputError(f"{source} cannot reference the same record twice")
    return _canonical_pair(left_id, right_id)


def _resolve_policy(policy: EvidencePolicy | PolicyConfig | None) -> EvidencePolicy:
    if policy is None:
        return EvidencePolicy()
    if isinstance(policy, EvidencePolicy):
        return policy
    if isinstance(policy, PolicyConfig):
        return EvidencePolicy.from_config(policy)
    raise TypeError("policy must be EvidencePolicy, PolicyConfig, or None")


def _validate_exact_clusters(
    clusters: Iterable[ExactDuplicateCluster],
    records_by_id: dict[str, ImageRecord],
) -> tuple[tuple[ExactDuplicateCluster, ...], dict[str, str]]:
    records_by_sha: dict[str, tuple[str, ...]] = {}
    grouped_ids: dict[str, list[str]] = defaultdict(list)
    for record in records_by_id.values():
        grouped_ids[record.byte_sha256].append(record.id)
    for byte_sha256, record_ids in grouped_ids.items():
        records_by_sha[byte_sha256] = tuple(sorted(record_ids))

    ordered = tuple(sorted(clusters, key=lambda cluster: cluster.cluster_id))
    seen_cluster_ids: set[str] = set()
    cluster_by_member: dict[str, str] = {}
    for cluster in ordered:
        if cluster.cluster_id in seen_cluster_ids:
            raise GraphInputError("exact clusters contain a duplicate cluster ID")
        seen_cluster_ids.add(cluster.cluster_id)
        if any(member_id not in records_by_id for member_id in cluster.member_ids):
            raise GraphInputError("exact cluster references an unknown record ID")
        expected_members = records_by_sha.get(cluster.byte_sha256, ())
        if cluster.member_ids != expected_members:
            raise GraphInputError(
                "exact cluster membership does not match records with its content SHA"
            )
        for member_id in cluster.member_ids:
            if member_id in cluster_by_member:
                raise GraphInputError("a record belongs to more than one exact cluster")
            cluster_by_member[member_id] = cluster.cluster_id
    return ordered, cluster_by_member


def _classify(accumulator: _EvidenceAccumulator) -> DuplicateClassification:
    if accumulator.exact_match:
        return DuplicateClassification.EXACT
    if accumulator.phash_distance is not None:
        return DuplicateClassification.TRANSFORMED_DUPLICATE
    if accumulator.cosine_similarity is None:
        raise RuntimeError("evidence accumulator contains no accepted evidence")
    return DuplicateClassification.SEMANTIC_CANDIDATE


def _decision(
    classification: DuplicateClassification,
    policy: EvidencePolicy,
) -> EdgeDecision:
    if classification is DuplicateClassification.EXACT:
        is_definite = policy.exact_is_duplicate
    elif classification is DuplicateClassification.TRANSFORMED_DUPLICATE:
        is_definite = policy.phash_is_duplicate
    else:
        is_definite = not policy.embedding_only_requires_review
    return EdgeDecision.DEFINITE if is_definite else EdgeDecision.REVIEW


def _confidence(
    classification: DuplicateClassification,
    accumulator: _EvidenceAccumulator,
) -> float:
    if classification is DuplicateClassification.EXACT:
        return 1.0
    if classification is DuplicateClassification.TRANSFORMED_DUPLICATE:
        if accumulator.phash_distance is None:
            raise RuntimeError("transformed classification is missing pHash evidence")
        return 1.0 - accumulator.phash_distance / 64.0
    if accumulator.cosine_similarity is None:
        raise RuntimeError("semantic classification is missing cosine evidence")
    return (accumulator.cosine_similarity + 1.0) / 2.0


def _build_families(
    record_ids: tuple[str, ...],
    edges: tuple[DuplicateEdge, ...],
) -> tuple[DuplicateFamily, ...]:
    union_find = _UnionFind(record_ids)
    definite_edges = tuple(edge for edge in edges if edge.decision is EdgeDecision.DEFINITE)
    for edge in definite_edges:
        union_find.union(edge.left_id, edge.right_id)

    members_by_root: dict[str, list[str]] = defaultdict(list)
    for record_id in record_ids:
        members_by_root[union_find.find(record_id)].append(record_id)

    families: list[DuplicateFamily] = []
    for members in members_by_root.values():
        if len(members) < 2:
            continue
        member_ids = tuple(sorted(members))
        member_set = set(member_ids)
        edge_count = sum(
            edge.left_id in member_set and edge.right_id in member_set
            for edge in definite_edges
        )
        families.append(
            DuplicateFamily(
                family_id=family_id_for(member_ids),
                member_ids=member_ids,
                edge_count=edge_count,
            )
        )
    return tuple(sorted(families, key=lambda family: family.family_id))


def build_evidence_graph(
    records: Iterable[ImageRecord],
    exact_clusters: Iterable[ExactDuplicateCluster] = (),
    phash_candidates: Iterable[PhashCandidatePair] = (),
    neighbor_candidates: Iterable[NeighborCandidate] = (),
    *,
    phash_threshold: int,
    cosine_threshold: float,
    policy: EvidencePolicy | PolicyConfig | None = None,
) -> EvidenceGraph:
    """Merge detector evidence and union only policy-approved definite edges.

    Exact clusters are represented by deterministic anchor-to-member edges,
    avoiding quadratic expansion for large identical-file groups. Candidate
    pairs between members of the same supplied exact cluster still receive
    exact classification when their other evidence is merged.
    """

    accepted_phash_threshold = _validate_phash_threshold(phash_threshold)
    accepted_cosine_threshold = _validate_cosine_threshold(cosine_threshold)
    accepted_policy = _resolve_policy(policy)

    record_snapshot = tuple(records)
    records_by_id = {record.id: record for record in record_snapshot}
    if len(records_by_id) != len(record_snapshot):
        raise GraphInputError("records must contain unique IDs")
    record_ids = tuple(sorted(records_by_id))
    known_ids = set(record_ids)
    ordered_clusters, exact_cluster_by_member = _validate_exact_clusters(
        exact_clusters,
        records_by_id,
    )

    evidence_by_pair: dict[tuple[str, str], _EvidenceAccumulator] = {}

    def accumulator_for(left_id: str, right_id: str, source: str) -> _EvidenceAccumulator:
        pair = _require_known_pair(left_id, right_id, known_ids, source)
        accumulator = evidence_by_pair.setdefault(pair, _EvidenceAccumulator())
        left_cluster = exact_cluster_by_member.get(pair[0])
        if left_cluster is not None and left_cluster == exact_cluster_by_member.get(pair[1]):
            accumulator.exact_match = True
        return accumulator

    for cluster in ordered_clusters:
        anchor_id = cluster.member_ids[0]
        for member_id in cluster.member_ids[1:]:
            accumulator_for(anchor_id, member_id, "exact cluster").exact_match = True

    for phash_candidate in phash_candidates:
        accumulator = accumulator_for(
            phash_candidate.left_id,
            phash_candidate.right_id,
            "pHash candidate",
        )
        if phash_candidate.distance <= accepted_phash_threshold and (
            accumulator.phash_distance is None
            or phash_candidate.distance < accumulator.phash_distance
        ):
            accumulator.phash_distance = phash_candidate.distance

    for neighbor_candidate in neighbor_candidates:
        accumulator = accumulator_for(
            neighbor_candidate.left_id,
            neighbor_candidate.right_id,
            "embedding candidate",
        )
        if neighbor_candidate.cosine_similarity >= accepted_cosine_threshold and (
            accumulator.cosine_similarity is None
            or neighbor_candidate.cosine_similarity > accumulator.cosine_similarity
        ):
            accumulator.cosine_similarity = neighbor_candidate.cosine_similarity

    edges: list[DuplicateEdge] = []
    for (left_id, right_id), accumulator in sorted(evidence_by_pair.items()):
        if (
            not accumulator.exact_match
            and accumulator.phash_distance is None
            and accumulator.cosine_similarity is None
        ):
            continue
        classification = _classify(accumulator)
        edges.append(
            DuplicateEdge(
                left_id=left_id,
                right_id=right_id,
                evidence=DuplicateEvidence(
                    exact_match=accumulator.exact_match,
                    phash_distance=accumulator.phash_distance,
                    cosine_similarity=accumulator.cosine_similarity,
                ),
                classification=classification,
                decision=_decision(classification, accepted_policy),
                confidence=_confidence(classification, accumulator),
            )
        )

    canonical_edges = tuple(edges)
    review_edges = tuple(
        edge for edge in canonical_edges if edge.decision is EdgeDecision.REVIEW
    )
    return EvidenceGraph(
        record_ids=record_ids,
        phash_threshold=accepted_phash_threshold,
        cosine_threshold=accepted_cosine_threshold,
        policy=accepted_policy,
        edges=canonical_edges,
        families=_build_families(record_ids, canonical_edges),
        review_edges=review_edges,
    )


__all__ = [
    "EvidenceGraph",
    "EvidencePolicy",
    "GraphInputError",
    "build_evidence_graph",
]
