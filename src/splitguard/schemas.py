"""Immutable domain contracts for SplitGuard Vision.

The schemas in this module are deliberately free of I/O and detector
implementations.  They define the stable boundary between ingestion,
detection, repair, benchmarking, and reporting while leaving algorithms free
to evolve behind those boundaries.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION: Literal["1.0"] = "1.0"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")
_IMMUTABLE_REVISION_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_NAMESPACE_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_STABLE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[0-9a-f]{16,64}$")

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
StableId = Annotated[
    str,
    Field(pattern=r"^[a-z][a-z0-9_]{0,31}_[0-9a-f]{16,64}$"),
]
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class StrictFrozenModel(BaseModel):
    """Base class for immutable, non-coercing public contracts."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    CUSTOM = "custom"


class ValidationIssueCode(StrEnum):
    INVALID_MANIFEST = "invalid_manifest"
    MISSING_PATH = "missing_path"
    NOT_A_FILE = "not_a_file"
    PATH_ESCAPE = "path_escape"
    DUPLICATE_PATH = "duplicate_path"
    MALFORMED_IMAGE = "malformed_image"
    UNSUPPORTED_FORMAT = "unsupported_format"
    IMAGE_TOO_LARGE = "image_too_large"
    IO_ERROR = "io_error"


class DuplicateClassification(StrEnum):
    EXACT = "exact"
    TRANSFORMED_DUPLICATE = "transformed_duplicate"
    SEMANTIC_CANDIDATE = "semantic_candidate"


class EdgeDecision(StrEnum):
    DEFINITE = "definite"
    REVIEW = "review"


class ConflictKind(StrEnum):
    EXACT_DUPLICATE = "exact_duplicate"
    NEAR_DUPLICATE = "near_duplicate"
    SEMANTIC_CANDIDATE = "semantic_candidate"


class TrainingCondition(StrEnum):
    CONTAMINATED = "contaminated"
    REPAIRED = "repaired"


_SPLIT_ORDER = {split: index for index, split in enumerate(Split)}
_CLASSIFICATION_ORDER = {
    DuplicateClassification.EXACT: 0,
    DuplicateClassification.TRANSFORMED_DUPLICATE: 1,
    DuplicateClassification.SEMANTIC_CANDIDATE: 2,
}


def stable_id(namespace: str, *parts: str, digest_length: int = 24) -> str:
    """Return a machine-independent, delimiter-safe deterministic identifier."""

    if _NAMESPACE_RE.fullmatch(namespace) is None:
        raise ValueError("namespace must be lowercase snake_case")
    if not 16 <= digest_length <= 64:
        raise ValueError("digest_length must be between 16 and 64")
    if not parts:
        raise ValueError("at least one identifier part is required")

    digest = hashlib.sha256()
    for part in (namespace, *parts):
        if not isinstance(part, str):
            raise TypeError("identifier parts must be strings")
        if "\x00" in part:
            raise ValueError("identifier parts cannot contain NUL characters")
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return f"{namespace}_{digest.hexdigest()[:digest_length]}"


def family_id_for(member_ids: Sequence[str]) -> str:
    """Return the canonical identifier for a duplicate family."""

    members = tuple(member_ids)
    if not members:
        raise ValueError("a duplicate family needs at least one member")
    if tuple(sorted(set(members))) != members:
        raise ValueError("member_ids must be sorted and unique")
    return stable_id("family", *members)


def _json_compatible(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite float at {path}")
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object key at {path} is not a string")
            _json_compatible(item, f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _json_compatible(item, f"{path}[{index}]")
        return
    raise TypeError(f"unsupported canonical JSON value at {path}: {type(value).__name__}")


def canonical_json(value: BaseModel | Mapping[str, Any] | Sequence[Any]) -> str:
    """Serialize a contract deterministically and reject NaN/Infinity."""

    payload: Any
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json", by_alias=True, exclude_none=False)
    else:
        payload = value
    _json_compatible(payload)
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_sha256(value: BaseModel | Mapping[str, Any] | Sequence[Any]) -> str:
    """Hash the canonical UTF-8 JSON representation of a contract."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _validate_relative_path(value: str) -> str:
    if not value:
        raise ValueError("path cannot be empty")
    if "\x00" in value or "\\" in value:
        raise ValueError("path must be a NUL-free POSIX relative path")
    if value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        raise ValueError("absolute and drive-qualified paths are not allowed")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("path cannot contain empty, current, or parent segments")
    if PurePosixPath(value).is_absolute():
        raise ValueError("path must be relative")
    return value


def _validate_sorted_unique(values: Sequence[str], field_name: str) -> None:
    if tuple(values) != tuple(sorted(set(values))):
        raise ValueError(f"{field_name} must be sorted and unique")


class ManifestEntry(StrictFrozenModel):
    id: StableId
    path: Annotated[str, Field(min_length=1, max_length=4096)]
    split: Split
    label: Annotated[str, Field(min_length=1, max_length=512)] | None = None
    ordinal: NonNegativeInt

    _relative_path = field_validator("path")(_validate_relative_path)


class ImageRecord(StrictFrozenModel):
    id: StableId
    path: Annotated[str, Field(min_length=1, max_length=4096)]
    split: Split
    label: Annotated[str, Field(min_length=1, max_length=512)] | None = None
    byte_sha256: Sha256
    byte_size: NonNegativeInt
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]
    format: Annotated[str, Field(min_length=1, max_length=32)]
    phash: Annotated[int, Field(ge=0, le=(1 << 64) - 1)] | None = None

    _relative_path = field_validator("path")(_validate_relative_path)

    @field_validator("format")
    @classmethod
    def canonical_format(cls, value: str) -> str:
        canonical = value.removeprefix(".").lower()
        if not canonical or not canonical.isalnum():
            raise ValueError("format must be an alphanumeric canonical name")
        return canonical


class ValidationIssue(StrictFrozenModel):
    record_id: StableId | None = None
    path: Annotated[str, Field(min_length=1, max_length=4096)] | None = None
    split: Split | None = None
    label: Annotated[str, Field(min_length=1, max_length=512)] | None = None
    code: ValidationIssueCode
    message: Annotated[str, Field(min_length=1, max_length=1000)]

    @field_validator("path")
    @classmethod
    def relative_path_if_present(cls, value: str | None) -> str | None:
        return None if value is None else _validate_relative_path(value)


class DuplicateEvidence(StrictFrozenModel):
    exact_match: bool = False
    phash_distance: Annotated[int, Field(ge=0, le=64)] | None = None
    cosine_similarity: Annotated[float, Field(ge=-1.0, le=1.0)] | None = None

    @model_validator(mode="after")
    def has_evidence(self) -> Self:
        if not self.exact_match and self.phash_distance is None and self.cosine_similarity is None:
            raise ValueError("at least one evidence value is required")
        return self


class DuplicateEdge(StrictFrozenModel):
    left_id: StableId
    right_id: StableId
    evidence: DuplicateEvidence
    classification: DuplicateClassification
    decision: EdgeDecision
    confidence: UnitFloat

    @model_validator(mode="after")
    def valid_pair_and_classification(self) -> Self:
        if self.left_id >= self.right_id:
            raise ValueError("edge identifiers must satisfy left_id < right_id")
        if self.classification is DuplicateClassification.EXACT and not self.evidence.exact_match:
            raise ValueError("exact classification requires exact-match evidence")
        if (
            self.classification is DuplicateClassification.SEMANTIC_CANDIDATE
            and self.evidence.cosine_similarity is None
        ):
            raise ValueError("semantic classification requires cosine evidence")
        if (
            self.classification is DuplicateClassification.TRANSFORMED_DUPLICATE
            and self.evidence.exact_match
        ):
            raise ValueError("an exact match cannot be classified only as transformed")
        return self


class DuplicateFamily(StrictFrozenModel):
    family_id: StableId
    member_ids: Annotated[tuple[StableId, ...], Field(min_length=1)]
    edge_count: NonNegativeInt = 0

    @field_validator("member_ids")
    @classmethod
    def canonical_members(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        _validate_sorted_unique(values, "member_ids")
        return values

    @model_validator(mode="after")
    def deterministic_family_id(self) -> Self:
        if self.family_id != family_id_for(self.member_ids):
            raise ValueError("family_id does not match the canonical member set")
        return self


class SplitBoundary(StrictFrozenModel):
    left: Split
    right: Split

    @model_validator(mode="after")
    def canonical_order(self) -> Self:
        if _SPLIT_ORDER[self.left] >= _SPLIT_ORDER[self.right]:
            raise ValueError("split boundary must contain distinct splits in canonical order")
        return self


class LeakageGroup(StrictFrozenModel):
    family_id: StableId
    member_ids: Annotated[tuple[StableId, ...], Field(min_length=2)]
    splits: Annotated[tuple[Split, ...], Field(min_length=2)]
    boundaries: Annotated[tuple[SplitBoundary, ...], Field(min_length=1)]
    labels: tuple[str, ...] = ()
    evidence_types: Annotated[
        tuple[DuplicateClassification, ...], Field(min_length=1)
    ]
    strongest_evidence: DuplicateClassification
    label_conflict: bool

    @model_validator(mode="after")
    def canonical_collections(self) -> Self:
        _validate_sorted_unique(self.member_ids, "member_ids")
        if self.splits != tuple(sorted(set(self.splits), key=_SPLIT_ORDER.__getitem__)):
            raise ValueError("splits must be sorted and unique")
        expected_boundaries = tuple(
            sorted(
                set(self.boundaries),
                key=lambda boundary: (
                    _SPLIT_ORDER[boundary.left],
                    _SPLIT_ORDER[boundary.right],
                ),
            )
        )
        if self.boundaries != expected_boundaries:
            raise ValueError("boundaries must be sorted and unique")
        _validate_sorted_unique(self.labels, "labels")
        expected_evidence = tuple(
            sorted(set(self.evidence_types), key=_CLASSIFICATION_ORDER.__getitem__)
        )
        if self.evidence_types != expected_evidence:
            raise ValueError("evidence_types must be sorted and unique")
        if self.strongest_evidence is not self.evidence_types[0]:
            raise ValueError("strongest_evidence must be the strongest listed evidence type")
        if self.label_conflict != (len(self.labels) > 1):
            raise ValueError("label_conflict must agree with the distinct labels")
        return self


class LabelConflict(StrictFrozenModel):
    family_id: StableId
    member_ids: Annotated[tuple[StableId, ...], Field(min_length=2)]
    labels: Annotated[tuple[str, ...], Field(min_length=2)]
    kind: ConflictKind

    @model_validator(mode="after")
    def canonical_collections(self) -> Self:
        _validate_sorted_unique(self.member_ids, "member_ids")
        _validate_sorted_unique(self.labels, "labels")
        return self


class PackageVersion(StrictFrozenModel):
    name: Annotated[str, Field(min_length=1, max_length=256)]
    version: Annotated[str, Field(min_length=1, max_length=256)]


class RunMetadata(StrictFrozenModel):
    timestamp: datetime
    git_commit_sha: Annotated[str, Field(pattern=r"^[0-9a-f]{7,64}$")] | None = None
    git_dirty: bool | None = None
    python_version: Annotated[str, Field(min_length=1, max_length=128)]
    os: Annotated[str, Field(min_length=1, max_length=256)]
    cpu: Annotated[str, Field(min_length=1, max_length=512)]
    cuda_available: bool
    gpu_model: Annotated[str, Field(min_length=1, max_length=512)] | None = None
    package_versions: tuple[PackageVersion, ...] = ()
    configuration_sha256: Sha256
    dataset_manifest_sha256: Sha256
    random_seeds: tuple[int, ...] = ()

    @field_validator("timestamp")
    @classmethod
    def timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a timezone")
        return value

    @model_validator(mode="after")
    def canonical_metadata(self) -> Self:
        package_names = tuple(package.name for package in self.package_versions)
        if package_names != tuple(sorted(set(package_names), key=str.casefold)):
            raise ValueError("package_versions must have unique names in canonical order")
        if self.random_seeds != tuple(sorted(set(self.random_seeds))):
            raise ValueError("random_seeds must be sorted and unique")
        if self.cuda_available is False and self.gpu_model is not None:
            raise ValueError("gpu_model requires cuda_available=true")
        return self


class AuditSummary(StrictFrozenModel):
    valid_image_count: NonNegativeInt
    invalid_image_count: NonNegativeInt
    leakage_group_count: NonNegativeInt
    contaminated_image_count: NonNegativeInt
    evaluation_image_count: NonNegativeInt
    contaminated_evaluation_fraction: UnitFloat
    exact_leakage_group_count: NonNegativeInt
    perceptual_leakage_group_count: NonNegativeInt
    embedding_only_review_count: NonNegativeInt
    cross_label_conflict_count: NonNegativeInt


class AuditArtifact(StrictFrozenModel):
    artifact_type: Literal["audit"] = "audit"
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    metadata: RunMetadata
    records: tuple[ImageRecord, ...]
    invalid_records: tuple[ValidationIssue, ...]
    edges: tuple[DuplicateEdge, ...]
    families: tuple[DuplicateFamily, ...]
    leakage_groups: tuple[LeakageGroup, ...]
    label_conflicts: tuple[LabelConflict, ...]
    summary: AuditSummary

    @model_validator(mode="after")
    def canonical_order(self) -> Self:
        record_ids = tuple(record.id for record in self.records)
        if record_ids != tuple(sorted(set(record_ids))):
            raise ValueError("records must have unique IDs in canonical order")
        edge_pairs = tuple((edge.left_id, edge.right_id) for edge in self.edges)
        if edge_pairs != tuple(sorted(set(edge_pairs))):
            raise ValueError("edges must have unique pairs in canonical order")
        family_ids = tuple(family.family_id for family in self.families)
        if family_ids != tuple(sorted(set(family_ids))):
            raise ValueError("families must have unique IDs in canonical order")
        return self


class SplitRatio(StrictFrozenModel):
    split: Split
    ratio: UnitFloat


class RepairAssignment(StrictFrozenModel):
    record_id: StableId
    family_id: StableId
    original_split: Split
    repaired_split: Split


class SplitStatistics(StrictFrozenModel):
    split: Split
    image_count: NonNegativeInt
    class_counts: tuple[tuple[str, NonNegativeInt], ...] = ()

    @field_validator("class_counts")
    @classmethod
    def canonical_class_counts(
        cls, values: tuple[tuple[str, int], ...]
    ) -> tuple[tuple[str, int], ...]:
        labels = tuple(label for label, _ in values)
        _validate_sorted_unique(labels, "class_counts labels")
        return values


class RepairSummary(StrictFrozenModel):
    objective_value: NonNegativeFloat
    split_size_error_before: UnitFloat
    split_size_error_after: UnitFloat
    class_divergence_before: UnitFloat
    class_divergence_after: UnitFloat
    definite_leakage_groups_before: NonNegativeInt
    definite_leakage_groups_after: NonNegativeInt
    moved_image_count: NonNegativeInt
    hard_group_invariant_satisfied: bool


class RepairArtifact(StrictFrozenModel):
    artifact_type: Literal["repair"] = "repair"
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    metadata: RunMetadata
    requested_ratios: Annotated[
        tuple[SplitRatio, ...], Field(min_length=3, max_length=3)
    ]
    integer_targets: Annotated[
        tuple[tuple[Split, NonNegativeInt], ...], Field(min_length=3, max_length=3)
    ]
    assignments: tuple[RepairAssignment, ...]
    excluded_invalid_ids: tuple[StableId, ...] = ()
    infeasibility_warnings: tuple[
        Annotated[str, Field(min_length=1, max_length=1000)], ...
    ] = ()
    split_size_weight: NonNegativeFloat
    class_balance_weight: NonNegativeFloat
    random_seed: Annotated[int, Field(ge=0)]
    local_improvement_iterations: NonNegativeInt
    repaired_manifest_sha256: Sha256
    before_split_statistics: tuple[SplitStatistics, ...]
    after_split_statistics: tuple[SplitStatistics, ...]
    summary: RepairSummary

    @model_validator(mode="after")
    def valid_repair_contract(self) -> Self:
        splits = tuple(item.split for item in self.requested_ratios)
        expected_splits = (Split.TRAIN, Split.VAL, Split.TEST)
        if splits != expected_splits:
            raise ValueError("requested_ratios must contain train, val, and test in order")
        if not math.isclose(
            sum(item.ratio for item in self.requested_ratios),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("requested split ratios must sum to one")
        target_splits = tuple(split for split, _ in self.integer_targets)
        if target_splits != expected_splits:
            raise ValueError("integer_targets must contain train, val, and test in order")
        if sum(count for _, count in self.integer_targets) != len(self.assignments):
            raise ValueError("integer_targets must sum to the assignment count")
        if self.split_size_weight == 0.0 and self.class_balance_weight == 0.0:
            raise ValueError("at least one repair objective weight must be positive")
        assignment_ids = tuple(item.record_id for item in self.assignments)
        if assignment_ids != tuple(sorted(set(assignment_ids))):
            raise ValueError("assignments must have unique record IDs in canonical order")
        _validate_sorted_unique(self.excluded_invalid_ids, "excluded_invalid_ids")
        _validate_sorted_unique(self.infeasibility_warnings, "infeasibility_warnings")
        return self


class BinaryMetrics(StrictFrozenModel):
    true_positives: NonNegativeInt
    false_positives: NonNegativeInt
    false_negatives: NonNegativeInt
    precision: UnitFloat
    recall: UnitFloat
    f1: UnitFloat

    @classmethod
    def from_counts(cls, true_positives: int, false_positives: int, false_negatives: int) -> Self:
        if min(true_positives, false_positives, false_negatives) < 0:
            raise ValueError("metric counts cannot be negative")
        precision_denominator = true_positives + false_positives
        recall_denominator = true_positives + false_negatives
        precision = true_positives / precision_denominator if precision_denominator else 0.0
        recall = true_positives / recall_denominator if recall_denominator else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        return cls(
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            precision=precision,
            recall=recall,
            f1=f1,
        )

    @model_validator(mode="after")
    def metrics_match_counts(self) -> Self:
        expected = type(self).from_counts_unchecked(
            self.true_positives,
            self.false_positives,
            self.false_negatives,
        )
        for name in ("precision", "recall", "f1"):
            if not math.isclose(
                getattr(self, name), getattr(expected, name), rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"{name} does not match the confusion counts")
        return self

    @classmethod
    def from_counts_unchecked(cls, tp: int, fp: int, fn: int) -> Self:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        return cls.model_construct(
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1=f1,
        )


class DetectionMetricRow(StrictFrozenModel):
    detector: Annotated[str, Field(min_length=1, max_length=128)]
    corruption_type: Annotated[str, Field(min_length=1, max_length=128)]
    threshold: float | None = None
    metrics: BinaryMetrics


class EmbeddingProvenance(StrictFrozenModel):
    """Serializable identity for the embedder that produced benchmark evidence."""

    backend: Literal["fake", "dinov2", "custom"]
    model_identity: Annotated[str, Field(min_length=1, max_length=1000)]
    model_revision: Annotated[str, Field(min_length=1, max_length=64)] | None = None
    preprocessing_version: Annotated[str, Field(min_length=1, max_length=256)]
    device: Literal["cpu", "cuda"]
    is_synthetic: bool

    @model_validator(mode="after")
    def backend_claims_are_consistent(self) -> Self:
        if self.backend == "fake":
            if not self.is_synthetic or not self.model_identity.startswith("fake:"):
                raise ValueError("fake provenance must be explicitly synthetic and fake-labeled")
            if self.model_revision is not None:
                raise ValueError("fake provenance cannot claim a model revision")
            if self.device != "cpu":
                raise ValueError("fake provenance must use the CPU device")
        elif self.backend == "dinov2":
            if self.is_synthetic:
                raise ValueError("DINOv2 provenance cannot be labeled synthetic")
            if self.model_revision is None or _IMMUTABLE_REVISION_RE.fullmatch(
                self.model_revision
            ) is None:
                raise ValueError("DINOv2 provenance requires an immutable revision")
            if not self.model_identity.startswith("huggingface:") or (
                f"@{self.model_revision}" not in self.model_identity
            ):
                raise ValueError("DINOv2 model identity must contain its pinned revision")
        return self

    @property
    def detector_name(self) -> str:
        """Return a compact detector label without obscuring fake versus DINOv2."""

        if self.backend == "fake":
            return "synthetic_fake_embedding_cosine_not_dinov2"
        if self.backend == "dinov2":
            if self.model_revision is None:  # pragma: no cover - validated above
                raise AssertionError("validated DINOv2 revision is missing")
            return f"dinov2_embedding_cosine@{self.model_revision[:12]}"
        digest = hashlib.sha256(self.model_identity.encode()).hexdigest()[:12]
        return f"custom_embedding_cosine@{digest}"


class DetectionBenchmarkArtifact(StrictFrozenModel):
    artifact_type: Literal["detection_benchmark"] = "detection_benchmark"
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    metadata: RunMetadata
    embedding_provenance: EmbeddingProvenance
    rows: tuple[DetectionMetricRow, ...]

    @model_validator(mode="after")
    def embedding_detectors_match_provenance(self) -> Self:
        expected = self.embedding_provenance.detector_name
        for row in self.rows:
            backend_specific = row.detector.startswith(
                (
                    "synthetic_fake_embedding",
                    "dinov2_embedding",
                    "custom_embedding",
                    "combined_exact_phash_plus_",
                )
            )
            if backend_specific and expected not in row.detector:
                raise ValueError("embedding detector labels must match artifact provenance")
        return self


class ScalingMetricRow(StrictFrozenModel):
    dataset_size: Annotated[int, Field(gt=0)]
    stage: Annotated[str, Field(min_length=1, max_length=128)]
    mode: Annotated[str, Field(min_length=1, max_length=128)]
    duration_seconds: NonNegativeFloat
    peak_memory_bytes: NonNegativeInt | None = None
    memory_measurement_scope: Literal["python_allocations_via_tracemalloc"] | None = None
    recall_at_k: UnitFloat | None = None

    @model_validator(mode="after")
    def memory_value_and_scope_are_paired(self) -> Self:
        if (self.peak_memory_bytes is None) != (self.memory_measurement_scope is None):
            raise ValueError("peak_memory_bytes and memory_measurement_scope must be set together")
        return self


class ScalingBenchmarkArtifact(StrictFrozenModel):
    artifact_type: Literal["scaling_benchmark"] = "scaling_benchmark"
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    metadata: RunMetadata
    rows: tuple[ScalingMetricRow, ...]


class AccuracyMetric(StrictFrozenModel):
    correct: NonNegativeInt
    total: NonNegativeInt
    accuracy: UnitFloat

    @model_validator(mode="after")
    def accuracy_matches_counts(self) -> Self:
        if self.correct > self.total:
            raise ValueError("correct cannot exceed total")
        expected = self.correct / self.total if self.total else 0.0
        if not math.isclose(self.accuracy, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("accuracy does not match correct/total")
        return self


class NamedAccuracy(StrictFrozenModel):
    name: Annotated[str, Field(min_length=1, max_length=512)]
    metric: AccuracyMetric


class TrainingRun(StrictFrozenModel):
    seed: int
    condition: TrainingCondition
    split_manifest_sha256: Sha256
    train_accuracy: AccuracyMetric
    validation_accuracy: AccuracyMetric
    test_accuracy: AccuracyMetric
    per_class_test_accuracy: tuple[NamedAccuracy, ...]
    contaminated_example_accuracy: AccuracyMetric | None = None
    clean_only_test_accuracy: AccuracyMetric
    duration_seconds: NonNegativeFloat

    @field_validator("per_class_test_accuracy")
    @classmethod
    def canonical_classes(cls, values: tuple[NamedAccuracy, ...]) -> tuple[NamedAccuracy, ...]:
        names = tuple(value.name for value in values)
        if names != tuple(sorted(set(names))):
            raise ValueError("per-class results must have unique names in canonical order")
        return values


class TrainingArtifact(StrictFrozenModel):
    artifact_type: Literal["training_results"] = "training_results"
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    metadata: RunMetadata
    runs: tuple[TrainingRun, ...]

    @field_validator("runs")
    @classmethod
    def canonical_runs(cls, values: tuple[TrainingRun, ...]) -> tuple[TrainingRun, ...]:
        keys = tuple((run.condition.value, run.seed) for run in values)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("training runs must be unique and in canonical order")
        return values


__all__ = [
    "SCHEMA_VERSION",
    "AccuracyMetric",
    "AuditArtifact",
    "AuditSummary",
    "BinaryMetrics",
    "ConflictKind",
    "DetectionBenchmarkArtifact",
    "DetectionMetricRow",
    "DuplicateClassification",
    "DuplicateEdge",
    "DuplicateEvidence",
    "DuplicateFamily",
    "EdgeDecision",
    "EmbeddingProvenance",
    "ImageRecord",
    "LabelConflict",
    "LeakageGroup",
    "ManifestEntry",
    "NamedAccuracy",
    "PackageVersion",
    "RepairArtifact",
    "RepairAssignment",
    "RepairSummary",
    "RunMetadata",
    "ScalingBenchmarkArtifact",
    "ScalingMetricRow",
    "Sha256",
    "Split",
    "SplitBoundary",
    "SplitRatio",
    "SplitStatistics",
    "StrictFrozenModel",
    "TrainingArtifact",
    "TrainingCondition",
    "TrainingRun",
    "ValidationIssue",
    "ValidationIssueCode",
    "canonical_json",
    "canonical_sha256",
    "family_id_for",
    "stable_id",
]
