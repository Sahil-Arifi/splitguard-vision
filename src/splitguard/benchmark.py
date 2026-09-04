"""Offline detector evaluation and clearly labeled synthetic scale benchmarks."""

from __future__ import annotations

import hashlib
import io
import math
import os
import re
import tempfile
import tracemalloc
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Annotated, Any, Generic, Literal, Self, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
import yaml
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from splitguard.hashing import (
    BKTree,
    PhashCandidatePair,
    brute_force_phash_pairs,
    compute_phash,
    fingerprint_records,
)
from splitguard.manifest import discover_image_folder
from splitguard.models.embedder import (
    DEFAULT_DINOV2_DIMENSION,
    DEFAULT_DINOV2_PREPROCESSING,
    DinoV2Embedder,
    Embedder,
    FakeEmbedder,
    normalize_embeddings,
)
from splitguard.neighbors import NeighborBenchmarkResult, benchmark_exact_vs_hnsw
from splitguard.schemas import (
    BinaryMetrics,
    DetectionBenchmarkArtifact,
    DetectionMetricRow,
    EmbeddingProvenance,
    ImageRecord,
    RunMetadata,
    ScalingBenchmarkArtifact,
    ScalingMetricRow,
    Split,
    StableId,
    StrictFrozenModel,
    canonical_sha256,
    stable_id,
)
from splitguard.synthetic import (
    CorruptionType,
    SyntheticCorruptionSet,
    generate_controlled_corruptions,
)
from splitguard.validation import scan_images

Float32Array = npt.NDArray[np.float32]
T = TypeVar("T")
ImageEmbedder: TypeAlias = Embedder
EmbeddingScaleSource: TypeAlias = Literal[
    "synthetic_random_unit_vectors_not_dinov2",
    "pixel_derived_fake_embeddings_not_dinov2",
]

_DINOV2_REVISION_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


class BenchmarkInputError(ValueError):
    """Raised when benchmark truth, evidence, or configuration is invalid."""


class _ConfigModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


class DetectionSweepConfig(_ConfigModel):
    """Threshold grid for detector-layer and combined-policy evaluation."""

    phash_thresholds: tuple[StrictInt, ...] = (0, 2, 4, 6, 8, 12)
    cosine_thresholds: tuple[float, ...] = (0.99, 0.97, 0.95, 0.9, 0.8)
    combined_phash_threshold: Annotated[StrictInt, Field(ge=0, le=64)] = 8
    combined_cosine_threshold: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.95
    embedding_only_is_duplicate: StrictBool = False
    embedding_backend: Literal["fake", "dinov2"] = "fake"
    embedding_model: Annotated[str, Field(min_length=1, max_length=512)] = (
        "fake:sha256-expand-v1"
    )
    embedding_revision: Annotated[str, Field(min_length=1, max_length=64)] | None = None
    embedding_device: Literal["auto", "cpu", "cuda"] = "cpu"
    embedding_batch_size: Annotated[StrictInt, Field(gt=0)] = 32
    source_count: Annotated[StrictInt, Field(ge=2)] = 16

    @field_validator("phash_thresholds", mode="before")
    @classmethod
    def normalize_phash_threshold_list(cls, values: object) -> object:
        return tuple(values) if isinstance(values, list) else values

    @field_validator("cosine_thresholds", mode="before")
    @classmethod
    def cosine_thresholds_are_not_booleans(cls, values: object) -> object:
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            if any(isinstance(value, bool) for value in values):
                raise TypeError("cosine_thresholds must contain numbers, not booleans")
        return tuple(values) if isinstance(values, list) else values

    @field_validator("combined_cosine_threshold", mode="before")
    @classmethod
    def combined_cosine_is_not_boolean(cls, value: object) -> object:
        if isinstance(value, bool):
            raise TypeError("combined_cosine_threshold must be a number, not a boolean")
        return value

    @model_validator(mode="after")
    def valid_embedding_backend(self) -> Self:
        if self.embedding_backend == "dinov2":
            if self.embedding_revision is None or _DINOV2_REVISION_RE.fullmatch(
                self.embedding_revision
            ) is None:
                raise ValueError(
                    "DINOv2 embedding_revision must be an immutable 40- or 64-character SHA"
                )
            if not self.embedding_model.strip():
                raise ValueError("DINOv2 embedding_model cannot be empty")
        else:
            if self.embedding_revision is not None:
                raise ValueError("fake embedding backend does not accept a model revision")
            if self.embedding_device != "cpu":
                raise ValueError("fake embedding backend must use the CPU device")
            if self.embedding_model != "fake:sha256-expand-v1":
                raise ValueError(
                    "fake embedding_model must be exactly 'fake:sha256-expand-v1'"
                )
        return self

    @field_validator("phash_thresholds")
    @classmethod
    def canonical_phash_thresholds(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if not values:
            raise ValueError("phash_thresholds cannot be empty")
        if any(isinstance(value, bool) or not 0 <= value <= 64 for value in values):
            raise ValueError("phash_thresholds must contain integers between 0 and 64")
        if values != tuple(sorted(set(values))):
            raise ValueError("phash_thresholds must be sorted and unique")
        return values

    @field_validator("cosine_thresholds")
    @classmethod
    def canonical_cosine_thresholds(
        cls,
        values: tuple[float, ...],
    ) -> tuple[float, ...]:
        if not values:
            raise ValueError("cosine_thresholds cannot be empty")
        if any(not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in values):
            raise ValueError("cosine_thresholds must contain finite values in [-1, 1]")
        if values != tuple(sorted(set(values), reverse=True)):
            raise ValueError("cosine_thresholds must be reverse-sorted and unique")
        return values


class ScalingBenchmarkConfig(_ConfigModel):
    """Sizes and index controls for reproducible synthetic scale measurements."""

    dataset_sizes: tuple[StrictInt, ...] = (1_000, 5_000, 10_000)
    brute_force_max_size: Annotated[StrictInt, Field(gt=0)] = 1_000
    phash_radius: Annotated[StrictInt, Field(ge=0, le=64)] = 8
    embedding_source: Literal["pixel_derived_fake_embeddings_not_dinov2"] = (
        "pixel_derived_fake_embeddings_not_dinov2"
    )
    embedding_dimension: Annotated[StrictInt, Field(gt=0)] = 64
    k: Annotated[StrictInt, Field(gt=0)] = 10
    threads: Annotated[StrictInt, Field(gt=0, le=1024)] = 1
    hnsw_m: Annotated[StrictInt, Field(ge=2)] = 32
    hnsw_ef_construction: Annotated[StrictInt, Field(ge=2)] = 200
    hnsw_ef_search: Annotated[StrictInt, Field(gt=0)] = 64

    @field_validator("dataset_sizes", mode="before")
    @classmethod
    def normalize_dataset_size_list(cls, values: object) -> object:
        return tuple(values) if isinstance(values, list) else values

    @field_validator("dataset_sizes")
    @classmethod
    def canonical_sizes(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if not values:
            raise ValueError("dataset_sizes cannot be empty")
        if any(isinstance(value, bool) or value <= 0 for value in values):
            raise ValueError("dataset_sizes must contain positive integers")
        if values != tuple(sorted(set(values))):
            raise ValueError("dataset_sizes must be sorted and unique")
        return values

    @model_validator(mode="after")
    def valid_hnsw_controls(self) -> Self:
        if self.hnsw_ef_construction < self.hnsw_m:
            raise ValueError("hnsw_ef_construction must be at least hnsw_m")
        if self.hnsw_ef_search < self.k:
            raise ValueError("hnsw_ef_search must be at least k")
        return self


class BenchmarkConfig(_ConfigModel):
    """Standalone benchmark configuration; it never selects a production model."""

    seed: Annotated[StrictInt, Field(ge=0, le=2**32 - 1)] = 20260903
    detection: DetectionSweepConfig = Field(default_factory=DetectionSweepConfig)
    scaling: ScalingBenchmarkConfig = Field(default_factory=ScalingBenchmarkConfig)


class DetectionObservation(StrictFrozenModel):
    """Independent truth plus observed evidence for one evaluated pair.

    ``is_duplicate`` is supplied by controlled ground truth. Detector outputs
    occupy separate fields and therefore cannot define their own labels.
    """

    left_id: StableId
    right_id: StableId
    corruption_type: Annotated[str, Field(min_length=1, max_length=128)]
    is_duplicate: bool
    exact_match: bool = False
    phash_distance: Annotated[int, Field(ge=0, le=64)] | None = None
    cosine_similarity: Annotated[float, Field(ge=-1.0, le=1.0)] | None = None
    embedding_provenance: EmbeddingProvenance | None = None

    @model_validator(mode="after")
    def canonical_pair_and_embedding_provenance(self) -> Self:
        if self.left_id >= self.right_id:
            raise ValueError("observation identifiers must satisfy left_id < right_id")
        if self.cosine_similarity is not None and self.embedding_provenance is None:
            raise ValueError("cosine evidence requires explicit embedding provenance")
        return self

    @property
    def embedding_source(self) -> str | None:
        """Compatibility label that never obscures fake versus real provenance."""

        if self.embedding_provenance is None:
            return None
        return self.embedding_provenance.detector_name


class RuntimeMeasurement(StrictFrozenModel):
    """Elapsed time and Python-allocation peak for one measured call."""

    duration_seconds: Annotated[float, Field(ge=0.0)]
    peak_memory_bytes: Annotated[int, Field(ge=0)]
    memory_scope: Literal["python_allocations_via_tracemalloc"] = (
        "python_allocations_via_tracemalloc"
    )


class MeasuredCall(Generic[T]):
    """A value paired with an explicit runtime measurement."""

    __slots__ = ("measurement", "value")

    def __init__(self, value: T, measurement: RuntimeMeasurement) -> None:
        self.value = value
        self.measurement = measurement


class EmbeddingScaleResult(StrictFrozenModel):
    """FAISS comparison with explicit vector provenance."""

    embedding_source: EmbeddingScaleSource
    comparison: NeighborBenchmarkResult
    rows: tuple[ScalingMetricRow, ...]

    @model_validator(mode="after")
    def rows_match_comparison(self) -> Self:
        if any(row.dataset_size != self.comparison.dataset_size for row in self.rows):
            raise ValueError("embedding scale rows must match the comparison dataset size")
        if not any(row.recall_at_k == self.comparison.recall_at_k for row in self.rows):
            raise ValueError("embedding scale rows must preserve measured recall@k")
        return self


def _detection_dataset_sha256(
    source_records: Sequence[ImageRecord],
    corruption_set: SyntheticCorruptionSet,
) -> str:
    return canonical_sha256(
        {
            "fixture_schema": "splitguard-controlled-detection-v2",
            "seed": corruption_set.seed,
            "source_records": [
                record.model_dump(mode="json") for record in source_records
            ],
            "corruption_set": corruption_set.model_dump(mode="json"),
        }
    )


class DetectionBenchmarkRun(StrictFrozenModel):
    """CLI-ready detection benchmark output with no absolute workspace paths."""

    dataset_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    source_records: tuple[ImageRecord, ...]
    corruption_set: SyntheticCorruptionSet
    embedding_provenance: EmbeddingProvenance
    observations: tuple[DetectionObservation, ...]
    rows: tuple[DetectionMetricRow, ...]

    @model_validator(mode="after")
    def canonical_and_complete(self) -> Self:
        source_ids = tuple(record.id for record in self.source_records)
        if source_ids != tuple(sorted(set(source_ids))):
            raise ValueError("source_records must have unique IDs in canonical order")
        if source_ids != self.corruption_set.source_ids:
            raise ValueError("source_records must exactly match the corruption sources")
        source_hashes = {record.id: record.byte_sha256 for record in self.source_records}
        if any(
            source_hashes[item.source_id] != item.source_sha256
            for item in self.corruption_set.injections
        ):
            raise ValueError("source payload hashes must match the corruption truth")
        observation_keys = tuple(
            (item.corruption_type, item.left_id, item.right_id)
            for item in self.observations
        )
        if observation_keys != tuple(sorted(set(observation_keys))):
            raise ValueError("observations must be unique and in canonical order")
        expected_truth = {
            (item.corruption_type.value, *item.canonical_pair): item.is_duplicate
            for item in self.corruption_set.injections
        }
        expected_truth.update(
            {
                (item.corruption_type.value, *item.canonical_pair): item.is_duplicate
                for item in self.corruption_set.negative_controls
            }
        )
        observed_truth = {
            (item.corruption_type, item.left_id, item.right_id): item.is_duplicate
            for item in self.observations
        }
        if observed_truth != expected_truth:
            raise ValueError("observations must cover every positive and negative-control pair")
        if any(
            item.embedding_provenance != self.embedding_provenance
            for item in self.observations
        ):
            raise ValueError("observations must use the run's declared embedding provenance")
        if self.dataset_sha256 != _detection_dataset_sha256(
            self.source_records,
            self.corruption_set,
        ):
            raise ValueError("dataset_sha256 must bind the seed and every fixture payload")
        return self


@dataclass(frozen=True, slots=True)
class _ObservationPair:
    source_id: str
    derived_id: str
    source_path: str
    derived_path: str
    source_sha256: str
    derived_sha256: str
    corruption_type: str
    is_duplicate: bool
    valid_image: bool

    @property
    def canonical_pair(self) -> tuple[str, str]:
        left_id, right_id = sorted((self.source_id, self.derived_id))
        return left_id, right_id


def load_benchmark_config(path: str | os.PathLike[str]) -> BenchmarkConfig:
    """Load a strict benchmark YAML document with sanitized I/O failures."""

    config_path = Path(path)
    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise BenchmarkInputError(
            f"benchmark configuration {config_path.name!r} could not be loaded"
        ) from exc
    if document is None:
        document = {}
    if not isinstance(document, Mapping) or any(
        not isinstance(key, str) for key in document
    ):
        raise BenchmarkInputError("benchmark configuration must be a string-keyed mapping")
    return BenchmarkConfig.model_validate(document)


def build_detection_embedder(
    config: DetectionSweepConfig,
    *,
    seed: int = 0,
    fake_dimension: int = 32,
) -> tuple[ImageEmbedder, EmbeddingProvenance]:
    """Construct the explicitly configured embedder without running inference.

    DINOv2 remains lazy: model/network access can occur only when an explicitly
    invoked benchmark calls ``embed``. Offline tests select the fake backend.
    """

    if config.embedding_backend == "fake":
        embedder: ImageEmbedder = FakeEmbedder(
            dimension=fake_dimension,
            seed=seed,
            preprocessing_version="benchmark-fake-rgb-v1",
        )
        provenance = EmbeddingProvenance(
            backend="fake",
            model_identity=embedder.model_identity,
            model_revision=None,
            preprocessing_version=embedder.preprocessing_version,
            device="cpu",
            is_synthetic=True,
        )
        return embedder, provenance

    revision = config.embedding_revision
    if revision is None:  # pragma: no cover - config validator guarantees it
        raise AssertionError("validated DINOv2 revision is missing")
    dino = DinoV2Embedder(
        model_name=config.embedding_model,
        revision=revision,
        preprocessing_version=DEFAULT_DINOV2_PREPROCESSING,
        device=config.embedding_device,
        dimension=DEFAULT_DINOV2_DIMENSION,
    )
    return dino, EmbeddingProvenance(
        backend="dinov2",
        model_identity=dino.model_identity,
        model_revision=revision,
        preprocessing_version=dino.preprocessing_version,
        device=dino.device,
        is_synthetic=False,
    )


def _validate_embedder_provenance(
    embedder: ImageEmbedder,
    provenance: EmbeddingProvenance,
) -> None:
    if embedder.model_identity != provenance.model_identity:
        raise BenchmarkInputError("embedder model identity does not match its provenance")
    if embedder.preprocessing_version != provenance.preprocessing_version:
        raise BenchmarkInputError("embedder preprocessing does not match its provenance")

    if isinstance(embedder, FakeEmbedder):
        if provenance.backend != "fake" or provenance.device != "cpu":
            raise BenchmarkInputError("FakeEmbedder requires fake CPU provenance")
        return
    if isinstance(embedder, DinoV2Embedder):
        if provenance.backend != "dinov2" or provenance.device != embedder.device:
            raise BenchmarkInputError("DinoV2Embedder provenance backend or device is incorrect")
        return

    runtime_device = getattr(embedder, "device", None)
    if provenance.backend != "custom" or runtime_device != provenance.device:
        raise BenchmarkInputError(
            "a generic embedder requires custom provenance and a matching device"
        )


def measure_runtime(operation: Callable[[], T]) -> MeasuredCall[T]:
    """Measure one call without claiming to capture native-library allocations."""

    if not callable(operation):
        raise TypeError("operation must be callable")
    already_tracing = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start()
    baseline_current, baseline_peak = tracemalloc.get_traced_memory()
    started = perf_counter()
    try:
        value = operation()
    finally:
        duration = perf_counter() - started
        current, peak = tracemalloc.get_traced_memory()
        if not already_tracing:
            tracemalloc.stop()
    peak_delta = max(0, peak - max(baseline_current, baseline_peak))
    # Retain the read so static analysis cannot mistake it for an unused sample.
    _ = current
    return MeasuredCall(
        value,
        RuntimeMeasurement(
            duration_seconds=duration,
            peak_memory_bytes=peak_delta,
        ),
    )


def _canonical_observations(
    observations: Iterable[DetectionObservation],
) -> tuple[DetectionObservation, ...]:
    snapshot = tuple(observations)
    keys = tuple(
        (item.corruption_type, item.left_id, item.right_id) for item in snapshot
    )
    if len(keys) != len(set(keys)):
        raise BenchmarkInputError(
            "observations must contain unique corruption-type and pair combinations"
        )
    return tuple(
        sorted(
            snapshot,
            key=lambda item: (item.corruption_type, item.left_id, item.right_id),
        )
    )


def _pairwise_observations(
    observations: Iterable[DetectionObservation],
) -> tuple[DetectionObservation, ...]:
    return tuple(
        item
        for item in _canonical_observations(observations)
        if item.corruption_type != CorruptionType.MALFORMED_FILE.value
    )


def _binary_metrics(
    observations: Sequence[DetectionObservation],
    predicted: Callable[[DetectionObservation], bool],
) -> BinaryMetrics:
    true_positives = sum(item.is_duplicate and predicted(item) for item in observations)
    false_positives = sum(not item.is_duplicate and predicted(item) for item in observations)
    false_negatives = sum(item.is_duplicate and not predicted(item) for item in observations)
    return BinaryMetrics.from_counts(true_positives, false_positives, false_negatives)


def _phash_predicts(item: DetectionObservation, *, threshold: int) -> bool:
    return item.phash_distance is not None and item.phash_distance <= threshold


def _embedding_predicts(item: DetectionObservation, *, threshold: float) -> bool:
    return item.cosine_similarity is not None and item.cosine_similarity >= threshold


def _groups(
    observations: tuple[DetectionObservation, ...],
) -> tuple[tuple[str, tuple[DetectionObservation, ...]], ...]:
    corruption_types = tuple(sorted({item.corruption_type for item in observations}))
    return tuple(
        (
            corruption_type,
            tuple(
                item for item in observations if item.corruption_type == corruption_type
            ),
        )
        for corruption_type in corruption_types
    )


def _embedding_detector_name(
    observations: Sequence[DetectionObservation],
) -> str:
    provenances = tuple(
        item.embedding_provenance
        for item in observations
        if item.embedding_provenance is not None
    )
    if provenances and any(item != provenances[0] for item in provenances[1:]):
        raise BenchmarkInputError(
            "one metric sweep cannot mix evidence from different embedding backends"
        )
    return (
        provenances[0].detector_name
        if provenances
        else "embedding_cosine_no_evidence"
    )


def evaluate_exact_detection(
    observations: Iterable[DetectionObservation],
) -> tuple[DetectionMetricRow, ...]:
    """Evaluate SHA-equivalence evidence separately for every corruption family."""

    canonical = _pairwise_observations(observations)
    return tuple(
        DetectionMetricRow(
            detector="sha256_exact",
            corruption_type=corruption_type,
            threshold=None,
            metrics=_binary_metrics(items, lambda item: item.exact_match),
        )
        for corruption_type, items in _groups(canonical)
    )


def _validated_phash_thresholds(thresholds: Iterable[int]) -> tuple[int, ...]:
    values = tuple(thresholds)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("pHash thresholds must be integers")
    if any(not 0 <= value <= 64 for value in values):
        raise ValueError("pHash thresholds must be between 0 and 64")
    if not values:
        raise ValueError("at least one pHash threshold is required")
    return tuple(sorted(set(values)))


def _validated_cosine_thresholds(thresholds: Iterable[float]) -> tuple[float, ...]:
    raw = tuple(thresholds)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw):
        raise TypeError("cosine thresholds must be numbers")
    values = tuple(float(value) for value in raw)
    if not values:
        raise ValueError("at least one cosine threshold is required")
    if any(not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in values):
        raise ValueError("cosine thresholds must be finite values in [-1, 1]")
    return tuple(sorted(set(values), reverse=True))


def evaluate_phash_thresholds(
    observations: Iterable[DetectionObservation],
    thresholds: Iterable[int],
) -> tuple[DetectionMetricRow, ...]:
    """Return a low-to-high Hamming-threshold PR sweep per corruption type."""

    canonical = _pairwise_observations(observations)
    accepted_thresholds = _validated_phash_thresholds(thresholds)
    rows = [
        DetectionMetricRow(
            detector="phash_hamming",
            corruption_type=corruption_type,
            threshold=float(threshold),
            metrics=_binary_metrics(
                items,
                partial(_phash_predicts, threshold=threshold),
            ),
        )
        for corruption_type, items in _groups(canonical)
        for threshold in accepted_thresholds
    ]
    return tuple(rows)


def evaluate_embedding_thresholds(
    observations: Iterable[DetectionObservation],
    thresholds: Iterable[float],
) -> tuple[DetectionMetricRow, ...]:
    """Return a high-to-low cosine-threshold PR sweep for declared embeddings.

    This function intentionally refuses cosine observations without provenance;
    the model contract enforces the same rule at construction time.
    """

    canonical = _pairwise_observations(observations)
    accepted_thresholds = _validated_cosine_thresholds(thresholds)
    detector_name = _embedding_detector_name(canonical)
    rows = [
        DetectionMetricRow(
            detector=detector_name,
            corruption_type=corruption_type,
            threshold=threshold,
            metrics=_binary_metrics(
                items,
                partial(_embedding_predicts, threshold=threshold),
            ),
        )
        for corruption_type, items in _groups(canonical)
        for threshold in accepted_thresholds
    ]
    return tuple(rows)


def evaluate_combined_policy(
    observations: Iterable[DetectionObservation],
    *,
    phash_threshold: int,
    cosine_threshold: float,
    embedding_only_is_duplicate: bool = False,
) -> tuple[DetectionMetricRow, ...]:
    """Evaluate exact+pHash policy, optionally promoting embedding-only pairs."""

    accepted_phash = _validated_phash_thresholds((phash_threshold,))[0]
    accepted_cosine = _validated_cosine_thresholds((cosine_threshold,))[0]
    canonical = _pairwise_observations(observations)
    embedding_detector = _embedding_detector_name(canonical)

    def predicted(item: DetectionObservation) -> bool:
        exact_or_phash = item.exact_match or (
            item.phash_distance is not None and item.phash_distance <= accepted_phash
        )
        semantic = (
            embedding_only_is_duplicate
            and item.cosine_similarity is not None
            and item.cosine_similarity >= accepted_cosine
        )
        return exact_or_phash or semantic

    detector = (
        f"combined_exact_phash_plus_{embedding_detector}"
        if embedding_only_is_duplicate
        else "combined_exact_phash_embedding_review_only"
    )
    return tuple(
        DetectionMetricRow(
            detector=detector,
            corruption_type=corruption_type,
            threshold=None,
            metrics=_binary_metrics(items, predicted),
        )
        for corruption_type, items in _groups(canonical)
    )


def detection_benchmark_rows(
    observations: Iterable[DetectionObservation],
    config: DetectionSweepConfig,
) -> tuple[DetectionMetricRow, ...]:
    """Evaluate every detection layer and return canonical per-corruption rows."""

    canonical = _pairwise_observations(observations)
    rows = (
        *evaluate_exact_detection(canonical),
        *evaluate_phash_thresholds(canonical, config.phash_thresholds),
        *evaluate_embedding_thresholds(canonical, config.cosine_thresholds),
        *evaluate_combined_policy(
            canonical,
            phash_threshold=config.combined_phash_threshold,
            cosine_threshold=config.combined_cosine_threshold,
            embedding_only_is_duplicate=config.embedding_only_is_duplicate,
        ),
    )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.detector,
                row.corruption_type,
                -math.inf if row.threshold is None else row.threshold,
            ),
        )
    )


def build_detection_artifact(
    metadata: RunMetadata,
    rows: Iterable[DetectionMetricRow],
    embedding_provenance: EmbeddingProvenance,
) -> DetectionBenchmarkArtifact:
    """Build a canonically ordered, schema-versioned detection artifact."""

    ordered = tuple(
        sorted(
            rows,
            key=lambda row: (
                row.detector,
                row.corruption_type,
                -math.inf if row.threshold is None else row.threshold,
            ),
        )
    )
    keys = tuple((row.detector, row.corruption_type, row.threshold) for row in ordered)
    if len(keys) != len(set(keys)):
        raise BenchmarkInputError("detection metric rows must be unique")
    return DetectionBenchmarkArtifact(
        metadata=metadata,
        embedding_provenance=embedding_provenance,
        rows=ordered,
    )


def _safe_relative_file(root: Path, relative_path: str) -> Path:
    parts = PurePosixPath(relative_path).parts
    candidate = root.joinpath(*parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise BenchmarkInputError("a synthetic benchmark image is unavailable") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise BenchmarkInputError("a synthetic benchmark path escapes its root") from exc
    if not resolved.is_file():
        raise BenchmarkInputError("a synthetic benchmark path is not a file")
    return resolved


def _decoded_rgb(payload: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            opened.load()
            return ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise BenchmarkInputError("a declared valid synthetic image could not be decoded") from exc


def _decoded_image_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    try:
        digest = hashlib.sha256()
        digest.update(rgb.width.to_bytes(8, "big", signed=False))
        digest.update(rgb.height.to_bytes(8, "big", signed=False))
        digest.update(rgb.tobytes())
        return digest.hexdigest()
    finally:
        rgb.close()


def _write_local_image_collection(root: Path, dataset_size: int, seed: int) -> None:
    """Create deterministic, unique 24x24 PNGs in ImageFolder layout."""

    for index in range(dataset_size):
        split = (
            Split.TRAIN
            if index % 10 < 8
            else Split.VAL
            if index % 10 == 8
            else Split.TEST
        )
        label = f"class_{index % 4}"
        path = root / split.value / label / f"image_{index:08d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(f"{seed}\0local-image\0{index}".encode()).digest()
        tiled = np.resize(np.frombuffer(digest, dtype=np.uint8), 24 * 24 * 3)
        pixels = tiled.reshape(24, 24, 3).copy()
        rows, columns = np.indices((24, 24))
        pixels[..., 0] ^= ((columns * 7 + rows * 3) % 256).astype(np.uint8)
        pixels[..., 1] ^= ((columns * 5 + rows * 11) % 256).astype(np.uint8)
        pixels[..., 2] ^= ((columns * 13 + rows * 2) % 256).astype(np.uint8)
        image = Image.fromarray(pixels)
        try:
            image.save(path, format="PNG", compress_level=1, optimize=False)
        finally:
            image.close()


def _validated_fixture_records(root: Path) -> tuple[ImageRecord, ...]:
    manifest = discover_image_folder(root)
    scan = scan_images(root, list(manifest.entries), max_image_pixels=1_000_000)
    if scan.issues or len(scan.records) != len(manifest.entries):
        raise RuntimeError("generated local benchmark fixture failed validation")
    return scan.records


def observe_synthetic_corruptions(
    source_root: str | os.PathLike[str],
    generated_root: str | os.PathLike[str],
    corruption_set: SyntheticCorruptionSet,
    *,
    embedder: ImageEmbedder | None = None,
    embedding_provenance: EmbeddingProvenance | None = None,
    embedding_batch_size: int = 32,
    fake_embedding_dimension: int = 32,
) -> tuple[DetectionObservation, ...]:
    """Measure positive and negative-control pairs with a declared embedder.

    The default remains ``FakeEmbedder`` for offline tests. A real DINOv2 run
    must pass both a generic ``ImageEmbedder`` and matching provenance, normally
    obtained from :func:`build_detection_embedder`.
    """

    if isinstance(embedding_batch_size, bool) or not isinstance(
        embedding_batch_size, int
    ):
        raise TypeError("embedding_batch_size must be an integer")
    if embedding_batch_size <= 0:
        raise ValueError("embedding_batch_size must be positive")
    if embedder is None:
        if embedding_provenance is not None:
            raise BenchmarkInputError("embedding provenance requires an embedder")
        fake = FakeEmbedder(
            dimension=fake_embedding_dimension,
            seed=corruption_set.seed,
            preprocessing_version="benchmark-fake-rgb-v1",
        )
        embedder = fake
        embedding_provenance = EmbeddingProvenance(
            backend="fake",
            model_identity=fake.model_identity,
            model_revision=None,
            preprocessing_version=fake.preprocessing_version,
            device="cpu",
            is_synthetic=True,
        )
    elif embedding_provenance is None:
        if not isinstance(embedder, FakeEmbedder):
            raise BenchmarkInputError(
                "non-fake embedders require explicit embedding provenance"
            )
        embedding_provenance = EmbeddingProvenance(
            backend="fake",
            model_identity=embedder.model_identity,
            model_revision=None,
            preprocessing_version=embedder.preprocessing_version,
            device="cpu",
            is_synthetic=True,
        )

    source = Path(source_root).resolve(strict=True)
    generated = Path(generated_root).resolve(strict=True)
    pair_specs = tuple(
        [
            _ObservationPair(
                source_id=item.source_id,
                derived_id=item.derived_id,
                source_path=item.source_path,
                derived_path=item.derived_path,
                source_sha256=item.source_sha256,
                derived_sha256=item.derived_sha256,
                corruption_type=item.corruption_type.value,
                is_duplicate=item.is_duplicate,
                valid_image=item.valid_image,
            )
            for item in corruption_set.injections
        ]
        + [
            _ObservationPair(
                source_id=item.source_id,
                derived_id=item.unrelated_derived_id,
                source_path=item.source_path,
                derived_path=item.unrelated_derived_path,
                source_sha256=item.source_sha256,
                derived_sha256=item.unrelated_derived_sha256,
                corruption_type=item.corruption_type.value,
                is_duplicate=False,
                valid_image=True,
            )
            for item in corruption_set.negative_controls
        ]
    )
    if embedder is None or embedding_provenance is None:  # pragma: no cover - resolved above
        raise AssertionError("embedder and provenance were not resolved")
    _validate_embedder_provenance(embedder, embedding_provenance)
    source_payloads: dict[str, bytes] = {}
    derived_payloads: dict[str, bytes] = {}
    source_images: dict[str, Image.Image] = {}
    derived_images: dict[str, Image.Image] = {}
    try:
        for pair in pair_specs:
            if pair.source_id not in source_payloads:
                source_path = _safe_relative_file(source, pair.source_path)
                payload = source_path.read_bytes()
                if hashlib.sha256(payload).hexdigest() != pair.source_sha256:
                    raise BenchmarkInputError("a synthetic benchmark source payload changed")
                source_payloads[pair.source_id] = payload
                source_images[pair.source_id] = _decoded_rgb(payload)
            elif hashlib.sha256(source_payloads[pair.source_id]).hexdigest() != pair.source_sha256:
                raise BenchmarkInputError("synthetic source truth contains inconsistent hashes")
            if pair.derived_id not in derived_payloads:
                derived_path = _safe_relative_file(generated, pair.derived_path)
                payload = derived_path.read_bytes()
                if hashlib.sha256(payload).hexdigest() != pair.derived_sha256:
                    raise BenchmarkInputError("a synthetic benchmark derived payload changed")
                derived_payloads[pair.derived_id] = payload
                if pair.valid_image:
                    derived_images[pair.derived_id] = _decoded_rgb(payload)
            elif (
                hashlib.sha256(derived_payloads[pair.derived_id]).hexdigest()
                != pair.derived_sha256
            ):
                raise BenchmarkInputError("synthetic derived truth contains inconsistent hashes")

        valid_pairs = tuple(pair for pair in pair_specs if pair.valid_image)
        content_by_endpoint: dict[tuple[str, str], str] = {}
        unique_images: dict[str, Image.Image] = {}
        for kind, images in (("source", source_images), ("derived", derived_images)):
            for record_id, image in sorted(images.items()):
                content_sha256 = _decoded_image_sha256(image)
                content_by_endpoint[(kind, record_id)] = content_sha256
                unique_images.setdefault(content_sha256, image)
        distinct_images = tuple(sorted(unique_images.items()))
        embedding_chunks: list[Float32Array] = []
        for offset in range(0, len(distinct_images), embedding_batch_size):
            batch = tuple(
                item[1] for item in distinct_images[offset : offset + embedding_batch_size]
            )
            vectors = embedder.embed(batch)
            embedding_chunks.append(
                normalize_embeddings(
                    vectors,
                    expected_rows=len(batch),
                    expected_dimension=embedder.dimension,
                )
            )
        embedding_matrix = (
            np.vstack(embedding_chunks)
            if embedding_chunks
            else np.empty((0, embedder.dimension), dtype=np.float32)
        )
        embedding_by_endpoint = {
            content_sha256: embedding_matrix[index]
            for index, (content_sha256, _) in enumerate(distinct_images)
        }
        cosine_by_pair = {
            (pair.corruption_type, *pair.canonical_pair): float(
                np.dot(
                    embedding_by_endpoint[
                        content_by_endpoint[("source", pair.source_id)]
                    ],
                    embedding_by_endpoint[
                        content_by_endpoint[("derived", pair.derived_id)]
                    ],
                )
            )
            for pair in valid_pairs
        }
        phash_by_content = {
            content_sha256: compute_phash(image)
            for content_sha256, image in distinct_images
        }

        observations: list[DetectionObservation] = []
        for pair in pair_specs:
            left_id, right_id = pair.canonical_pair
            if pair.valid_image:
                cosine = cosine_by_pair[(pair.corruption_type, left_id, right_id)]
                phash_distance = (
                    phash_by_content[content_by_endpoint[("source", pair.source_id)]]
                    ^ phash_by_content[content_by_endpoint[("derived", pair.derived_id)]]
                ).bit_count()
            else:
                cosine = None
                phash_distance = None
            observations.append(
                DetectionObservation(
                    left_id=left_id,
                    right_id=right_id,
                    corruption_type=pair.corruption_type,
                    is_duplicate=pair.is_duplicate,
                    exact_match=(
                        pair.valid_image
                        and source_payloads[pair.source_id]
                        == derived_payloads[pair.derived_id]
                    ),
                    phash_distance=phash_distance,
                    cosine_similarity=(
                        None if cosine is None else max(-1.0, min(1.0, cosine))
                    ),
                    embedding_provenance=embedding_provenance,
                )
            )
    finally:
        for image in (*source_images.values(), *derived_images.values()):
            image.close()
    return _canonical_observations(observations)


def run_detection_benchmark(
    config: BenchmarkConfig,
    *,
    workspace: str | os.PathLike[str] | None = None,
    embedder: ImageEmbedder | None = None,
    embedding_provenance: EmbeddingProvenance | None = None,
) -> DetectionBenchmarkRun:
    """Run the complete controlled detector benchmark core for CLI callers.

    When ``workspace`` is omitted, all image files live only for this call in a
    temporary directory. A provided workspace receives a new deterministic
    ``splitguard-detection-benchmark`` child. Returned contracts contain only
    stable IDs, hashes, and relative paths.
    """

    if (embedder is None) != (embedding_provenance is None):
        raise BenchmarkInputError(
            "embedder and embedding_provenance must be supplied together"
        )
    selected_embedder: ImageEmbedder
    selected_provenance: EmbeddingProvenance
    if embedder is None or embedding_provenance is None:
        selected_embedder, selected_provenance = build_detection_embedder(
            config.detection,
            seed=config.seed,
        )
    else:
        selected_embedder = embedder
        selected_provenance = embedding_provenance

    def execute(run_root: Path) -> DetectionBenchmarkRun:
        source_root = run_root / "source"
        generated_root = run_root / "generated"
        source_root.mkdir(parents=True)
        _write_local_image_collection(
            source_root,
            config.detection.source_count,
            config.seed,
        )
        source_records = _validated_fixture_records(source_root)
        corruption_set = generate_controlled_corruptions(
            source_root,
            source_records,
            generated_root,
            seed=config.seed,
        )
        observations = observe_synthetic_corruptions(
            source_root,
            generated_root,
            corruption_set,
            embedder=selected_embedder,
            embedding_provenance=selected_provenance,
            embedding_batch_size=config.detection.embedding_batch_size,
        )
        rows = detection_benchmark_rows(observations, config.detection)
        dataset_sha256 = _detection_dataset_sha256(source_records, corruption_set)
        return DetectionBenchmarkRun(
            dataset_sha256=dataset_sha256,
            source_records=source_records,
            corruption_set=corruption_set,
            embedding_provenance=selected_provenance,
            observations=observations,
            rows=rows,
        )

    if workspace is None:
        with tempfile.TemporaryDirectory(prefix="splitguard-detection-") as temporary:
            return execute(Path(temporary) / "run")

    parent = Path(workspace).resolve(strict=False)
    parent.mkdir(parents=True, exist_ok=True)
    run_root = parent / "splitguard-detection-benchmark"
    if run_root.exists():
        raise BenchmarkInputError("benchmark workspace child already exists")
    return execute(run_root)


def generate_synthetic_phash_records(
    dataset_size: int,
    *,
    seed: int,
) -> tuple[ImageRecord, ...]:
    """Generate deterministic hash-only records for index scaling measurements."""

    if isinstance(dataset_size, bool) or not isinstance(dataset_size, int):
        raise TypeError("dataset_size must be an integer")
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    records: list[ImageRecord] = []
    previous_hash = 0
    for index in range(dataset_size):
        payload = hashlib.sha256(f"{seed}\0phash\0{index}".encode()).digest()
        phash = int.from_bytes(payload[:8], "big")
        # Seed sparse known neighbors while retaining a realistic mostly-random set.
        if index % 23 == 1:
            phash = previous_hash ^ (1 << (index % 64))
        previous_hash = phash
        path = f"synthetic/phash/{index:08d}.png"
        records.append(
            ImageRecord(
                id=stable_id("bench", str(seed), "phash", str(index)),
                path=path,
                split=Split.TRAIN,
                label=None,
                byte_sha256=hashlib.sha256(payload + b"record").hexdigest(),
                byte_size=0,
                width=1,
                height=1,
                format="png",
                phash=phash,
            )
        )
    return tuple(sorted(records, key=lambda record: record.id))


def _phash_entries(records: Iterable[ImageRecord]) -> tuple[tuple[str, int], ...]:
    snapshot = tuple(records)
    ids = tuple(record.id for record in snapshot)
    if len(ids) != len(set(ids)):
        raise BenchmarkInputError("pHash benchmark records must have unique IDs")
    if any(record.phash is None for record in snapshot):
        raise BenchmarkInputError("every pHash benchmark record needs a pHash")
    entries: list[tuple[str, int]] = []
    for record in snapshot:
        if record.phash is None:  # pragma: no cover - guarded collectively above
            raise AssertionError("validated pHash unexpectedly missing")
        entries.append((record.id, record.phash))
    return tuple(sorted(entries))


def benchmark_phash_index(
    records: Iterable[ImageRecord],
    *,
    radius: int,
    brute_force_max_size: int,
) -> tuple[ScalingMetricRow, ...]:
    """Measure BK-tree build/query and verify small-N output against brute force."""

    if isinstance(radius, bool) or not isinstance(radius, int):
        raise TypeError("radius must be an integer")
    if not 0 <= radius <= 64:
        raise ValueError("radius must be between 0 and 64")
    if isinstance(brute_force_max_size, bool) or not isinstance(
        brute_force_max_size, int
    ):
        raise TypeError("brute_force_max_size must be an integer")
    if brute_force_max_size <= 0:
        raise ValueError("brute_force_max_size must be positive")
    snapshot = tuple(records)
    entries = _phash_entries(snapshot)
    if not entries:
        raise BenchmarkInputError("pHash benchmark requires at least one record")

    def build() -> BKTree:
        tree = BKTree()
        for record_id, phash in sorted(entries, key=lambda item: (item[1], item[0])):
            tree.add(phash, record_id)
        return tree

    built = measure_runtime(build)

    def query() -> tuple[PhashCandidatePair, ...]:
        pairs: dict[tuple[str, str], int] = {}
        for record_id, phash in entries:
            for match in built.value.search(phash, radius):
                if match.record_id == record_id:
                    continue
                left_id, right_id = sorted((record_id, match.record_id))
                pairs[(left_id, right_id)] = match.distance
        return tuple(
            PhashCandidatePair(left_id=left, right_id=right, distance=distance)
            for (left, right), distance in sorted(pairs.items())
        )

    queried = measure_runtime(query)
    rows = [
        ScalingMetricRow(
            dataset_size=len(entries),
            stage="phash_index_build",
            mode="bk_tree",
            duration_seconds=built.measurement.duration_seconds,
            peak_memory_bytes=built.measurement.peak_memory_bytes,
            memory_measurement_scope=built.measurement.memory_scope,
        ),
        ScalingMetricRow(
            dataset_size=len(entries),
            stage="phash_query",
            mode="bk_tree",
            duration_seconds=queried.measurement.duration_seconds,
            peak_memory_bytes=queried.measurement.peak_memory_bytes,
            memory_measurement_scope=queried.measurement.memory_scope,
        ),
    ]
    if len(entries) <= brute_force_max_size:
        brute = measure_runtime(lambda: brute_force_phash_pairs(snapshot, radius))
        if queried.value != brute.value:
            raise RuntimeError("BK-tree results disagree with the brute-force reference")
        rows.append(
            ScalingMetricRow(
                dataset_size=len(entries),
                stage="phash_query",
                mode="brute_force_reference_small_n_only",
                duration_seconds=brute.measurement.duration_seconds,
                peak_memory_bytes=brute.measurement.peak_memory_bytes,
                memory_measurement_scope=brute.measurement.memory_scope,
            )
        )
    return tuple(sorted(rows, key=lambda row: (row.stage, row.mode)))


def generate_synthetic_embeddings(
    dataset_size: int,
    dimension: int,
    *,
    seed: int,
) -> tuple[tuple[str, ...], Float32Array]:
    """Generate deterministic random unit vectors, explicitly not DINO features."""

    for value, name in ((dataset_size, "dataset_size"), (dimension, "dimension")):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(dataset_size, dimension)).astype(np.float32)
    matrix = normalize_embeddings(
        vectors,
        expected_rows=dataset_size,
        expected_dimension=dimension,
    )
    ids = tuple(
        stable_id("bench", str(seed), "embedding", str(index))
        for index in range(dataset_size)
    )
    return ids, matrix


def _benchmark_embedding_indexes(
    record_ids: Iterable[str],
    embeddings: np.ndarray[Any, Any],
    *,
    embedding_source: EmbeddingScaleSource,
    k: int,
    threads: int = 1,
    hnsw_m: int = 32,
    hnsw_ef_construction: int = 200,
    hnsw_ef_search: int = 64,
) -> EmbeddingScaleResult:
    """Measure FlatIP versus HNSW with an explicit vector-source label."""

    ids = tuple(record_ids)
    measured = measure_runtime(
        lambda: benchmark_exact_vs_hnsw(
            ids,
            embeddings,
            k=k,
            threads=threads,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
            hnsw_ef_search=hnsw_ef_search,
        )
    )
    comparison = measured.value
    exact_mode = f"{embedding_source}_flat_ip_exact"
    hnsw_mode = f"{embedding_source}_hnsw_approximate"
    rows = (
        ScalingMetricRow(
            dataset_size=comparison.dataset_size,
            stage="embedding_index_build",
            mode=exact_mode,
            duration_seconds=comparison.flat_build_seconds,
        ),
        ScalingMetricRow(
            dataset_size=comparison.dataset_size,
            stage="embedding_index_build",
            mode=hnsw_mode,
            duration_seconds=comparison.hnsw_build_seconds,
        ),
        ScalingMetricRow(
            dataset_size=comparison.dataset_size,
            stage="embedding_query",
            mode=exact_mode,
            duration_seconds=comparison.flat_query_seconds,
            recall_at_k=1.0,
        ),
        ScalingMetricRow(
            dataset_size=comparison.dataset_size,
            stage="embedding_query",
            mode=hnsw_mode,
            duration_seconds=comparison.hnsw_query_seconds,
            recall_at_k=comparison.recall_at_k,
        ),
        ScalingMetricRow(
            dataset_size=comparison.dataset_size,
            stage="embedding_flat_vs_hnsw_total",
            mode=embedding_source,
            duration_seconds=measured.measurement.duration_seconds,
            peak_memory_bytes=measured.measurement.peak_memory_bytes,
            memory_measurement_scope=measured.measurement.memory_scope,
            recall_at_k=comparison.recall_at_k,
        ),
    )
    return EmbeddingScaleResult(
        embedding_source=embedding_source,
        comparison=comparison,
        rows=rows,
    )


def benchmark_synthetic_embedding_indexes(
    record_ids: Iterable[str],
    embeddings: np.ndarray[Any, Any],
    *,
    k: int,
    threads: int = 1,
    hnsw_m: int = 32,
    hnsw_ef_construction: int = 200,
    hnsw_ef_search: int = 64,
) -> EmbeddingScaleResult:
    """Measure FlatIP versus HNSW on random synthetic vectors, never DINOv2."""

    return _benchmark_embedding_indexes(
        record_ids,
        embeddings,
        embedding_source="synthetic_random_unit_vectors_not_dinov2",
        k=k,
        threads=threads,
        hnsw_m=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
        hnsw_ef_search=hnsw_ef_search,
    )


def _embed_local_image_records(
    root: Path,
    records: Sequence[ImageRecord],
    embedder: ImageEmbedder,
    *,
    batch_size: int = 64,
) -> tuple[tuple[str, ...], Float32Array]:
    ordered = tuple(sorted(records, key=lambda record: record.id))
    chunks: list[Float32Array] = []
    for offset in range(0, len(ordered), batch_size):
        batch_records = ordered[offset : offset + batch_size]
        images: list[Image.Image] = []
        try:
            for record in batch_records:
                payload = _safe_relative_file(root, record.path).read_bytes()
                if hashlib.sha256(payload).hexdigest() != record.byte_sha256:
                    raise BenchmarkInputError("a scale benchmark image changed after validation")
                images.append(_decoded_rgb(payload))
            vectors = embedder.embed(tuple(images))
            chunks.append(
                normalize_embeddings(
                    vectors,
                    expected_rows=len(batch_records),
                    expected_dimension=embedder.dimension,
                )
            )
        finally:
            for image in images:
                image.close()
    matrix = (
        np.vstack(chunks)
        if chunks
        else np.empty((0, embedder.dimension), dtype=np.float32)
    )
    return tuple(record.id for record in ordered), matrix


def _benchmark_local_image_collection(
    dataset_size: int,
    *,
    seed: int,
    config: ScalingBenchmarkConfig,
) -> tuple[ScalingMetricRow, ...]:
    """Measure the local audit and retrieval path on generated image files."""

    with tempfile.TemporaryDirectory(prefix="splitguard-scale-images-") as temporary:
        root = Path(temporary) / "dataset"
        generated = measure_runtime(
            partial(_write_local_image_collection, root, dataset_size, seed)
        )
        manifest = measure_runtime(partial(discover_image_folder, root))
        validated = measure_runtime(
            partial(
                scan_images,
                root,
                list(manifest.value.entries),
                max_image_pixels=1_000_000,
            )
        )
        if validated.value.issues or len(validated.value.records) != dataset_size:
            raise RuntimeError("generated scale fixture failed image validation")
        fingerprinted = measure_runtime(
            partial(fingerprint_records, root, validated.value.records)
        )
        phash_rows = benchmark_phash_index(
            fingerprinted.value,
            radius=config.phash_radius,
            brute_force_max_size=config.brute_force_max_size,
        )
        scale_embedder = FakeEmbedder(
            dimension=config.embedding_dimension,
            seed=seed,
            preprocessing_version="scaling-fake-rgb-v1",
        )
        scale_provenance = EmbeddingProvenance(
            backend="fake",
            model_identity=scale_embedder.model_identity,
            model_revision=None,
            preprocessing_version=scale_embedder.preprocessing_version,
            device="cpu",
            is_synthetic=True,
        )
        _validate_embedder_provenance(scale_embedder, scale_provenance)
        embedded = measure_runtime(
            partial(
                _embed_local_image_records,
                root,
                validated.value.records,
                scale_embedder,
            )
        )
        embedding_ids, embedding_matrix = embedded.value
        embedding_benchmark = _benchmark_embedding_indexes(
            embedding_ids,
            embedding_matrix,
            embedding_source=config.embedding_source,
            k=config.k,
            threads=config.threads,
            hnsw_m=config.hnsw_m,
            hnsw_ef_construction=config.hnsw_ef_construction,
            hnsw_ef_search=config.hnsw_ef_search,
        )

        rows = [
            ScalingMetricRow(
                dataset_size=dataset_size,
                stage="local_image_generation",
                mode="deterministic_generated_png_files",
                duration_seconds=generated.measurement.duration_seconds,
                peak_memory_bytes=generated.measurement.peak_memory_bytes,
                memory_measurement_scope=generated.measurement.memory_scope,
            ),
            ScalingMetricRow(
                dataset_size=dataset_size,
                stage="manifest_discovery",
                mode="imagefolder_local_files",
                duration_seconds=manifest.measurement.duration_seconds,
                peak_memory_bytes=manifest.measurement.peak_memory_bytes,
                memory_measurement_scope=manifest.measurement.memory_scope,
            ),
            ScalingMetricRow(
                dataset_size=dataset_size,
                stage="image_validation_sha256",
                mode="pillow_decode_and_sha256_local_files",
                duration_seconds=validated.measurement.duration_seconds,
                peak_memory_bytes=validated.measurement.peak_memory_bytes,
                memory_measurement_scope=validated.measurement.memory_scope,
            ),
            ScalingMetricRow(
                dataset_size=dataset_size,
                stage="phash_computation",
                mode="internal_64bit_dct_local_files",
                duration_seconds=fingerprinted.measurement.duration_seconds,
                peak_memory_bytes=fingerprinted.measurement.peak_memory_bytes,
                memory_measurement_scope=fingerprinted.measurement.memory_scope,
            ),
            ScalingMetricRow(
                dataset_size=dataset_size,
                stage="image_embedding",
                mode=config.embedding_source,
                duration_seconds=embedded.measurement.duration_seconds,
                peak_memory_bytes=embedded.measurement.peak_memory_bytes,
                memory_measurement_scope=embedded.measurement.memory_scope,
            ),
            *phash_rows,
            *embedding_benchmark.rows,
        ]
        audit_rows = [
            row
            for row in rows
            if row.stage
            in {
                "manifest_discovery",
                "image_validation_sha256",
                "phash_computation",
                "phash_index_build",
                "phash_query",
                "image_embedding",
                "embedding_index_build",
                "embedding_query",
            }
            and row.mode != "brute_force_reference_small_n_only"
        ]
        rows.append(
            ScalingMetricRow(
                dataset_size=dataset_size,
                stage="total_local_audit",
                mode="sum_manifest_validation_phash_bk_embedding_and_faiss_stages",
                duration_seconds=sum(row.duration_seconds for row in audit_rows),
            )
        )
        return tuple(sorted(rows, key=lambda row: (row.stage, row.mode)))


def run_scaling_benchmarks(config: BenchmarkConfig) -> tuple[ScalingMetricRow, ...]:
    """Run local-image audit and pixel-derived fake-embedding retrieval stages."""

    rows: list[ScalingMetricRow] = []
    scaling = config.scaling
    for dataset_size in scaling.dataset_sizes:
        rows.extend(
            _benchmark_local_image_collection(
                dataset_size,
                seed=config.seed,
                config=scaling,
            )
        )
    return tuple(
        sorted(rows, key=lambda row: (row.dataset_size, row.stage, row.mode))
    )


def build_scaling_artifact(
    metadata: RunMetadata,
    rows: Iterable[ScalingMetricRow],
) -> ScalingBenchmarkArtifact:
    """Build a canonically ordered, schema-versioned scaling artifact."""

    ordered = tuple(
        sorted(rows, key=lambda row: (row.dataset_size, row.stage, row.mode))
    )
    keys = tuple((row.dataset_size, row.stage, row.mode) for row in ordered)
    if len(keys) != len(set(keys)):
        raise BenchmarkInputError("scaling metric rows must be unique")
    return ScalingBenchmarkArtifact(metadata=metadata, rows=ordered)


__all__ = [
    "BenchmarkConfig",
    "BenchmarkInputError",
    "DetectionBenchmarkRun",
    "DetectionObservation",
    "DetectionSweepConfig",
    "EmbeddingProvenance",
    "EmbeddingScaleResult",
    "ImageEmbedder",
    "MeasuredCall",
    "RuntimeMeasurement",
    "ScalingBenchmarkConfig",
    "benchmark_phash_index",
    "benchmark_synthetic_embedding_indexes",
    "build_detection_artifact",
    "build_detection_embedder",
    "build_scaling_artifact",
    "detection_benchmark_rows",
    "evaluate_combined_policy",
    "evaluate_embedding_thresholds",
    "evaluate_exact_detection",
    "evaluate_phash_thresholds",
    "generate_synthetic_embeddings",
    "generate_synthetic_phash_records",
    "load_benchmark_config",
    "measure_runtime",
    "observe_synthetic_corruptions",
    "run_detection_benchmark",
    "run_scaling_benchmarks",
]
