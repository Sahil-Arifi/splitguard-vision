from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image
from pydantic import ValidationError

import splitguard.benchmark as benchmark_module
from splitguard.benchmark import (
    BenchmarkConfig,
    BenchmarkInputError,
    DetectionBenchmarkRun,
    DetectionObservation,
    DetectionSweepConfig,
    EmbeddingProvenance,
    ScalingBenchmarkConfig,
    benchmark_phash_index,
    benchmark_synthetic_embedding_indexes,
    build_detection_artifact,
    build_detection_embedder,
    build_scaling_artifact,
    detection_benchmark_rows,
    evaluate_combined_policy,
    evaluate_embedding_thresholds,
    evaluate_exact_detection,
    evaluate_phash_thresholds,
    generate_synthetic_embeddings,
    generate_synthetic_phash_records,
    load_benchmark_config,
    measure_runtime,
    observe_synthetic_corruptions,
    run_detection_benchmark,
    run_scaling_benchmarks,
)
from splitguard.models.embedder import FakeEmbedder
from splitguard.schemas import ImageRecord, RunMetadata, Split, stable_id
from splitguard.synthetic import CorruptionType, generate_controlled_corruptions


def pair_ids(left: str, right: str) -> tuple[str, str]:
    left_id, right_id = sorted((stable_id("img", left), stable_id("img", right)))
    return left_id, right_id


def fake_provenance() -> EmbeddingProvenance:
    return EmbeddingProvenance(
        backend="fake",
        model_identity="fake:sha256-expand-v1:seed=7:dimension=16:l2=float32-v1",
        model_revision=None,
        preprocessing_version="benchmark-fake-rgb-v1",
        device="cpu",
        is_synthetic=True,
    )


class CountingFakeEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__(
            dimension=8,
            seed=23,
            preprocessing_version="benchmark-fake-rgb-v1",
        )
        self.image_count = 0

    def embed(self, images: Sequence[Image.Image]) -> npt.NDArray[np.float32]:
        self.image_count += len(images)
        return super().embed(images)


class GenericCountingEmbedder:
    def __init__(self, model_identity: str = "custom:pixel-counting-v1") -> None:
        self.model_identity = model_identity
        self.preprocessing_version = "custom-rgb-v1"
        self.dimension = 8
        self.device = "cpu"
        self.image_count = 0
        self._delegate = FakeEmbedder(dimension=self.dimension, seed=29)

    def embed(self, images: Sequence[Image.Image]) -> npt.NDArray[np.float32]:
        self.image_count += len(images)
        return self._delegate.embed(images)


def observation(
    left: str,
    right: str,
    corruption_type: str,
    *,
    duplicate: bool,
    exact: bool = False,
    phash: int | None = None,
    cosine: float | None = None,
) -> DetectionObservation:
    left_id, right_id = pair_ids(left, right)
    return DetectionObservation(
        left_id=left_id,
        right_id=right_id,
        corruption_type=corruption_type,
        is_duplicate=duplicate,
        exact_match=exact,
        phash_distance=phash,
        cosine_similarity=cosine,
        embedding_provenance=fake_provenance() if cosine is not None else None,
    )


def sample_observations() -> tuple[DetectionObservation, ...]:
    return (
        observation(
            "exact-source",
            "exact-derived",
            "exact_copy",
            duplicate=True,
            exact=True,
            phash=0,
            cosine=1.0,
        ),
        observation(
            "exact-negative-a",
            "exact-negative-b",
            "exact_copy",
            duplicate=False,
            phash=20,
            cosine=0.1,
        ),
        observation(
            "jpeg-source",
            "jpeg-derived",
            "jpeg_recompression",
            duplicate=True,
            phash=3,
            cosine=0.96,
        ),
        observation(
            "jpeg-negative-a",
            "jpeg-negative-b",
            "jpeg_recompression",
            duplicate=False,
            phash=1,
            cosine=0.97,
        ),
    )


def metadata() -> RunMetadata:
    return RunMetadata(
        timestamp=datetime(2026, 9, 3, tzinfo=UTC),
        git_commit_sha=None,
        git_dirty=None,
        python_version="3.11.9",
        os="test-os",
        cpu="test-cpu",
        cuda_available=False,
        gpu_model=None,
        package_versions=(),
        configuration_sha256="1" * 64,
        dataset_manifest_sha256="2" * 64,
        random_seeds=(7,),
    )


def make_source(
    root: Path,
    name: str = "pattern",
    split: Split = Split.TRAIN,
    label: str = "cat",
) -> ImageRecord:
    relative = f"{split.value}/{label}/{name}.png"
    path = root / relative
    path.parent.mkdir(parents=True)
    array = np.zeros((32, 40, 3), dtype=np.uint8)
    rows, columns = np.indices(array.shape[:2])
    name_offset = sum(name.encode("utf-8")) % 256
    array[..., 0] = (columns * 9 + rows * 2 + name_offset) % 256
    array[..., 1] = (columns * 3 + rows * 7 + name_offset * 3) % 256
    array[..., 2] = (columns * 5 + rows * 11 + name_offset * 5) % 256
    Image.fromarray(array).save(path, format="PNG")
    payload = path.read_bytes()
    return ImageRecord(
        id=stable_id("img", relative),
        path=relative,
        split=split,
        label=label,
        byte_sha256=hashlib.sha256(payload).hexdigest(),
        byte_size=len(payload),
        width=40,
        height=32,
        format="png",
    )


def test_detector_layers_report_each_corruption_separately() -> None:
    observations = tuple(reversed(sample_observations()))

    exact_rows = evaluate_exact_detection(observations)
    exact_by_type = {row.corruption_type: row.metrics for row in exact_rows}
    assert exact_by_type["exact_copy"].f1 == 1.0
    assert exact_by_type["jpeg_recompression"].false_negatives == 1

    phash_rows = evaluate_phash_thresholds(observations, (4, 2, 4))
    jpeg_phash = [row for row in phash_rows if row.corruption_type == "jpeg_recompression"]
    assert [row.threshold for row in jpeg_phash] == [2.0, 4.0]
    assert jpeg_phash[0].metrics == jpeg_phash[0].metrics.from_counts(0, 1, 1)
    assert jpeg_phash[1].metrics == jpeg_phash[1].metrics.from_counts(1, 1, 0)

    embedding_rows = evaluate_embedding_thresholds(observations, (0.95, 0.98))
    jpeg_embedding = [
        row for row in embedding_rows if row.corruption_type == "jpeg_recompression"
    ]
    assert [row.threshold for row in jpeg_embedding] == [0.98, 0.95]
    assert jpeg_embedding[0].metrics.false_negatives == 1
    assert jpeg_embedding[1].metrics.false_positives == 1
    assert all("not_dinov2" in row.detector for row in embedding_rows)


def test_combined_policy_keeps_embedding_only_pairs_review_only_by_default() -> None:
    semantic = observation(
        "semantic-source",
        "semantic-derived",
        "brightness_shift",
        duplicate=True,
        phash=20,
        cosine=0.99,
    )

    review_only = evaluate_combined_policy(
        (semantic,),
        phash_threshold=8,
        cosine_threshold=0.95,
    )[0]
    promoted = evaluate_combined_policy(
        (semantic,),
        phash_threshold=8,
        cosine_threshold=0.95,
        embedding_only_is_duplicate=True,
    )[0]

    assert review_only.metrics.false_negatives == 1
    assert promoted.metrics.true_positives == 1
    assert "review_only" in review_only.detector
    assert "not_dinov2" in promoted.detector


def test_complete_detection_rows_are_canonical_and_artifact_serializes() -> None:
    rows = detection_benchmark_rows(
        sample_observations(),
        DetectionSweepConfig(
            phash_thresholds=(0, 4),
            cosine_thresholds=(0.99, 0.9),
            combined_phash_threshold=4,
            combined_cosine_threshold=0.9,
        ),
    )
    artifact = build_detection_artifact(metadata(), reversed(rows), fake_provenance())
    payload = json.loads(artifact.model_dump_json())

    assert artifact.rows == rows
    assert artifact.embedding_provenance == fake_provenance()
    assert payload["artifact_type"] == "detection_benchmark"
    assert payload["embedding_provenance"]["backend"] == "fake"
    assert len({(row.detector, row.corruption_type, row.threshold) for row in rows}) == len(
        rows
    )
    assert {row.corruption_type for row in rows} == {
        "exact_copy",
        "jpeg_recompression",
    }
    _, dino_provenance = build_detection_embedder(
        DetectionSweepConfig(
            embedding_backend="dinov2",
            embedding_model="facebook/dinov2-small",
            embedding_revision="a" * 40,
            embedding_device="cpu",
        )
    )
    with pytest.raises(ValidationError, match="must match artifact provenance"):
        build_detection_artifact(metadata(), rows, dino_provenance)


def test_synthetic_observation_pipeline_is_offline_and_truth_remains_independent(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source = make_source(source_root)
    generated_root = tmp_path / "generated"
    corruption_set = generate_controlled_corruptions(
        source_root,
        (source,),
        generated_root,
        seed=13,
        corruption_types=(
            CorruptionType.EXACT_COPY,
            CorruptionType.RESIZE,
            CorruptionType.MALFORMED_FILE,
        ),
    )

    observations = observe_synthetic_corruptions(
        source_root,
        generated_root,
        corruption_set,
        fake_embedding_dimension=12,
    )
    by_type = {item.corruption_type: item for item in observations}

    assert by_type["exact_copy"].exact_match is True
    assert by_type["exact_copy"].is_duplicate is True
    assert by_type["resize"].exact_match is False
    assert by_type["resize"].is_duplicate is True
    assert by_type["malformed_file"].is_duplicate is False
    assert by_type["malformed_file"].phash_distance is None
    assert all(
        item.embedding_source == "synthetic_fake_embedding_cosine_not_dinov2"
        for item in observations
        if item.cosine_similarity is not None
    )
    assert all(
        item.embedding_provenance is None
        or item.embedding_provenance.backend == "fake"
        for item in observations
    )
    assert not evaluate_exact_detection((by_type["malformed_file"],))
    assert not evaluate_phash_thresholds((by_type["malformed_file"],), (8,))
    assert not evaluate_embedding_thresholds((by_type["malformed_file"],), (0.9,))


@pytest.mark.parametrize("mutated_target", ["source", "valid_derived", "malformed"])
def test_observation_refuses_any_mutated_fixture_payload(
    tmp_path: Path,
    mutated_target: str,
) -> None:
    source_root = tmp_path / "source"
    source = make_source(source_root)
    generated_root = tmp_path / "generated"
    corruption_set = generate_controlled_corruptions(
        source_root,
        (source,),
        generated_root,
        corruption_types=(CorruptionType.EXACT_COPY, CorruptionType.MALFORMED_FILE),
    )
    if mutated_target == "source":
        target = source_root / source.path
    else:
        corruption = (
            CorruptionType.EXACT_COPY
            if mutated_target == "valid_derived"
            else CorruptionType.MALFORMED_FILE
        )
        injection = next(
            item for item in corruption_set.injections if item.corruption_type is corruption
        )
        target = generated_root / injection.derived_path
    target.write_bytes(target.read_bytes() + b"tampered")

    with pytest.raises(BenchmarkInputError, match="payload changed"):
        observe_synthetic_corruptions(source_root, generated_root, corruption_set)


def test_observation_embeds_each_distinct_decoded_image_once_and_checks_provenance(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source = make_source(source_root)
    generated_root = tmp_path / "generated"
    corruption_set = generate_controlled_corruptions(
        source_root,
        (source,),
        generated_root,
        corruption_types=(
            CorruptionType.EXACT_COPY,
            CorruptionType.CROSS_SPLIT_COPY,
            CorruptionType.CROSS_LABEL_DUPLICATE,
        ),
    )
    embedder = CountingFakeEmbedder()
    provenance = EmbeddingProvenance(
        backend="fake",
        model_identity=embedder.model_identity,
        model_revision=None,
        preprocessing_version=embedder.preprocessing_version,
        device="cpu",
        is_synthetic=True,
    )

    observations = observe_synthetic_corruptions(
        source_root,
        generated_root,
        corruption_set,
        embedder=embedder,
        embedding_provenance=provenance,
    )

    assert len(observations) == 3
    assert embedder.image_count == 1
    mismatched = provenance.model_copy(update={"preprocessing_version": "wrong-v1"})
    untouched_embedder = CountingFakeEmbedder()
    with pytest.raises(BenchmarkInputError, match="preprocessing"):
        observe_synthetic_corruptions(
            source_root,
            generated_root,
            corruption_set,
            embedder=untouched_embedder,
            embedding_provenance=mismatched,
        )
    assert untouched_embedder.image_count == 0

    generic = GenericCountingEmbedder()
    custom_provenance = EmbeddingProvenance(
        backend="custom",
        model_identity=generic.model_identity,
        model_revision=None,
        preprocessing_version=generic.preprocessing_version,
        device="cpu",
        is_synthetic=True,
    )
    observe_synthetic_corruptions(
        source_root,
        generated_root,
        corruption_set,
        embedder=generic,
        embedding_provenance=custom_provenance,
    )
    assert generic.image_count == 1
    wrong_device = custom_provenance.model_copy(update={"device": "cuda"})
    with pytest.raises(BenchmarkInputError, match="matching device"):
        observe_synthetic_corruptions(
            source_root,
            generated_root,
            corruption_set,
            embedder=GenericCountingEmbedder(),
            embedding_provenance=wrong_device,
        )
    fake_named_generic = GenericCountingEmbedder("fake:generic-counting-v1")
    fake_backend_claim = EmbeddingProvenance(
        backend="fake",
        model_identity=fake_named_generic.model_identity,
        model_revision=None,
        preprocessing_version=fake_named_generic.preprocessing_version,
        device="cpu",
        is_synthetic=True,
    )
    with pytest.raises(BenchmarkInputError, match="custom provenance"):
        observe_synthetic_corruptions(
            source_root,
            generated_root,
            corruption_set,
            embedder=fake_named_generic,
            embedding_provenance=fake_backend_claim,
        )


def test_multi_source_observations_include_independent_negative_controls(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    sources = (
        make_source(source_root, "first", Split.TRAIN, "cat"),
        make_source(source_root, "second", Split.TEST, "dog"),
    )
    generated_root = tmp_path / "generated"
    corruption_set = generate_controlled_corruptions(
        source_root,
        sources,
        generated_root,
        seed=21,
        corruption_types=(
            CorruptionType.EXACT_COPY,
            CorruptionType.RESIZE,
            CorruptionType.MALFORMED_FILE,
        ),
    )

    observations = observe_synthetic_corruptions(
        source_root,
        generated_root,
        corruption_set,
        fake_embedding_dimension=12,
    )

    assert len(corruption_set.injections) == 6
    assert len(corruption_set.negative_controls) == 4
    assert len(observations) == 10
    for corruption_type in ("exact_copy", "resize"):
        group = [
            item for item in observations if item.corruption_type == corruption_type
        ]
        assert sum(item.is_duplicate for item in group) == 2
        assert sum(not item.is_duplicate for item in group) == 2
    exact_metrics = evaluate_exact_detection(observations)
    exact_copy = next(
        row for row in exact_metrics if row.corruption_type == "exact_copy"
    )
    assert exact_copy.metrics.true_positives == 2
    assert exact_copy.metrics.false_positives == 0
    assert exact_copy.metrics.precision == 1.0


def test_configured_dinov2_embedder_is_pinned_lazy_and_truthfully_labeled() -> None:
    revision = "a" * 40
    config = DetectionSweepConfig(
        embedding_backend="dinov2",
        embedding_model="facebook/dinov2-small",
        embedding_revision=revision,
        embedding_device="cpu",
        embedding_batch_size=2,
    )

    embedder, provenance = build_detection_embedder(config)

    assert f"@{revision}" in embedder.model_identity
    assert provenance.backend == "dinov2"
    assert provenance.model_revision == revision
    assert provenance.device == "cpu"
    assert provenance.is_synthetic is False
    assert provenance.detector_name == f"dinov2_embedding_cosine@{revision[:12]}"


def test_high_level_detection_runner_is_complete_deterministic_and_privacy_safe(
    tmp_path: Path,
) -> None:
    config = BenchmarkConfig(
        seed=31,
        detection=DetectionSweepConfig(
            phash_thresholds=(0,),
            cosine_thresholds=(0.9,),
            embedding_batch_size=3,
            source_count=2,
        ),
        scaling=ScalingBenchmarkConfig(dataset_sizes=(4,), k=2),
    )

    first = run_detection_benchmark(config, workspace=tmp_path / "first")
    second = run_detection_benchmark(config, workspace=tmp_path / "second")

    assert isinstance(first, DetectionBenchmarkRun)
    assert first.dataset_sha256 == second.dataset_sha256
    assert first.source_records == second.source_records
    assert first.corruption_set == second.corruption_set
    assert first.observations == second.observations
    assert first.rows == second.rows
    assert len(first.corruption_set.injections) == len(CorruptionType) * 2
    assert len(first.corruption_set.negative_controls) == (len(CorruptionType) - 1) * 2
    assert len(first.observations) == 34
    assert CorruptionType.MALFORMED_FILE.value not in {
        row.corruption_type for row in first.rows
    }
    assert (tmp_path / "first" / "splitguard-detection-benchmark").is_dir()
    serialized = first.model_dump_json()
    assert str(tmp_path) not in serialized
    assert "synthetic_fake_embedding_cosine_not_dinov2" in serialized
    forged = first.model_dump()
    forged["dataset_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="must bind"):
        DetectionBenchmarkRun.model_validate(forged)

    changed_seed_config = config.model_copy(update={"seed": 32})
    changed_seed = run_detection_benchmark(
        changed_seed_config,
        workspace=tmp_path / "changed-seed",
    )
    assert changed_seed.dataset_sha256 != first.dataset_sha256


def test_phash_scale_benchmark_matches_brute_force_only_at_small_n() -> None:
    records = generate_synthetic_phash_records(72, seed=5)

    compared = benchmark_phash_index(records, radius=3, brute_force_max_size=72)
    indexed_only = benchmark_phash_index(records, radius=3, brute_force_max_size=10)

    assert {row.mode for row in compared} == {
        "bk_tree",
        "brute_force_reference_small_n_only",
    }
    assert all(row.dataset_size == 72 for row in compared)
    assert all(row.duration_seconds >= 0.0 for row in compared)
    assert all(row.peak_memory_bytes is not None for row in compared)
    assert "brute_force_reference_small_n_only" not in {
        row.mode for row in indexed_only
    }


def test_flat_vs_hnsw_uses_explicit_synthetic_provenance_and_measured_recall() -> None:
    ids, vectors = generate_synthetic_embeddings(80, 16, seed=19)

    result = benchmark_synthetic_embedding_indexes(
        ids,
        vectors,
        k=5,
        threads=1,
        hnsw_m=8,
        hnsw_ef_construction=40,
        hnsw_ef_search=24,
    )

    assert result.embedding_source == "synthetic_random_unit_vectors_not_dinov2"
    assert 0.0 <= result.comparison.recall_at_k <= 1.0
    assert all("synthetic" in row.mode for row in result.rows)
    assert all("not_dinov2" in row.mode for row in result.rows)
    hnsw_query = next(
        row
        for row in result.rows
        if row.stage == "embedding_query" and "hnsw" in row.mode
    )
    assert hnsw_query.recall_at_k == result.comparison.recall_at_k


def test_configured_scale_run_and_artifact_are_canonical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = BenchmarkConfig(
        seed=3,
        detection=DetectionSweepConfig(),
        scaling=ScalingBenchmarkConfig(
            dataset_sizes=(24,),
            brute_force_max_size=24,
            phash_radius=2,
            embedding_dimension=8,
            k=3,
            threads=1,
            hnsw_m=4,
            hnsw_ef_construction=20,
            hnsw_ef_search=12,
        ),
    )

    def random_vectors_are_forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("configured scaling must embed the generated image pixels")

    monkeypatch.setattr(
        benchmark_module,
        "generate_synthetic_embeddings",
        random_vectors_are_forbidden,
    )
    rows = run_scaling_benchmarks(config)
    artifact = build_scaling_artifact(metadata(), reversed(rows))

    assert artifact.rows == rows
    assert json.loads(artifact.model_dump_json())["artifact_type"] == "scaling_benchmark"
    local_stages = {
        "local_image_generation",
        "manifest_discovery",
        "image_validation_sha256",
        "phash_computation",
        "phash_index_build",
        "phash_query",
        "image_embedding",
        "total_local_audit",
    }
    assert local_stages <= {row.stage for row in rows}
    assert all(
        row.memory_measurement_scope == "python_allocations_via_tracemalloc"
        for row in rows
        if row.peak_memory_bytes is not None
    )
    total = next(row for row in rows if row.stage == "total_local_audit")
    assert total.peak_memory_bytes is None
    component_stages = {
        "manifest_discovery",
        "image_validation_sha256",
        "phash_computation",
        "phash_index_build",
        "phash_query",
        "image_embedding",
        "embedding_index_build",
        "embedding_query",
    }
    expected_total = sum(
        row.duration_seconds
        for row in rows
        if row.stage in component_stages
        and row.mode != "brute_force_reference_small_n_only"
    )
    assert total.duration_seconds == pytest.approx(expected_total)
    assert any(
        row.mode == "brute_force_reference_small_n_only" for row in rows
    )
    assert any(row.stage == "embedding_query" for row in rows)
    assert any(row.recall_at_k is not None for row in rows)
    assert all(
        "not_dinov2" in row.mode
        for row in rows
        if row.stage.startswith("embedding") or row.stage == "image_embedding"
    )


def test_committed_benchmark_configuration_is_strict_and_explicit() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "benchmark.yaml"

    config = load_benchmark_config(config_path)

    assert config.scaling.dataset_sizes == (1_000, 5_000, 10_000)
    assert config.scaling.brute_force_max_size == 1_000
    assert config.scaling.embedding_source == "pixel_derived_fake_embeddings_not_dinov2"
    assert config.detection.embedding_only_is_duplicate is False
    assert config.detection.embedding_backend == "dinov2"
    assert config.detection.embedding_model == "facebook/dinov2-small"
    assert config.detection.embedding_revision == "ed25f3a31f01632728cabb09d1542f84ab7b0056"
    assert config.detection.embedding_device == "auto"
    assert config.detection.embedding_batch_size == 32
    assert config.detection.source_count == 16


def test_invalid_config_thresholds_and_duplicate_observations_are_rejected(
    tmp_path: Path,
) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("unknown: true\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="extra_forbidden"):
        load_benchmark_config(invalid)
    with pytest.raises(ValueError, match="sorted and unique"):
        DetectionSweepConfig(phash_thresholds=(4, 2))
    with pytest.raises(ValueError, match="reverse-sorted"):
        DetectionSweepConfig(cosine_thresholds=(0.8, 0.9))
    with pytest.raises(ValueError, match="immutable"):
        DetectionSweepConfig(
            embedding_backend="dinov2",
            embedding_model="facebook/dinov2-small",
        )
    with pytest.raises(ValueError, match="must be exactly"):
        DetectionSweepConfig(embedding_model="facebook/dinov2-small")
    normalized = DetectionSweepConfig.model_validate(
        {
            "phash_thresholds": [0, 4],
            "cosine_thresholds": [0.99, 0.9],
            "source_count": 2,
        }
    )
    assert normalized.phash_thresholds == (0, 4)
    assert normalized.cosine_thresholds == (0.99, 0.9)
    with pytest.raises(ValidationError, match="int_type"):
        DetectionSweepConfig.model_validate({"source_count": "2"})
    invalid_nested = tmp_path / "invalid-nested.yaml"
    invalid_nested.write_text("detection:\n  unknown: true\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="extra_forbidden"):
        load_benchmark_config(invalid_nested)
    item = sample_observations()[0]
    with pytest.raises(BenchmarkInputError, match="unique"):
        evaluate_exact_detection((item, item))


@pytest.mark.parametrize(
    "call,error",
    [
        (lambda: evaluate_phash_thresholds((), (True,)), TypeError),
        (lambda: evaluate_phash_thresholds((), (65,)), ValueError),
        (lambda: evaluate_embedding_thresholds((), (float("nan"),)), ValueError),
        (lambda: generate_synthetic_embeddings(0, 4, seed=0), ValueError),
        (lambda: generate_synthetic_phash_records(1, seed=True), TypeError),
    ],
)
def test_benchmark_inputs_are_validated(
    call: object,
    error: type[Exception],
) -> None:
    assert callable(call)
    with pytest.raises(error):
        call()


def test_runtime_measurement_labels_python_memory_scope() -> None:
    result = measure_runtime(lambda: [index for index in range(100)])

    assert result.value == list(range(100))
    assert result.measurement.duration_seconds >= 0.0
    assert result.measurement.peak_memory_bytes >= 0
    assert result.measurement.memory_scope == "python_allocations_via_tracemalloc"
