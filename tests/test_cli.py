"""CLI bootstrap tests."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Literal, Never

import numpy as np
import pytest
from click import unstyle
from PIL import Image
from pytest import MonkeyPatch
from typer.testing import CliRunner

import splitguard.cli as cli_module
import splitguard.training as training_module
from splitguard.benchmark import load_benchmark_config
from splitguard.cli import app
from splitguard.config import load_config
from splitguard.repair import MaterializationResult
from splitguard.schemas import (
    AuditArtifact,
    DetectionBenchmarkArtifact,
    RepairArtifact,
    ScalingBenchmarkArtifact,
    Split,
    TrainingArtifact,
    TrainingCondition,
    canonical_json,
    canonical_sha256,
)
from splitguard.training import CifarExperimentConfig
from splitguard.training import run_cifar_experiment as run_cifar_experiment_core

runner = CliRunner()


def test_root_help_is_useful() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Audit and repair image dataset split leakage" in result.stdout
    assert "benchmark-detection" in result.stdout
    assert "benchmark-scale" in result.stdout
    assert "experiment" in result.stdout
    assert "version" in result.stdout
    assert "repair" in result.stdout
    assert "materialize" in result.stdout


def test_version_supports_short_output() -> None:
    result = runner.invoke(app, ["version", "--short"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"


def _write_small_benchmark_config(path: Path) -> Path:
    path.write_text(
        """\
seed: 17
detection:
  phash_thresholds: [0, 4]
  cosine_thresholds: [0.99, 0.80]
  combined_phash_threshold: 4
  combined_cosine_threshold: 0.90
  embedding_only_is_duplicate: false
  embedding_backend: fake
  embedding_model: fake:sha256-expand-v1
  embedding_revision: null
  embedding_device: cpu
  embedding_batch_size: 4
  source_count: 4
scaling:
  dataset_sizes: [8]
  brute_force_max_size: 8
  phash_radius: 4
  embedding_source: pixel_derived_fake_embeddings_not_dinov2
  embedding_dimension: 8
  k: 2
  threads: 1
  hnsw_m: 4
  hnsw_ef_construction: 8
  hnsw_ef_search: 4
""",
        encoding="utf-8",
    )
    return path


def test_benchmark_commands_have_useful_help() -> None:
    for command, purpose in (
        ("benchmark-detection", "precision, recall, and F1"),
        ("benchmark-scale", "synthetic neighbor scaling"),
    ):
        result = runner.invoke(app, [command, "--help"])
        output = unstyle(result.stdout)

        assert result.exit_code == 0
        assert purpose in output
        assert "--config" in output
        assert "--output" in output


def test_benchmark_detection_runs_offline_fake_backend_and_serializes(
    tmp_path: Path,
) -> None:
    config_path = _write_small_benchmark_config(tmp_path / "benchmark.yaml")
    output = tmp_path / "private-results" / "detection.json"
    config = load_benchmark_config(config_path)
    expected_config_hash = canonical_sha256(config.model_dump(mode="json"))

    result = runner.invoke(
        app,
        [
            "benchmark-detection",
            "--config",
            str(config_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    artifact = DetectionBenchmarkArtifact.model_validate_json(output.read_bytes())
    assert artifact.metadata.configuration_sha256 == expected_config_hash
    assert artifact.metadata.random_seeds == (17,)
    assert artifact.metadata.git_commit_sha is not None
    assert artifact.embedding_provenance.backend == "fake"
    assert artifact.embedding_provenance.is_synthetic is True
    assert artifact.rows
    embedding_rows = [
        row
        for row in artifact.rows
        if row.detector.startswith("synthetic_fake_embedding")
    ]
    assert embedding_rows
    assert all("synthetic_fake" in row.detector for row in embedding_rows)

    success = json.loads(result.stdout)
    assert success["artifact"] == output.name
    assert success["configuration_sha256"] == expected_config_hash
    assert success["dataset_manifest_sha256"] == artifact.metadata.dataset_manifest_sha256
    assert success["embedding_backend"] == "fake"
    assert success["embedding_is_synthetic"] is True
    assert success["metric_rows"] == len(artifact.rows)
    assert success["observations"] > 0
    assert success["seed"] == 17
    assert success["source_count"] == 4
    assert str(tmp_path) not in result.stdout
    assert str(tmp_path) not in output.read_text(encoding="utf-8")
    assert not tuple(output.parent.glob("*.tmp"))


def test_benchmark_scale_runs_small_n_and_labels_synthetic_vectors(
    tmp_path: Path,
) -> None:
    config_path = _write_small_benchmark_config(tmp_path / "benchmark.yaml")
    output = tmp_path / "private-results" / "scaling.json"
    config = load_benchmark_config(config_path)
    expected_config_hash = canonical_sha256(config.model_dump(mode="json"))
    expected_fixture_hash = canonical_sha256(
        {
            "dataset_sizes": [8],
            "embedding_dimension": 8,
            "embedding_source": "pixel_derived_fake_embeddings_not_dinov2",
            "fixture_schema": "splitguard-scaling-fixtures-v1",
            "local_image_generator": "deterministic_generated_png_files-v1",
            "seed": 17,
        }
    )

    result = runner.invoke(
        app,
        [
            "benchmark-scale",
            "--config",
            str(config_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    artifact = ScalingBenchmarkArtifact.model_validate_json(output.read_bytes())
    assert artifact.metadata.configuration_sha256 == expected_config_hash
    assert artifact.metadata.dataset_manifest_sha256 == expected_fixture_hash
    assert artifact.metadata.random_seeds == (17,)
    assert artifact.metadata.git_commit_sha is not None
    assert artifact.rows
    assert {row.dataset_size for row in artifact.rows} == {8}
    assert any(row.stage == "local_image_generation" for row in artifact.rows)
    assert any(
        row.mode == "pixel_derived_fake_embeddings_not_dinov2"
        for row in artifact.rows
    )
    assert any(row.recall_at_k is not None for row in artifact.rows)

    success = json.loads(result.stdout)
    assert success["artifact"] == output.name
    assert success["configuration_sha256"] == expected_config_hash
    assert success["dataset_manifest_sha256"] == expected_fixture_hash
    assert success["dataset_sizes"] == [8]
    assert success["embedding_source"] == "pixel_derived_fake_embeddings_not_dinov2"
    assert success["metric_rows"] == len(artifact.rows)
    assert success["seed"] == 17
    assert str(tmp_path) not in result.stdout
    assert str(tmp_path) not in output.read_text(encoding="utf-8")
    assert not tuple(output.parent.glob("*.tmp"))


def test_benchmark_config_errors_are_private_and_leave_no_partial_output(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "private" / "invalid.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        f"unknown_private_path: {tmp_path.as_posix()}\n",
        encoding="utf-8",
    )
    output = tmp_path / "results" / "detection.json"

    result = runner.invoke(
        app,
        [
            "benchmark-detection",
            "--config",
            str(config_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    assert "benchmark failed strict local" in result.output
    assert "validation" in result.output
    assert "Traceback" not in result.output
    assert str(tmp_path) not in result.output
    assert not output.exists()


def _write_small_experiment_config(path: Path) -> Path:
    path.write_text(
        """\
data:
  data_dir: private/cifar
  download: true
  num_classes: 2
  train_per_class: 2
  validation_per_class: 1
  test_per_class: 1
  contamination_per_class: 1
  sampling_seed: 19
  contamination: resize
repair:
  split_size_weight: 1.0
  class_balance_weight: 1.0
  seed: 23
  local_iterations: 10
training:
  seeds: [3]
  epochs: 1
  batch_size: 2
  learning_rate: 0.001
  weight_decay: 0.0
  device: cpu
  num_workers: 0
  augmentation: none
""",
        encoding="utf-8",
    )
    return path


def _generated_cifar_arrays() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(20260903)
    training_images: list[np.ndarray] = []
    training_labels: list[int] = []
    test_images: list[np.ndarray] = []
    test_labels: list[int] = []
    for class_id in range(2):
        for _ in range(4):
            image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
            image[:, :, class_id] = np.clip(
                image[:, :, class_id].astype(np.int16) + 40,
                0,
                255,
            ).astype(np.uint8)
            training_images.append(image)
            training_labels.append(class_id)
        for _ in range(2):
            test_images.append(
                rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
            )
            test_labels.append(class_id)
    return (
        np.stack(training_images),
        np.asarray(training_labels, dtype=np.int64),
        np.stack(test_images),
        np.asarray(test_labels, dtype=np.int64),
    )


def test_experiment_help_names_config_and_raw_artifact_defaults() -> None:
    result = runner.invoke(app, ["experiment", "--help"])
    output = unstyle(result.stdout)

    assert result.exit_code == 0
    assert "matched contaminated and repaired" in output
    assert "--config" in output
    assert "cifar10_experiment.yaml" in output
    assert "--output" in output
    assert "training_results.json" in output


def test_experiment_runs_generated_arrays_without_cifar_download(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    config_path = _write_small_experiment_config(tmp_path / "experiment.yaml")
    output = tmp_path / "private-results" / "training.json"
    expected_config = CifarExperimentConfig.from_yaml(config_path)
    returned: list[TrainingArtifact] = []
    calls: list[tuple[CifarExperimentConfig, Path, Path]] = []

    def fail_if_cifar_loader_runs(*_args: object, **_kwargs: object) -> Never:
        raise AssertionError("the CLI test must not load or download CIFAR-10")

    def run_generated(
        config: CifarExperimentConfig,
        *,
        project_root: str | Path | None = None,
        repo_root: str | Path | None = None,
    ) -> TrainingArtifact:
        assert project_root is not None
        assert repo_root is not None
        calls.append((config, Path(project_root), Path(repo_root)))
        artifact = run_cifar_experiment_core(
            config,
            source_arrays=_generated_cifar_arrays(),
            project_root=project_root,
            repo_root=repo_root,
        )
        returned.append(artifact)
        return artifact

    monkeypatch.setattr(training_module, "load_cifar10_arrays", fail_if_cifar_loader_runs)
    monkeypatch.setattr(cli_module, "run_cifar_experiment", run_generated)

    result = runner.invoke(
        app,
        ["experiment", "--config", str(config_path), "--output", str(output)],
    )

    assert result.exit_code == 0, result.output
    artifact = TrainingArtifact.model_validate_json(output.read_bytes())
    assert len(calls) == 1
    assert calls[0][0] == expected_config
    assert calls[0][0].data.download is True
    assert len(returned) == 1
    assert artifact.metadata == returned[0].metadata
    assert artifact.metadata.configuration_sha256 == expected_config.config_hash
    assert artifact.metadata.random_seeds == (3,)
    assert artifact.metadata.git_commit_sha is not None
    assert artifact.summary.dataset_source == "provided_arrays"
    assert artifact.summary.injected_family_count == 2
    assert artifact.summary.resolved_device == "cpu"
    assert artifact.summary.repair_summary.hard_group_invariant_satisfied is True
    assert tuple(run.condition for run in artifact.runs) == (
        TrainingCondition.CONTAMINATED,
        TrainingCondition.REPAIRED,
    )
    assert all(run.test_accuracy.total > 0 for run in artifact.runs)
    assert output.read_text(encoding="utf-8") == canonical_json(artifact) + "\n"

    success = json.loads(result.stdout)
    assert success["artifact"] == output.name
    assert success["configuration_sha256"] == artifact.metadata.configuration_sha256
    assert success["dataset_manifest_sha256"] == artifact.metadata.dataset_manifest_sha256
    assert success["dataset_source"] == "provided_arrays"
    assert success["injected_family_count"] == 2
    assert success["repair_hard_group_invariant_satisfied"] is True
    assert success["resolved_device"] == "cpu"
    assert success["run_count"] == 2
    assert success["seeds"] == [3]
    assert [row["condition"] for row in success["results"]] == [
        "contaminated",
        "repaired",
    ]
    assert [row["split_manifest_sha256"] for row in success["results"]] == [
        run.split_manifest_sha256 for run in artifact.runs
    ]
    assert all("shared_clean_holdout_accuracy" in row for row in success["results"])
    assert all("non_injected_test_accuracy" in row for row in success["results"])
    assert "inflates" not in result.stdout
    assert str(tmp_path) not in result.stdout
    assert str(tmp_path) not in output.read_text(encoding="utf-8")
    assert not tuple(output.parent.glob("*.tmp"))


def test_experiment_rejects_invalid_config_without_partial_output(tmp_path: Path) -> None:
    config_path = tmp_path / "private" / "invalid-experiment.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        f"unknown_private_path: {tmp_path.as_posix()}\n",
        encoding="utf-8",
    )
    output = tmp_path / "results" / "training.json"

    result = runner.invoke(
        app,
        ["experiment", "--config", str(config_path), "--output", str(output)],
    )

    assert result.exit_code == 2
    assert "experiment configuration failed" in result.output
    assert "validation" in result.output
    assert "Traceback" not in result.output
    assert str(tmp_path) not in result.output
    assert not output.exists()


def test_experiment_hides_runtime_paths_and_leaves_no_partial_output(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    config_path = _write_small_experiment_config(tmp_path / "experiment.yaml")
    output = tmp_path / "results" / "training.json"

    def fail_with_private_path(*_args: object, **_kwargs: object) -> Never:
        raise RuntimeError(f"failed near {tmp_path}")

    monkeypatch.setattr(cli_module, "run_cifar_experiment", fail_with_private_path)

    result = runner.invoke(
        app,
        ["experiment", "--config", str(config_path), "--output", str(output)],
    )
    rendered = unstyle(result.output)

    assert result.exit_code == 2
    assert "experiment failed" in rendered
    assert "training artifact" in rendered
    assert "published" in rendered
    assert "Traceback" not in rendered
    assert str(tmp_path) not in rendered
    assert not output.exists()


def test_scan_reports_exact_duplicates_and_invalid_images(tmp_path: Path) -> None:
    train = tmp_path / "train" / "cat"
    test = tmp_path / "test" / "cat"
    train.mkdir(parents=True)
    test.mkdir(parents=True)
    image_path = train / "source.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image_path)
    (test / "copy.png").write_bytes(image_path.read_bytes())
    (test / "broken.png").write_bytes(b"not an image")

    result = runner.invoke(app, ["scan", str(tmp_path)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["mode"] == "image_folder"
    assert payload["manifest_entries"] == 3
    assert payload["valid_images"] == 2
    assert payload["invalid_images"] == 1
    assert payload["exact_duplicate_groups"] == 1
    assert payload["exact_cross_split_groups"] == 1
    assert payload["invalid_records"][0]["path"] == "test/cat/broken.png"
    assert str(tmp_path) not in result.stdout


def test_scan_reports_manifest_errors_without_traceback(tmp_path: Path) -> None:
    result = runner.invoke(app, ["scan", str(tmp_path / "missing")])

    assert result.exit_code == 2
    assert "dataset root does not exist" in result.output
    assert "Traceback" not in result.output


def test_audit_writes_private_path_free_artifact_without_network(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    train = dataset / "train" / "cat"
    test = dataset / "test" / "dog"
    train.mkdir(parents=True)
    test.mkdir(parents=True)
    source = train / "source.png"
    Image.new("RGB", (12, 12), (40, 80, 120)).save(source)
    (test / "copy.png").write_bytes(source.read_bytes())
    (test / "broken.png").write_bytes(b"broken")
    output = tmp_path / "results" / "audit.json"

    result = runner.invoke(
        app,
        [
            "audit",
            str(dataset),
            "--config",
            str(Path(__file__).parents[1] / "configs" / "default.yaml"),
            "--output",
            str(output),
            "--no-embeddings",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "audit"
    assert payload["summary"]["valid_image_count"] == 2
    assert payload["summary"]["invalid_image_count"] == 1
    assert payload["summary"]["leakage_group_count"] == 1
    assert payload["summary"]["exact_leakage_group_count"] == 1
    assert payload["summary"]["cross_label_conflict_count"] == 1
    assert str(tmp_path) not in output.read_text(encoding="utf-8")
    assert "audit.json" in result.stdout


def create_audit_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset = tmp_path / "private-dataset"
    train = dataset / "train" / "cat"
    test = dataset / "test" / "dog"
    train.mkdir(parents=True)
    test.mkdir(parents=True)
    source = train / "source.png"
    Image.new("RGB", (10, 10), (15, 30, 45)).save(source)
    (test / "copy.png").write_bytes(source.read_bytes())
    (test / "broken.png").write_bytes(b"not an image")
    audit_path = tmp_path / "audit" / "audit.json"
    config_path = Path(__file__).parents[1] / "configs" / "default.yaml"

    result = runner.invoke(
        app,
        [
            "audit",
            str(dataset),
            "--config",
            str(config_path),
            "--output",
            str(audit_path),
            "--no-embeddings",
        ],
    )

    assert result.exit_code == 0, result.output
    return dataset, audit_path, config_path


def test_repair_writes_complete_artifact_and_manifest_with_provenance(
    tmp_path: Path,
) -> None:
    _, audit_path, config_path = create_audit_fixture(tmp_path)
    repair_path = tmp_path / "repair-output" / "repair.json"
    manifest_path = tmp_path / "repair-output" / "repaired_manifest.csv"

    result = runner.invoke(
        app,
        [
            "repair",
            str(audit_path),
            "--ratios",
            "0.8,0.1,0.1",
            "--config",
            str(config_path),
            "--output",
            str(repair_path),
        ],
    )

    assert result.exit_code == 0, result.output
    artifact = RepairArtifact.model_validate_json(repair_path.read_bytes())
    source_audit = AuditArtifact.model_validate_json(audit_path.read_bytes())
    config = load_config(config_path)
    expected_configuration_sha256 = canonical_sha256(
        {
            "base_configuration_sha256": config.config_hash,
            "repair": {
                "class_balance_weight": artifact.class_balance_weight,
                "local_improvement_iterations": artifact.local_improvement_iterations,
                "random_seed": artifact.random_seed,
                "split_size_weight": artifact.split_size_weight,
            },
            "requested_ratios": [
                {"ratio": item.ratio, "split": item.split.value}
                for item in artifact.requested_ratios
            ],
        }
    )
    manifest_bytes = manifest_path.read_bytes()
    rows = list(csv.DictReader(manifest_path.read_text(encoding="utf-8").splitlines()))
    repaired_splits = {row["split"] for row in rows}

    assert artifact.artifact_type == "repair"
    assert tuple(item.ratio for item in artifact.requested_ratios) == (0.8, 0.1, 0.1)
    assert artifact.integer_targets == (
        (Split.TRAIN, 2),
        (Split.VAL, 0),
        (Split.TEST, 0),
    )
    assert artifact.split_size_weight == config.repair.split_size_weight
    assert artifact.class_balance_weight == config.repair.class_balance_weight
    assert artifact.random_seed == config.repair.random_seed
    assert (
        artifact.local_improvement_iterations
        == config.repair.local_improvement_iterations
    )
    assert artifact.metadata.configuration_sha256 == expected_configuration_sha256
    assert (
        artifact.metadata.dataset_manifest_sha256
        == source_audit.metadata.dataset_manifest_sha256
    )
    assert artifact.metadata.random_seeds == (config.repair.random_seed,)
    assert artifact.repaired_manifest_sha256 == hashlib.sha256(manifest_bytes).hexdigest()
    assert artifact.summary.definite_leakage_groups_before == 1
    assert artifact.summary.definite_leakage_groups_after == 0
    assert artifact.summary.hard_group_invariant_satisfied is True
    assert len(rows) == 2
    assert len(repaired_splits) == 1
    assert artifact.excluded_invalid_ids

    success = json.loads(result.stdout)
    assert success["source_audit_sha256"] == canonical_sha256(source_audit)
    assert success["audit_configuration_sha256"] == source_audit.metadata.configuration_sha256
    assert success["configuration_sha256"] == expected_configuration_sha256
    assert success["repaired_manifest_sha256"] == artifact.repaired_manifest_sha256
    assert success["warnings"] == list(artifact.infeasibility_warnings)
    assert str(tmp_path) not in result.stdout
    assert str(tmp_path) not in repair_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in manifest_path.read_text(encoding="utf-8")


def test_repair_strictly_rejects_malformed_audit_without_partial_outputs(
    tmp_path: Path,
) -> None:
    audit_path = tmp_path / "private" / "malformed-audit.json"
    audit_path.parent.mkdir()
    audit_path.write_text('{"artifact_type":"audit"}', encoding="utf-8")
    repair_path = tmp_path / "results" / "repair.json"
    manifest_path = tmp_path / "results" / "repaired.csv"

    result = runner.invoke(
        app,
        [
            "repair",
            str(audit_path),
            "--ratios",
            "0.8,0.1,0.1",
            "--output",
            str(repair_path),
            "--manifest-output",
            str(manifest_path),
        ],
    )

    assert result.exit_code == 2
    assert "failed strict" in result.output
    assert "validation" in result.output
    assert "Traceback" not in result.output
    assert str(tmp_path) not in result.output
    assert not repair_path.exists()
    assert not manifest_path.exists()


@pytest.mark.parametrize(
    ("ratios", "message"),
    (
        ("0.8,0.2", "exactly three"),
        ("0.8,0.2,not-a-number", "exactly three"),
        ("nan,0.5,0.5", "finite"),
        ("-0.1,0.5,0.6", "cannot be negative"),
        ("0.8,0.1,0.2", "sum to one"),
    ),
)
def test_repair_rejects_invalid_ratio_option_before_writing(
    tmp_path: Path,
    ratios: str,
    message: str,
) -> None:
    audit_path = tmp_path / "audit.json"
    audit_path.write_text("{}", encoding="utf-8")

    result = runner.invoke(
        app,
        ["repair", str(audit_path), "--ratios", ratios],
    )

    assert result.exit_code == 2
    assert message in result.output
    assert "Traceback" not in result.output


def test_repair_refuses_to_overwrite_its_source_audit(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.json"
    audit_path.write_text("{}", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "repair",
            str(audit_path),
            "--ratios",
            "0.8,0.1,0.1",
            "--output",
            str(audit_path),
        ],
    )

    assert result.exit_code == 2
    assert "must not" in result.output
    assert "overwrite" in result.output
    assert audit_path.read_text(encoding="utf-8") == "{}"


def test_materialize_copy_round_trip_is_verified_and_privacy_safe(tmp_path: Path) -> None:
    dataset, audit_path, config_path = create_audit_fixture(tmp_path)
    repair_path = tmp_path / "repair" / "repair.json"
    manifest_path = tmp_path / "repair" / "repaired_manifest.csv"
    repair_result = runner.invoke(
        app,
        [
            "repair",
            str(audit_path),
            "--ratios",
            "0.8,0.1,0.1",
            "--config",
            str(config_path),
            "--output",
            str(repair_path),
            "--manifest-output",
            str(manifest_path),
        ],
    )
    assert repair_result.exit_code == 0, repair_result.output
    source_hashes = {
        path.relative_to(dataset).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.rglob("*.png")
    }
    output_dir = tmp_path / "materialized-private-output"

    result = runner.invoke(
        app,
        [
            "materialize",
            str(manifest_path),
            str(output_dir),
            "--dataset-root",
            str(dataset),
            "--mode",
            "copy",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    materialized_images = tuple((output_dir / "images").iterdir())
    assert payload["mode"] == "copy"
    assert payload["output"] == output_dir.name
    assert payload["record_count"] == 2
    assert payload["verified_file_count"] == 2
    assert len(materialized_images) == 2
    assert (output_dir / "manifest.csv").is_file()
    assert source_hashes == {
        path.relative_to(dataset).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.rglob("*.png")
    }
    assert str(tmp_path) not in result.stdout
    assert str(tmp_path) not in (output_dir / "manifest.csv").read_text(encoding="utf-8")


def test_materialize_passes_explicit_symlink_mode_to_helper(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    calls: list[tuple[Path, Path, Path, str]] = []

    def fake_materialize(
        manifest_path: str | Path,
        dataset_root: str | Path,
        output_dir: str | Path,
        *,
        mode: Literal["copy", "symlink"] = "copy",
    ) -> MaterializationResult:
        calls.append(
            (Path(manifest_path), Path(dataset_root), Path(output_dir), mode)
        )
        return MaterializationResult(
            mode=mode,
            record_count=1,
            verified_file_count=1,
            manifest_sha256="a" * 64,
        )

    monkeypatch.setattr(cli_module, "materialize_repaired_manifest", fake_materialize)
    manifest_path = tmp_path / "repaired.csv"
    dataset_root = tmp_path / "source"
    output_dir = tmp_path / "output"

    result = runner.invoke(
        app,
        [
            "materialize",
            str(manifest_path),
            str(output_dir),
            "--dataset-root",
            str(dataset_root),
            "--mode",
            "symlink",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [(manifest_path, dataset_root, output_dir, "symlink")]
    assert json.loads(result.stdout)["mode"] == "symlink"


def test_materialize_reports_safe_domain_error_without_traceback(tmp_path: Path) -> None:
    manifest_path = tmp_path / "private" / "missing.csv"
    dataset_root = tmp_path / "private" / "missing-dataset"
    output_dir = tmp_path / "private" / "output"

    result = runner.invoke(
        app,
        [
            "materialize",
            str(manifest_path),
            str(output_dir),
            "--dataset-root",
            str(dataset_root),
        ],
    )

    assert result.exit_code == 2
    assert "dataset root must be" in result.output
    assert "directory" in result.output
    assert "Traceback" not in result.output
    assert str(tmp_path) not in result.output
    assert not output_dir.exists()
