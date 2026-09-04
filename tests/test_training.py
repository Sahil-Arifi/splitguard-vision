"""Offline tests for the controlled CIFAR-10 integrity experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Never

import numpy as np
import pytest
import torch
from pydantic import ValidationError

import splitguard.training as training_module
from splitguard.models.tiny_classifier import TinyCifarClassifier
from splitguard.schemas import (
    Split,
    TrainingArtifact,
    TrainingCondition,
    TrainingExperimentSummary,
    TrainingRun,
)
from splitguard.training import (
    CifarDataConfig,
    CifarExperimentConfig,
    CifarRepairConfig,
    ExperimentConfigError,
    TrainingConfig,
    build_controlled_cifar_splits,
    run_cifar_experiment,
)


def source_arrays(
    *,
    classes: int = 2,
    training_per_class: int = 8,
    test_per_class: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260903)
    training_images: list[np.ndarray] = []
    training_labels: list[int] = []
    test_images: list[np.ndarray] = []
    test_labels: list[int] = []
    for class_id in range(classes):
        for _ in range(training_per_class):
            image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
            image[:, :, class_id % 3] = np.clip(
                image[:, :, class_id % 3].astype(np.int16) + 40,
                0,
                255,
            ).astype(np.uint8)
            training_images.append(image)
            training_labels.append(class_id)
        for _ in range(test_per_class):
            test_images.append(rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8))
            test_labels.append(class_id)
    return (
        np.stack(training_images),
        np.asarray(training_labels, dtype=np.int64),
        np.stack(test_images),
        np.asarray(test_labels, dtype=np.int64),
    )


def small_config(*, epochs: int = 1) -> CifarExperimentConfig:
    return CifarExperimentConfig(
        data=CifarDataConfig(
            data_dir="private/cifar",
            download=False,
            num_classes=2,
            train_per_class=4,
            validation_per_class=2,
            test_per_class=3,
            contamination_per_class=1,
            sampling_seed=19,
            contamination="resize",
        ),
        repair=CifarRepairConfig(
            split_size_weight=1.0,
            class_balance_weight=1.0,
            seed=23,
            local_iterations=25,
        ),
        training=TrainingConfig(
            seeds=(3,),
            epochs=epochs,
            batch_size=4,
            learning_rate=0.001,
            weight_decay=0.0,
            device="cpu",
            num_workers=0,
            augmentation="none",
        ),
    )


def test_tiny_classifier_validates_shape_contract() -> None:
    model = TinyCifarClassifier(num_classes=3)

    assert model(torch.zeros((2, 3, 32, 32))).shape == (2, 3)
    with pytest.raises(ValueError, match="NCHW shape"):
        model(torch.zeros((2, 32, 32, 3)))
    with pytest.raises(TypeError, match="floating-point"):
        model(torch.zeros((2, 3, 32, 32), dtype=torch.uint8))
    with pytest.raises(ValueError, match="at least two"):
        TinyCifarClassifier(num_classes=1)
    with pytest.raises(TypeError, match="integer"):
        TinyCifarClassifier(num_classes=True)


def test_experiment_config_is_portable_strict_and_hashable(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """
data:
  data_dir: private/cifar
  download: false
  num_classes: 2
  train_per_class: 4
  validation_per_class: 2
  test_per_class: 3
  contamination_per_class: 1
  sampling_seed: 19
  contamination: jpeg
training:
  seeds: [3]
  epochs: 1
  batch_size: 4
  learning_rate: 0.001
  weight_decay: 0.0
  device: cpu
  num_workers: 0
  augmentation: none
""".lstrip(),
        encoding="utf-8",
    )

    loaded = CifarExperimentConfig.from_yaml(config_path)

    assert loaded.data.contamination == "jpeg"
    assert len(loaded.config_hash) == 64
    assert loaded.config_hash == CifarExperimentConfig.from_yaml(config_path).config_hash
    with pytest.raises(ValueError, match="project-relative"):
        CifarDataConfig(data_dir="C:/private/cifar")
    with pytest.raises(ValueError, match="cannot exceed"):
        CifarDataConfig(train_per_class=1, contamination_per_class=2)
    with pytest.raises(ValueError, match="valid boolean"):
        CifarDataConfig(download=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="valid integer"):
        TrainingConfig(epochs="1")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sorted and unique"):
        TrainingConfig(seeds=(42, 7))
    with pytest.raises(ValueError, match="unsigned 32-bit"):
        TrainingConfig(seeds=(2**32,))
    with pytest.raises(ValueError, match="greater than 0"):
        CifarRepairConfig(split_size_weight=0.0)


def test_checked_in_config_is_explicit_and_cpu_practical() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "cifar10_experiment.yaml"

    config = CifarExperimentConfig.from_yaml(config_path)

    assert config.data.download is True
    assert config.training.seeds == (7, 42, 101)
    assert config.training.num_workers == 0
    assert config.training.augmentation == "none"
    assert config.repair.seed == 271828
    assert config.repair.local_iterations == 250
    assert config.data.train_per_class * config.data.num_classes == 3_000
    assert config.data.validation_per_class * config.data.num_classes == 500
    assert config.data.test_per_class * config.data.num_classes == 1_000
    assert config.data.contamination_per_class * config.data.num_classes == 200


def test_invalid_experiment_documents_are_sanitized(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("unknown: true\n", encoding="utf-8")

    with pytest.raises(ExperimentConfigError, match="failed validation") as error:
        CifarExperimentConfig.from_yaml(invalid)

    assert str(tmp_path) not in str(error.value)
    with pytest.raises(ExperimentConfigError, match="could not read"):
        CifarExperimentConfig.from_yaml(tmp_path / "missing.yaml")


@pytest.mark.parametrize("kind", ["resize", "jpeg"])
def test_controlled_splits_are_deterministic_and_keep_ground_truth_independent(
    kind: Literal["resize", "jpeg"],
) -> None:
    arrays = source_arrays()
    experiment_config = small_config()
    config = experiment_config.data.model_copy(update={"contamination": kind})

    first = build_controlled_cifar_splits(
        *arrays,
        config,
        repair_config=experiment_config.repair,
    )
    second = build_controlled_cifar_splits(
        *arrays,
        config,
        repair_config=experiment_config.repair,
    )

    assert first.contaminated_manifest_sha256 == second.contaminated_manifest_sha256
    assert first.repaired_manifest_sha256 == second.repaired_manifest_sha256
    assert first.ground_truth_sha256 == second.ground_truth_sha256
    assert len(first.ground_truth_sha256) == 64
    assert first.contaminated_manifest_sha256 != first.repaired_manifest_sha256
    assert first.repair_plan_sha256 == second.repair_plan_sha256
    assert first.train_images.shape[0] == 8
    assert first.validation_images.shape[0] == 4
    assert first.clean_test_images.shape[0] == 6
    assert first.contaminant_images.shape[0] == 2
    assert first.contaminated_test_images.shape[0] == 8
    assert first.repaired_train_images.shape[0] == 8
    assert first.repaired_test_images.shape[0] == 8
    assert first.repair_plan.definite_leakage_groups_before == 2
    assert first.repair_plan.definite_leakage_groups_after == 0
    assert first.repair_plan.hard_group_invariant_satisfied is True
    assert tuple(count for _, count in first.repair_plan.integer_targets) == (8, 0, 2)
    assert first.contaminant_source_train_positions == second.contaminant_source_train_positions
    assert all(
        not np.array_equal(first.train_images[position], first.contaminant_images[index])
        for index, position in enumerate(first.contaminant_source_train_positions)
    )
    assert not first.train_images.flags.writeable
    assert np.array_equal(first.contaminant_labels, np.asarray((0, 1), dtype=np.int64))
    assert len(first.contamination_ground_truth) == 2
    for offset, row in enumerate(first.contamination_ground_truth):
        assert row.source_train_position == first.contaminant_source_train_positions[offset]
        assert row.contaminated_test_position == len(first.clean_test_images) + offset
        assert row.label == int(first.contaminant_labels[offset])
        assert row.expected_relationship == "transformed_duplicate"
        assert row.repair_requires_same_split is True
        assert row.source_split == "train"
        assert row.contaminated_split == "test"
        assert row.corruption == kind
        assert row.source_sha256 != row.derived_sha256
    repaired_split_by_id = {
        assignment.record_id: assignment.repaired_split
        for assignment in first.repair_plan.assignments
    }
    assert all(
        repaired_split_by_id[row.source_record_id] == repaired_split_by_id[row.derived_record_id]
        for row in first.contamination_ground_truth
    )


def test_controlled_split_rejects_bad_shapes_and_short_classes() -> None:
    arrays = source_arrays()
    config = small_config().data
    bad_images = arrays[0][:, :, :, 0]

    with pytest.raises(ExperimentConfigError, match="32x32 RGB"):
        build_controlled_cifar_splits(bad_images, arrays[1], arrays[2], arrays[3], config)
    with pytest.raises(ExperimentConfigError, match="required"):
        build_controlled_cifar_splits(arrays[0][:4], arrays[1][:4], arrays[2], arrays[3], config)


def test_controlled_split_rejects_native_and_degenerate_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays = source_arrays()
    config = small_config().data
    duplicated_train = arrays[0].copy()
    class_zero = np.flatnonzero(arrays[1] == 0)
    duplicated_train[class_zero] = duplicated_train[class_zero[0]]

    with pytest.raises(ExperimentConfigError, match="native duplicate pixels"):
        build_controlled_cifar_splits(
            duplicated_train,
            arrays[1],
            arrays[2],
            arrays[3],
            config,
        )

    monkeypatch.setattr(
        training_module,
        "_derive_contaminant",
        lambda image, _kind: image.copy(),
    )
    with pytest.raises(ExperimentConfigError, match="unchanged source image"):
        build_controlled_cifar_splits(*arrays, config)

    selected_train, _selected_validation = training_module._stratified_indices(
        arrays[1],
        (config.train_per_class, config.validation_per_class),
        num_classes=config.num_classes,
        seed=config.sampling_seed,
    )
    other_selected_clean_image = arrays[0][selected_train[1]]
    monkeypatch.setattr(
        training_module,
        "_derive_contaminant",
        lambda _image, _kind: other_selected_clean_image.copy(),
    )
    with pytest.raises(ExperimentConfigError, match="selected clean image"):
        build_controlled_cifar_splits(*arrays, config)

    monkeypatch.setattr(
        training_module,
        "_derive_contaminant",
        lambda image, _kind: np.zeros_like(image),
    )
    with pytest.raises(ExperimentConfigError, match="another derivative"):
        build_controlled_cifar_splits(*arrays, config)


def test_offline_training_runs_matched_conditions_without_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = small_config()
    arrays = source_arrays()
    expected_splits = build_controlled_cifar_splits(
        *arrays,
        config.data,
        repair_config=config.repair,
    )

    def fail_if_loader_is_called(*_args: object, **_kwargs: object) -> Never:
        raise AssertionError("offline source arrays must bypass the CIFAR-10 loader")

    monkeypatch.setattr(training_module, "load_cifar10_arrays", fail_if_loader_is_called)

    artifact = run_cifar_experiment(
        config,
        source_arrays=arrays,
        project_root=tmp_path,
        repo_root=tmp_path,
    )

    assert tuple(run.condition for run in artifact.runs) == (
        TrainingCondition.CONTAMINATED,
        TrainingCondition.REPAIRED,
    )
    assert artifact.metadata.random_seeds == (3,)
    assert artifact.metadata.configuration_sha256 == config.config_hash
    assert artifact.metadata.dataset_manifest_sha256 == expected_splits.dataset_sha256
    assert tuple(run.seed for run in artifact.runs) == (3, 3)
    assert artifact.runs[0].split_manifest_sha256 == (expected_splits.contaminated_manifest_sha256)
    assert artifact.runs[1].split_manifest_sha256 == expected_splits.repaired_manifest_sha256
    assert artifact.runs[0].split_manifest_sha256 != artifact.runs[1].split_manifest_sha256
    assert artifact.runs[0].test_accuracy.total == 8
    assert artifact.runs[1].test_accuracy.total == 8
    assert all(run.train_accuracy.total == 8 for run in artifact.runs)
    assert all(run.resolved_device == "cpu" for run in artifact.runs)
    assert all(run.shared_clean_holdout_accuracy.total == 6 for run in artifact.runs)
    assert all(run.clean_only_test_accuracy.total == 6 for run in artifact.runs)
    assert artifact.runs[0].non_injected_test_accuracy.total == 6
    assert artifact.runs[1].non_injected_test_accuracy.total == 8
    assert artifact.runs[0].contaminated_example_accuracy is not None
    assert artifact.runs[0].contaminated_example_accuracy.total == 2
    assert artifact.runs[1].contaminated_example_accuracy is None
    assert all(len(run.per_class_test_accuracy) == 2 for run in artifact.runs)

    summary = artifact.summary
    assert summary.dataset_source == "provided_arrays"
    assert summary.injected_family_count == 2
    assert summary.ground_truth == expected_splits.contamination_ground_truth
    assert summary.ground_truth_sha256 == expected_splits.ground_truth_sha256
    assert summary.repair_plan_sha256 == expected_splits.repair_plan_sha256
    assert summary.shared_clean_holdout_sha256 == expected_splits.shared_clean_holdout_sha256
    assert summary.sampling_seed == config.data.sampling_seed
    assert summary.repair_seed == config.repair.seed
    assert summary.training_seeds == config.training.seeds
    assert summary.requested_device == "cpu"
    assert summary.resolved_device == "cpu"
    assert summary.repair_summary.definite_leakage_groups_before == 2
    assert summary.repair_summary.definite_leakage_groups_after == 0
    assert summary.repair_summary.hard_group_invariant_satisfied is True
    assert tuple(
        condition.injected_derivative_test_count for condition in summary.condition_summaries
    ) == (2, 0)
    assert tuple(
        condition.non_injected_test_count for condition in summary.condition_summaries
    ) == (6, 8)

    restored = TrainingArtifact.model_validate_json(artifact.model_dump_json())
    assert restored == artifact

    missing_pair = artifact.model_dump()
    missing_pair["runs"] = tuple(missing_pair["runs"][:-1])
    with pytest.raises(ValidationError, match="at least 2"):
        TrainingArtifact.model_validate(missing_pair)

    mismatched_pair = artifact.model_dump()
    replacement = dict(mismatched_pair["runs"][1])
    replacement["condition"] = TrainingCondition.CONTAMINATED
    replacement["seed"] = 4
    mismatched_pair["runs"] = (mismatched_pair["runs"][0], replacement)
    with pytest.raises(ValidationError, match="exactly one contaminated and repaired"):
        TrainingArtifact.model_validate(mismatched_pair)

    wrong_ground_truth_hash = summary.model_dump()
    wrong_ground_truth_hash["ground_truth_sha256"] = "f" * 64
    with pytest.raises(ValidationError, match="ground_truth_sha256"):
        TrainingExperimentSummary.model_validate(wrong_ground_truth_hash)

    inconsistent_classes = artifact.runs[0].model_dump()
    per_class_rows = list(inconsistent_classes["per_class_test_accuracy"])
    per_class_rows.append(
        {
            "name": "zz_extra",
            "metric": {"correct": 0, "total": 1, "accuracy": 0.0},
        }
    )
    inconsistent_classes["per_class_test_accuracy"] = tuple(per_class_rows)
    with pytest.raises(ValidationError, match="per-class metrics must sum"):
        TrainingRun.model_validate(inconsistent_classes)


def test_repaired_injected_accuracy_uses_only_derivatives_still_in_test(
    tmp_path: Path,
) -> None:
    base = small_config()
    paired_only_data = base.data.model_copy(
        update={"train_per_class": 1, "validation_per_class": 1}
    )
    config = base.model_copy(update={"data": paired_only_data})
    arrays = source_arrays()
    expected_splits = build_controlled_cifar_splits(
        *arrays,
        config.data,
        repair_config=config.repair,
    )
    repaired_split_by_id = {
        assignment.record_id: assignment.repaired_split
        for assignment in expected_splits.repair_plan.assignments
    }
    expected_offsets = tuple(
        offset
        for offset, row in enumerate(expected_splits.contamination_ground_truth)
        if repaired_split_by_id[row.derived_record_id] is Split.TEST
    )
    injected_images, injected_labels = expected_splits.injected_derivative_test_data(
        TrainingCondition.REPAIRED
    )
    assert len(expected_offsets) == 1
    assert np.array_equal(
        injected_images,
        expected_splits.contaminant_images[np.asarray(expected_offsets, dtype=np.int64)],
    )
    assert np.array_equal(
        injected_labels,
        expected_splits.contaminant_labels[np.asarray(expected_offsets, dtype=np.int64)],
    )

    artifact = run_cifar_experiment(
        config,
        source_arrays=arrays,
        project_root=tmp_path,
        repo_root=tmp_path,
    )

    contaminated, repaired = artifact.runs
    contaminated_summary, repaired_summary = artifact.summary.condition_summaries
    assert contaminated_summary.injected_derivative_test_count == 2
    assert repaired_summary.injected_derivative_test_count == 1
    assert contaminated.contaminated_example_accuracy is not None
    assert contaminated.contaminated_example_accuracy.total == 2
    assert repaired.contaminated_example_accuracy is not None
    assert repaired.contaminated_example_accuracy.total == 1
    assert repaired.non_injected_test_accuracy.total == 7
    assert repaired.test_accuracy.total == 8
