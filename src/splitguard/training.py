"""Controlled CIFAR-10 split-integrity experiment.

The experiment deliberately optimizes for a fair contaminated-versus-repaired
comparison, not state-of-the-art CIFAR-10 accuracy. Tests pass generated arrays
directly and therefore never download CIFAR-10.
"""

from __future__ import annotations

import hashlib
import io
import os
import random
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated, Literal, Self, cast

import numpy as np
import numpy.typing as npt
import torch
import yaml
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from splitguard.metrics import collect_run_metadata
from splitguard.models.tiny_classifier import TinyCifarClassifier
from splitguard.repair import RepairPlan, repair_splits
from splitguard.schemas import (
    AccuracyMetric,
    DuplicateFamily,
    ImageRecord,
    NamedAccuracy,
    Split,
    TrainingArtifact,
    TrainingCondition,
    TrainingConditionSummary,
    TrainingContaminationGroundTruth,
    TrainingExperimentSummary,
    TrainingRun,
    canonical_sha256,
    family_id_for,
    stable_id,
)

UInt8Images = npt.NDArray[np.uint8]
IntLabels = npt.NDArray[np.int64]
DeviceChoice = Literal["auto", "cpu", "cuda"]
ResolvedDevice = Literal["cpu", "cuda"]
_CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465), dtype=torch.float32).view(3, 1, 1)
_CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616), dtype=torch.float32).view(3, 1, 1)
_CIFAR_CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class ExperimentConfigError(ValueError):
    """Raised when an experiment configuration or source dataset is invalid."""


class _FrozenConfig(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class CifarDataConfig(_FrozenConfig):
    """Dataset selection and controlled-contamination settings."""

    data_dir: str = ".splitguard-data/cifar10"
    download: bool = True
    num_classes: Annotated[int, Field(ge=2, le=100)] = 10
    train_per_class: Annotated[int, Field(gt=0)] = 300
    validation_per_class: Annotated[int, Field(gt=0)] = 50
    test_per_class: Annotated[int, Field(gt=0)] = 100
    contamination_per_class: Annotated[int, Field(gt=0)] = 20
    sampling_seed: Annotated[int, Field(ge=0, le=2**32 - 1)] = 314159
    contamination: Literal["resize", "jpeg"] = "resize"

    @field_validator("data_dir")
    @classmethod
    def portable_data_dir(cls, value: str) -> str:
        raw = value.strip()
        windows = PureWindowsPath(raw)
        posix = PurePosixPath(raw.replace("\\", "/"))
        if (
            not raw
            or "\x00" in raw
            or windows.drive
            or windows.root
            or posix.is_absolute()
            or ".." in posix.parts
            or posix == PurePosixPath(".")
        ):
            raise ValueError("data_dir must be a portable project-relative path")
        return posix.as_posix()

    @model_validator(mode="after")
    def feasible_contamination(self) -> Self:
        if self.contamination_per_class > self.train_per_class:
            raise ValueError("contamination_per_class cannot exceed train_per_class")
        return self


class TrainingConfig(_FrozenConfig):
    """Settings held identical across contaminated and repaired conditions."""

    seeds: Annotated[tuple[int, ...], Field(min_length=1)] = (7, 42, 101)
    epochs: Annotated[int, Field(gt=0)] = 3
    batch_size: Annotated[int, Field(gt=0)] = 128
    learning_rate: Annotated[float, Field(gt=0.0)] = 0.001
    weight_decay: Annotated[float, Field(ge=0.0)] = 0.0001
    device: DeviceChoice = "auto"
    num_workers: Annotated[int, Field(ge=0)] = 0
    augmentation: Literal["none"] = "none"

    @field_validator("seeds", mode="before")
    @classmethod
    def yaml_seed_sequence(cls, value: object) -> object:
        # YAML sequences load as lists. Normalize only the container while
        # leaving strict validation of every seed to Pydantic.
        return tuple(value) if isinstance(value, list) else value

    @field_validator("seeds")
    @classmethod
    def canonical_seeds(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if any(value < 0 or value > 2**32 - 1 for value in values):
            raise ValueError("seeds must be unsigned 32-bit integers")
        if values != tuple(sorted(set(values))):
            raise ValueError("seeds must be sorted and unique")
        return values


class CifarRepairConfig(_FrozenConfig):
    """Controlled repair settings applied once before matched training runs."""

    split_size_weight: Annotated[float, Field(gt=0.0)] = 1.0
    class_balance_weight: Annotated[float, Field(ge=0.0)] = 1.0
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] = 271828
    local_iterations: Annotated[int, Field(ge=0, le=10_000)] = 250


class CifarExperimentConfig(_FrozenConfig):
    """Complete validated configuration for the CIFAR integrity experiment."""

    data: CifarDataConfig = Field(default_factory=CifarDataConfig)
    repair: CifarRepairConfig = Field(default_factory=CifarRepairConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> CifarExperimentConfig:
        config_path = Path(path)
        try:
            payload: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ExperimentConfigError(
                f"could not read experiment configuration {config_path.name!r}"
            ) from exc
        except yaml.YAMLError as exc:
            raise ExperimentConfigError("experiment configuration contains invalid YAML") from exc
        if not isinstance(payload, dict):
            raise ExperimentConfigError("experiment configuration must contain a mapping")
        try:
            return cls.model_validate(payload)
        except ValueError as exc:
            raise ExperimentConfigError("experiment configuration failed validation") from exc

    @property
    def config_hash(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))


CifarContaminantGroundTruth = TrainingContaminationGroundTruth


@dataclass(frozen=True, slots=True)
class ControlledCifarSplits:
    """Array-backed contaminated and repaired splits with independent provenance."""

    train_images: UInt8Images
    train_labels: IntLabels
    validation_images: UInt8Images
    validation_labels: IntLabels
    clean_test_images: UInt8Images
    clean_test_labels: IntLabels
    contaminant_images: UInt8Images
    contaminant_labels: IntLabels
    contaminant_source_train_positions: tuple[int, ...]
    contamination_ground_truth: tuple[CifarContaminantGroundTruth, ...]
    derived_record_ids: tuple[str, ...]
    ground_truth_sha256: str
    repaired_train_images: UInt8Images
    repaired_train_labels: IntLabels
    repaired_test_images: UInt8Images
    repaired_test_labels: IntLabels
    repaired_pool_test_record_ids: tuple[str, ...]
    repair_plan: RepairPlan
    repair_plan_sha256: str
    contaminated_manifest_sha256: str
    repaired_manifest_sha256: str
    shared_clean_holdout_sha256: str
    dataset_sha256: str

    @property
    def contaminated_test_images(self) -> UInt8Images:
        return np.concatenate((self.clean_test_images, self.contaminant_images), axis=0)

    @property
    def contaminated_test_labels(self) -> IntLabels:
        return np.concatenate((self.clean_test_labels, self.contaminant_labels), axis=0)

    def injected_derivative_test_data(
        self,
        condition: TrainingCondition,
    ) -> tuple[UInt8Images, IntLabels]:
        """Return only injected derivatives that are actually in this test split."""

        if condition is TrainingCondition.CONTAMINATED:
            return self.contaminant_images, self.contaminant_labels
        derived_ids = set(self.derived_record_ids)
        clean_count = len(self.clean_test_images)
        positions = tuple(
            clean_count + offset
            for offset, record_id in enumerate(self.repaired_pool_test_record_ids)
            if record_id in derived_ids
        )
        return _take_arrays(self.repaired_test_images, self.repaired_test_labels, positions)

    def non_injected_test_data(
        self,
        condition: TrainingCondition,
    ) -> tuple[UInt8Images, IntLabels]:
        """Return all actual test records other than the injected derivatives."""

        if condition is TrainingCondition.CONTAMINATED:
            return self.clean_test_images, self.clean_test_labels
        derived_ids = set(self.derived_record_ids)
        clean_count = len(self.clean_test_images)
        positions = tuple(range(clean_count)) + tuple(
            clean_count + offset
            for offset, record_id in enumerate(self.repaired_pool_test_record_ids)
            if record_id not in derived_ids
        )
        return _take_arrays(self.repaired_test_images, self.repaired_test_labels, positions)


class _ArrayDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, images: UInt8Images, labels: IntLabels) -> None:
        self._images = images
        self._labels = labels

    def __len__(self) -> int:
        return int(self._labels.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image = torch.from_numpy(self._images[index].copy()).permute(2, 0, 1).float() / 255.0
        image = (image - _CIFAR_MEAN) / _CIFAR_STD
        return image, torch.tensor(int(self._labels[index]), dtype=torch.long)


def _readonly(array: npt.NDArray[np.generic]) -> npt.NDArray[np.generic]:
    result = np.ascontiguousarray(array).copy()
    result.setflags(write=False)
    return result


def _take_arrays(
    images: UInt8Images,
    labels: IntLabels,
    positions: Sequence[int],
) -> tuple[UInt8Images, IntLabels]:
    index = np.asarray(tuple(positions), dtype=np.int64)
    return (
        images[index],
        labels[index],
    )


def _validate_source_arrays(
    train_images: UInt8Images,
    train_labels: IntLabels,
    test_images: UInt8Images,
    test_labels: IntLabels,
    num_classes: int,
) -> None:
    for name, images, labels in (
        ("training", train_images, train_labels),
        ("test", test_images, test_labels),
    ):
        if images.dtype != np.uint8 or images.ndim != 4 or images.shape[1:] != (32, 32, 3):
            raise ExperimentConfigError(f"{name} images must be uint8 NHWC 32x32 RGB arrays")
        if labels.dtype != np.int64 or labels.ndim != 1 or len(images) != len(labels):
            raise ExperimentConfigError(f"{name} labels must be aligned int64 vectors")
        if labels.size and (int(labels.min()) < 0 or int(labels.max()) >= num_classes):
            raise ExperimentConfigError(f"{name} labels fall outside configured classes")


def _stratified_indices(
    labels: IntLabels,
    counts: Sequence[int],
    *,
    num_classes: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    requested = sum(counts)
    buckets: list[list[int]] = [[] for _ in counts]
    for class_id in range(num_classes):
        available = np.flatnonzero(labels == class_id)
        if available.size < requested:
            raise ExperimentConfigError(
                f"class {class_id} has {available.size} items but {requested} are required"
            )
        rng = np.random.default_rng(seed + 104729 * class_id)
        chosen = rng.permutation(available)[:requested]
        offset = 0
        for bucket, count in zip(buckets, counts, strict=True):
            bucket.extend(int(index) for index in chosen[offset : offset + count])
            offset += count
    return tuple(tuple(bucket) for bucket in buckets)


def _derive_contaminant(image: UInt8Images, kind: Literal["resize", "jpeg"]) -> UInt8Images:
    source = Image.fromarray(image, mode="RGB")
    if kind == "resize":
        derived = source.resize((27, 27), Image.Resampling.BILINEAR).resize(
            (32, 32), Image.Resampling.BICUBIC
        )
    else:
        buffer = io.BytesIO()
        source.save(buffer, format="JPEG", quality=62, optimize=False, progressive=False)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            derived = decoded.convert("RGB")
    return np.asarray(derived, dtype=np.uint8)


def _array_digest(*arrays: npt.NDArray[np.generic]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        canonical = np.ascontiguousarray(array)
        shape = ",".join(str(value) for value in canonical.shape).encode("ascii")
        dtype = canonical.dtype.str.encode("ascii")
        digest.update(len(shape).to_bytes(4, "big"))
        digest.update(shape)
        digest.update(len(dtype).to_bytes(2, "big"))
        digest.update(dtype)
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def _validated_unique_pixel_digests(
    cohorts: Sequence[tuple[str, UInt8Images]],
) -> frozenset[str]:
    """Return clean digests after requiring a collision-free native baseline."""

    digests: set[str] = set()
    for _cohort_name, images in cohorts:
        for image in images:
            digest = _array_digest(image)
            if digest in digests:
                raise ExperimentConfigError(
                    "selected clean CIFAR subset contains native duplicate pixels"
                )
            digests.add(digest)
    return frozenset(digests)


def _virtual_image_record(
    image: UInt8Images,
    label: int,
    *,
    record_id: str,
    split: Split,
) -> ImageRecord:
    """Adapt one in-memory CIFAR image to the immutable repair contract."""

    return ImageRecord(
        id=record_id,
        path=f"virtual/{split.value}/{record_id}.rgb",
        split=split,
        label=str(label),
        byte_sha256=_array_digest(image),
        byte_size=int(image.nbytes),
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        format="rgbarray",
    )


def _assigned_pool_arrays(
    plan: RepairPlan,
    images_by_id: Mapping[str, UInt8Images],
    labels_by_id: Mapping[str, int],
    split: Split,
) -> tuple[UInt8Images, IntLabels, tuple[str, ...]]:
    record_ids = tuple(
        assignment.record_id
        for assignment in plan.assignments
        if assignment.repaired_split is split
    )
    if not record_ids:
        return (
            np.empty((0, 32, 32, 3), dtype=np.uint8),
            np.empty((0,), dtype=np.int64),
            (),
        )
    return (
        np.stack([images_by_id[record_id] for record_id in record_ids]).astype(
            np.uint8,
            copy=False,
        ),
        np.asarray([labels_by_id[record_id] for record_id in record_ids], dtype=np.int64),
        record_ids,
    )


def build_controlled_cifar_splits(
    train_images: UInt8Images,
    train_labels: IntLabels,
    test_images: UInt8Images,
    test_labels: IntLabels,
    config: CifarDataConfig,
    *,
    repair_config: CifarRepairConfig | None = None,
) -> ControlledCifarSplits:
    """Build contaminated and optimizer-repaired splits from independent ground truth.

    The original validation and clean test samples remain fixed holdouts. The
    repair optimizer operates on the affected train/test pool, with residual
    ratios chosen so the full repaired split has the same sizes as the
    contaminated split. Injected source/derivative pairs are definite,
    indivisible families; singleton training records may move to preserve the
    controlled test size and class balance.
    """

    _validate_source_arrays(
        train_images,
        train_labels,
        test_images,
        test_labels,
        config.num_classes,
    )
    selected_train, selected_validation = _stratified_indices(
        train_labels,
        (config.train_per_class, config.validation_per_class),
        num_classes=config.num_classes,
        seed=config.sampling_seed,
    )
    (selected_test,) = _stratified_indices(
        test_labels,
        (config.test_per_class,),
        num_classes=config.num_classes,
        seed=config.sampling_seed + 1,
    )

    train_index = np.asarray(selected_train, dtype=np.int64)
    validation_index = np.asarray(selected_validation, dtype=np.int64)
    test_index = np.asarray(selected_test, dtype=np.int64)
    chosen_train_images = train_images[train_index]
    chosen_train_labels = train_labels[train_index]
    chosen_validation_images = train_images[validation_index]
    chosen_validation_labels = train_labels[validation_index]
    chosen_test_images = test_images[test_index]
    chosen_test_labels = test_labels[test_index]
    clean_pixel_digests = _validated_unique_pixel_digests(
        (
            ("train", chosen_train_images),
            ("validation", chosen_validation_images),
            ("test", chosen_test_images),
        )
    )

    source_positions: list[int] = []
    for class_id in range(config.num_classes):
        positions = np.flatnonzero(chosen_train_labels == class_id)
        source_positions.extend(
            int(position) for position in positions[: config.contamination_per_class]
        )
    contaminant_images = np.stack(
        [
            _derive_contaminant(chosen_train_images[position], config.contamination)
            for position in source_positions
        ]
    ).astype(np.uint8, copy=False)
    contaminant_labels = chosen_train_labels[np.asarray(source_positions, dtype=np.int64)]
    if any(
        np.array_equal(chosen_train_images[position], contaminant_images[offset])
        for offset, position in enumerate(source_positions)
    ):
        raise ExperimentConfigError("configured contamination produced an unchanged source image")
    derived_pixel_digests: set[str] = set()
    for image in contaminant_images:
        digest = _array_digest(image)
        if digest in clean_pixel_digests or digest in derived_pixel_digests:
            raise ExperimentConfigError(
                "an injected derivative collides with a selected clean image or another derivative"
            )
        derived_pixel_digests.add(digest)

    pool_records: list[ImageRecord] = []
    images_by_id: dict[str, UInt8Images] = {}
    labels_by_id: dict[str, int] = {}
    source_record_ids: list[str] = []
    for position, original_index in enumerate(train_index):
        image = chosen_train_images[position]
        label = int(chosen_train_labels[position])
        record_id = stable_id(
            "cifar",
            "source_train",
            str(int(original_index)),
            _array_digest(image),
        )
        source_record_ids.append(record_id)
        pool_records.append(
            _virtual_image_record(
                image,
                label,
                record_id=record_id,
                split=Split.TRAIN,
            )
        )
        images_by_id[record_id] = image
        labels_by_id[record_id] = label

    derived_record_ids: list[str] = []
    definite_families: list[DuplicateFamily] = []
    for offset, position in enumerate(source_positions):
        image = contaminant_images[offset]
        label = int(contaminant_labels[offset])
        source_record_id = source_record_ids[position]
        derived_record_id = stable_id(
            "cifar",
            "derived",
            config.contamination,
            source_record_id,
            _array_digest(image),
        )
        derived_record_ids.append(derived_record_id)
        pool_records.append(
            _virtual_image_record(
                image,
                label,
                record_id=derived_record_id,
                split=Split.TEST,
            )
        )
        images_by_id[derived_record_id] = image
        labels_by_id[derived_record_id] = label
        members = tuple(sorted((source_record_id, derived_record_id)))
        definite_families.append(
            DuplicateFamily(
                family_id=family_id_for(members),
                member_ids=members,
                edge_count=1,
            )
        )

    ground_truth = tuple(
        CifarContaminantGroundTruth(
            source_record_id=source_record_ids[position],
            derived_record_id=derived_record_ids[offset],
            source_train_position=position,
            contaminated_test_position=len(chosen_test_images) + offset,
            label=int(contaminant_labels[offset]),
            corruption=config.contamination,
            source_sha256=_array_digest(chosen_train_images[position]),
            derived_sha256=_array_digest(contaminant_images[offset]),
        )
        for offset, position in enumerate(source_positions)
    )
    ground_truth_sha = canonical_sha256(
        tuple(item.model_dump(mode="json") for item in ground_truth)
    )

    accepted_repair = repair_config or CifarRepairConfig()
    pool_size = len(pool_records)
    repair_plan = repair_splits(
        pool_records,
        definite_families,
        target_ratios={
            Split.TRAIN: len(chosen_train_images) / pool_size,
            Split.VAL: 0.0,
            Split.TEST: len(contaminant_images) / pool_size,
        },
        split_size_weight=accepted_repair.split_size_weight,
        class_balance_weight=accepted_repair.class_balance_weight,
        seed=accepted_repair.seed,
        local_iterations=accepted_repair.local_iterations,
    )
    actual_pool_counts = {
        split: sum(assignment.repaired_split is split for assignment in repair_plan.assignments)
        for split in (Split.TRAIN, Split.VAL, Split.TEST)
    }
    expected_pool_counts = {
        Split.TRAIN: len(chosen_train_images),
        Split.VAL: 0,
        Split.TEST: len(contaminant_images),
    }
    if (
        not repair_plan.hard_group_invariant_satisfied
        or repair_plan.definite_leakage_groups_after != 0
    ):
        raise ExperimentConfigError("SplitGuard repair failed its duplicate-family invariant")
    if actual_pool_counts != expected_pool_counts:
        raise ExperimentConfigError(
            "SplitGuard repair could not preserve controlled train/test sizes"
        )

    repaired_train_images, repaired_train_labels, _repaired_train_record_ids = (
        _assigned_pool_arrays(
            repair_plan,
            images_by_id,
            labels_by_id,
            Split.TRAIN,
        )
    )
    (
        repaired_pool_test_images,
        repaired_pool_test_labels,
        repaired_pool_test_record_ids,
    ) = _assigned_pool_arrays(
        repair_plan,
        images_by_id,
        labels_by_id,
        Split.TEST,
    )
    repaired_test_images = np.concatenate(
        (chosen_test_images, repaired_pool_test_images),
        axis=0,
    )
    repaired_test_labels = np.concatenate(
        (chosen_test_labels, repaired_pool_test_labels),
        axis=0,
    )
    repair_plan_sha = canonical_sha256(repair_plan)

    dataset_sha = _array_digest(train_images, train_labels, test_images, test_labels)
    contaminated_hash = canonical_sha256(
        {
            "condition": TrainingCondition.CONTAMINATED.value,
            "train": _array_digest(chosen_train_images, chosen_train_labels),
            "validation": _array_digest(chosen_validation_images, chosen_validation_labels),
            "test": _array_digest(
                chosen_test_images,
                chosen_test_labels,
                contaminant_images,
                contaminant_labels,
            ),
            "ground_truth_sha256": ground_truth_sha,
            "family_assignments": tuple(
                {
                    "source_record_id": item.source_record_id,
                    "derived_record_id": item.derived_record_id,
                    "source_split": item.source_split,
                    "derived_split": item.contaminated_split,
                }
                for item in ground_truth
            ),
        }
    )
    repaired_hash = canonical_sha256(
        {
            "condition": TrainingCondition.REPAIRED.value,
            "train": _array_digest(repaired_train_images, repaired_train_labels),
            "validation": _array_digest(chosen_validation_images, chosen_validation_labels),
            "test": _array_digest(repaired_test_images, repaired_test_labels),
            "ground_truth_sha256": ground_truth_sha,
            "repair_plan_sha256": repair_plan_sha,
        }
    )
    return ControlledCifarSplits(
        train_images=cast(UInt8Images, _readonly(chosen_train_images)),
        train_labels=cast(IntLabels, _readonly(chosen_train_labels)),
        validation_images=cast(UInt8Images, _readonly(chosen_validation_images)),
        validation_labels=cast(IntLabels, _readonly(chosen_validation_labels)),
        clean_test_images=cast(UInt8Images, _readonly(chosen_test_images)),
        clean_test_labels=cast(IntLabels, _readonly(chosen_test_labels)),
        contaminant_images=cast(UInt8Images, _readonly(contaminant_images)),
        contaminant_labels=cast(IntLabels, _readonly(contaminant_labels)),
        contaminant_source_train_positions=tuple(source_positions),
        contamination_ground_truth=ground_truth,
        derived_record_ids=tuple(derived_record_ids),
        ground_truth_sha256=ground_truth_sha,
        repaired_train_images=cast(UInt8Images, _readonly(repaired_train_images)),
        repaired_train_labels=cast(IntLabels, _readonly(repaired_train_labels)),
        repaired_test_images=cast(UInt8Images, _readonly(repaired_test_images)),
        repaired_test_labels=cast(IntLabels, _readonly(repaired_test_labels)),
        repaired_pool_test_record_ids=repaired_pool_test_record_ids,
        repair_plan=repair_plan,
        repair_plan_sha256=repair_plan_sha,
        contaminated_manifest_sha256=contaminated_hash,
        repaired_manifest_sha256=repaired_hash,
        shared_clean_holdout_sha256=_array_digest(
            chosen_test_images,
            chosen_test_labels,
        ),
        dataset_sha256=dataset_sha,
    )


def load_cifar10_arrays(
    config: CifarDataConfig,
    *,
    project_root: str | Path | None = None,
) -> tuple[UInt8Images, IntLabels, UInt8Images, IntLabels]:
    """Load CIFAR-10 only for an explicitly invoked real experiment."""

    from torchvision.datasets import CIFAR10

    project = (Path.cwd() if project_root is None else Path(project_root)).resolve()
    root = (project / Path(config.data_dir)).resolve()
    if not root.is_relative_to(project):
        raise ExperimentConfigError("CIFAR-10 data directory escapes the project root")
    try:
        train = CIFAR10(root=str(root), train=True, download=config.download)
        test = CIFAR10(root=str(root), train=False, download=config.download)
    except (OSError, RuntimeError) as exc:
        raise ExperimentConfigError("CIFAR-10 could not be loaded for the experiment") from exc
    return (
        np.asarray(train.data, dtype=np.uint8),
        np.asarray(train.targets, dtype=np.int64),
        np.asarray(test.data, dtype=np.uint8),
        np.asarray(test.targets, dtype=np.int64),
    )


def _resolve_device(choice: DeviceChoice) -> torch.device:
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise ExperimentConfigError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if choice == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    # Required by deterministic CUDA matrix multiplication on supported builds.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _loader(
    images: UInt8Images,
    labels: IntLabels,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[tuple[Tensor, Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        _ArrayDataset(images, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        generator=generator,
        persistent_workers=workers > 0,
    )


def _batches(loader: DataLoader[tuple[Tensor, Tensor]]) -> Iterator[tuple[Tensor, Tensor]]:
    yield from loader


def _evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
) -> AccuracyMetric:
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in _batches(loader):
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    return AccuracyMetric(correct=correct, total=total, accuracy=correct / total if total else 0.0)


def _per_class_accuracy(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
    class_names: Sequence[str],
) -> tuple[NamedAccuracy, ...]:
    correct = [0] * len(class_names)
    totals = [0] * len(class_names)
    model.eval()
    with torch.inference_mode():
        for images, labels in _batches(loader):
            labels_on_device = labels.to(device)
            predictions = model(images.to(device)).argmax(dim=1)
            for class_id in range(len(class_names)):
                mask = labels_on_device == class_id
                totals[class_id] += int(mask.sum().item())
                correct[class_id] += int(((predictions == labels_on_device) & mask).sum().item())
    rows = [
        NamedAccuracy(
            name=name,
            metric=AccuracyMetric(
                correct=correct[index],
                total=totals[index],
                accuracy=correct[index] / totals[index] if totals[index] else 0.0,
            ),
        )
        for index, name in enumerate(class_names)
    ]
    return tuple(sorted(rows, key=lambda row: row.name))


def _train_condition(
    splits: ControlledCifarSplits,
    condition: TrainingCondition,
    settings: TrainingConfig,
    *,
    seed: int,
    num_classes: int,
    device: torch.device,
) -> TrainingRun:
    _seed_everything(seed)
    if condition is TrainingCondition.CONTAMINATED:
        train_images, train_labels = splits.train_images, splits.train_labels
        test_images, test_labels = (
            splits.contaminated_test_images,
            splits.contaminated_test_labels,
        )
        manifest_hash = splits.contaminated_manifest_sha256
    else:
        train_images, train_labels = splits.repaired_train_images, splits.repaired_train_labels
        test_images, test_labels = splits.repaired_test_images, splits.repaired_test_labels
        manifest_hash = splits.repaired_manifest_sha256

    injected_images, injected_labels = splits.injected_derivative_test_data(condition)
    non_injected_images, non_injected_labels = splits.non_injected_test_data(condition)

    train_loader = _loader(
        train_images,
        train_labels,
        batch_size=settings.batch_size,
        workers=settings.num_workers,
        shuffle=True,
        seed=seed,
    )
    train_eval_loader = _loader(
        train_images,
        train_labels,
        batch_size=settings.batch_size,
        workers=settings.num_workers,
        shuffle=False,
        seed=seed,
    )
    validation_loader = _loader(
        splits.validation_images,
        splits.validation_labels,
        batch_size=settings.batch_size,
        workers=settings.num_workers,
        shuffle=False,
        seed=seed,
    )
    test_loader = _loader(
        test_images,
        test_labels,
        batch_size=settings.batch_size,
        workers=settings.num_workers,
        shuffle=False,
        seed=seed,
    )
    clean_loader = _loader(
        splits.clean_test_images,
        splits.clean_test_labels,
        batch_size=settings.batch_size,
        workers=settings.num_workers,
        shuffle=False,
        seed=seed,
    )
    non_injected_loader = _loader(
        non_injected_images,
        non_injected_labels,
        batch_size=settings.batch_size,
        workers=settings.num_workers,
        shuffle=False,
        seed=seed,
    )
    injected_loader = (
        _loader(
            injected_images,
            injected_labels,
            batch_size=settings.batch_size,
            workers=settings.num_workers,
            shuffle=False,
            seed=seed,
        )
        if len(injected_labels)
        else None
    )

    model = TinyCifarClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=settings.epochs,
    )
    started = time.perf_counter()
    for _ in range(settings.epochs):
        model.train()
        for images, labels in _batches(train_loader):
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()
    duration = time.perf_counter() - started

    class_names = (
        _CIFAR_CLASS_NAMES
        if num_classes == len(_CIFAR_CLASS_NAMES)
        else tuple(f"class_{index}" for index in range(num_classes))
    )
    return TrainingRun(
        seed=seed,
        condition=condition,
        resolved_device=cast(ResolvedDevice, device.type),
        split_manifest_sha256=manifest_hash,
        train_accuracy=_evaluate(model, train_eval_loader, device),
        validation_accuracy=_evaluate(model, validation_loader, device),
        test_accuracy=_evaluate(model, test_loader, device),
        per_class_test_accuracy=_per_class_accuracy(model, test_loader, device, class_names),
        contaminated_example_accuracy=(
            _evaluate(model, injected_loader, device) if injected_loader is not None else None
        ),
        shared_clean_holdout_accuracy=_evaluate(model, clean_loader, device),
        non_injected_test_accuracy=_evaluate(model, non_injected_loader, device),
        duration_seconds=duration,
    )


def run_cifar_experiment(
    config: CifarExperimentConfig,
    *,
    source_arrays: tuple[UInt8Images, IntLabels, UInt8Images, IntLabels] | None = None,
    project_root: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> TrainingArtifact:
    """Run matched contaminated/repaired training conditions and return raw metrics."""

    dataset_source: Literal["torchvision", "provided_arrays"] = (
        "torchvision" if source_arrays is None else "provided_arrays"
    )
    arrays = (
        load_cifar10_arrays(config.data, project_root=project_root)
        if source_arrays is None
        else source_arrays
    )
    splits = build_controlled_cifar_splits(
        *arrays,
        config.data,
        repair_config=config.repair,
    )
    device = _resolve_device(config.training.device)
    runs = tuple(
        _train_condition(
            splits,
            condition,
            config.training,
            seed=seed,
            num_classes=config.data.num_classes,
            device=device,
        )
        for condition in (TrainingCondition.CONTAMINATED, TrainingCondition.REPAIRED)
        for seed in config.training.seeds
    )
    metadata = collect_run_metadata(
        config.config_hash,
        splits.dataset_sha256,
        config.training.seeds,
        repo_root=repo_root,
    )
    _, repaired_injected_labels = splits.injected_derivative_test_data(TrainingCondition.REPAIRED)
    _, repaired_non_injected_labels = splits.non_injected_test_data(TrainingCondition.REPAIRED)
    summary = TrainingExperimentSummary(
        dataset_source=dataset_source,
        num_classes=config.data.num_classes,
        corruption=config.data.contamination,
        injected_family_count=len(splits.contamination_ground_truth),
        sampling_seed=config.data.sampling_seed,
        repair_seed=config.repair.seed,
        training_seeds=config.training.seeds,
        requested_device=config.training.device,
        resolved_device=cast(ResolvedDevice, device.type),
        ground_truth=splits.contamination_ground_truth,
        ground_truth_sha256=splits.ground_truth_sha256,
        repair_plan_sha256=splits.repair_plan_sha256,
        shared_clean_holdout_sha256=splits.shared_clean_holdout_sha256,
        shared_clean_holdout_count=len(splits.clean_test_labels),
        condition_summaries=(
            TrainingConditionSummary(
                condition=TrainingCondition.CONTAMINATED,
                split_manifest_sha256=splits.contaminated_manifest_sha256,
                train_image_count=len(splits.train_labels),
                validation_image_count=len(splits.validation_labels),
                test_image_count=len(splits.contaminated_test_labels),
                injected_derivative_test_count=len(splits.contaminant_labels),
                non_injected_test_count=len(splits.clean_test_labels),
            ),
            TrainingConditionSummary(
                condition=TrainingCondition.REPAIRED,
                split_manifest_sha256=splits.repaired_manifest_sha256,
                train_image_count=len(splits.repaired_train_labels),
                validation_image_count=len(splits.validation_labels),
                test_image_count=len(splits.repaired_test_labels),
                injected_derivative_test_count=len(repaired_injected_labels),
                non_injected_test_count=len(repaired_non_injected_labels),
            ),
        ),
        repair_summary=splits.repair_plan.summary,
        repair_split_size_weight=config.repair.split_size_weight,
        repair_class_balance_weight=config.repair.class_balance_weight,
        repair_local_iterations=config.repair.local_iterations,
        repair_warnings=splits.repair_plan.infeasibility_warnings,
    )
    return TrainingArtifact(metadata=metadata, summary=summary, runs=runs)


__all__ = [
    "CifarContaminantGroundTruth",
    "CifarDataConfig",
    "CifarExperimentConfig",
    "CifarRepairConfig",
    "ControlledCifarSplits",
    "ExperimentConfigError",
    "TrainingConfig",
    "build_controlled_cifar_splits",
    "load_cifar10_arrays",
    "run_cifar_experiment",
]
