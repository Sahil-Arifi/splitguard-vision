"""External labeled-pair evaluation with a calibration-only threshold choice.

Labels and source-family assignments must come from independent human review.
The loader checks declared family, path, byte, and decoded-pixel separation;
it cannot discover undeclared relationships between transformed images.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from PIL import Image, ImageOps

from splitguard.hashing import PHASH_ALGORITHM_ID, compute_phash
from splitguard.metrics import binary_metrics, collect_run_metadata
from splitguard.schemas import (
    BinaryMetrics,
    RunMetadata,
    StrictFrozenModel,
    canonical_sha256,
)

_COLUMNS = ("left", "right", "label", "left_group", "right_group")
_MAX_FILE_BYTES = 100_000_000


class LabeledPairError(ValueError):
    """Invalid labels, images, fold independence, or calibration settings."""


@dataclass(frozen=True)
class PairObservation:
    left: str
    right: str
    label: bool
    left_group: str
    right_group: str
    exact: bool
    distance: int


@dataclass(frozen=True)
class _Fingerprint:
    byte_sha256: str
    pixel_sha256: str
    phash: int


@dataclass(frozen=True)
class _Fold:
    pairs: tuple[PairObservation, ...]
    images: dict[str, _Fingerprint]
    groups: frozenset[str]
    csv_sha256: str


class ThresholdResult(StrictFrozenModel):
    # -1 means SHA equality only; no perceptual candidate is accepted.
    radius: int
    pair_count: int
    true_negatives: int
    metrics: BinaryMetrics


class LabeledPairArtifact(StrictFrozenModel):
    artifact_type: Literal["labeled_pair_evaluation"] = "labeled_pair_evaluation"
    schema_version: Literal["1.0"] = "1.0"
    detector: str = f"sha256-or-{PHASH_ALGORITHM_ID}"
    label_source: str
    calibration_csv_sha256: str
    evaluation_csv_sha256: str
    minimum_calibration_precision: float
    selected_radius: int
    selection_rule: str = "max recall, then precision, then smallest radius; calibration only"
    calibration_sweep: tuple[ThresholdResult, ...]
    calibration: ThresholdResult
    evaluation: ThresholdResult
    metadata: RunMetadata
    limitations: tuple[str, ...] = (
        "External labels and family assignments are operator supplied; authorship is not verified.",
        "Independence checks cannot identify undeclared transformed source-family overlap.",
        "Pair metrics reflect the supplied pair distribution, not dataset-wide prevalence.",
        "Calibration precision does not guarantee held-out or deployment precision.",
        "This evaluates SHA/pHash pair decisions, not DINO retrieval or end-to-end split repair.",
    )


def _resolve_image(root: Path, value: str) -> tuple[str, Path]:
    relative = PurePosixPath(value)
    if (
        not value
        or relative.is_absolute()
        or "\\" in value
        or ":" in value
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise LabeledPairError("image paths must be normalized relative POSIX paths")
    resolved = root.joinpath(*relative.parts).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise LabeledPairError("image must be an existing file inside --dataset-root")
    return resolved.relative_to(root).as_posix(), resolved


def _fingerprint(path: Path, max_image_pixels: int) -> _Fingerprint:
    try:
        if path.stat().st_size > _MAX_FILE_BYTES:
            raise LabeledPairError("image exceeds the 100 MB encoded-file limit")
        data = path.read_bytes()
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > max_image_pixels:
                raise LabeledPairError("image exceeds the decoded-pixel limit")
            corrected = ImageOps.exif_transpose(image)
            # Match the detector's white-background alpha normalization before
            # hashing full-resolution pixels, including palette transparency.
            if "A" in corrected.getbands() or "transparency" in corrected.info:
                rgba = corrected.convert("RGBA")
                background = Image.new("RGBA", rgba.size, color=(255, 255, 255, 255))
                pixels = Image.alpha_composite(background, rgba).convert("RGB")
            else:
                pixels = corrected.convert("RGB")
            digest = hashlib.sha256()
            digest.update(f"RGB:{pixels.width}:{pixels.height}:".encode())
            digest.update(pixels.tobytes())
            return _Fingerprint(
                hashlib.sha256(data).hexdigest(), digest.hexdigest(), compute_phash(image)
            )
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        if isinstance(exc, LabeledPairError):
            raise
        raise LabeledPairError("could not decode a labeled-pair image") from exc


def _load_fold(path: Path, root: Path, max_image_pixels: int) -> _Fold:
    try:
        document = path.read_bytes()
        reader = csv.DictReader(io.StringIO(document.decode("utf-8-sig")))
        if reader.fieldnames != list(_COLUMNS):
            raise LabeledPairError(f"CSV header must be {','.join(_COLUMNS)}")
        pairs: list[PairObservation] = []
        images: dict[str, _Fingerprint] = {}
        group_by_image: dict[str, str] = {}
        group_by_content: dict[str, str] = {}
        seen_pairs: set[tuple[str, str]] = set()
        for row in reader:
            if None in row or any(value is None or not value.strip() for value in row.values()):
                raise LabeledPairError("each CSV row must contain five nonempty fields")
            row = {key: value.strip() for key, value in row.items()}
            if row["label"] not in ("0", "1"):
                raise LabeledPairError("label must be 1 (duplicate) or 0 (distinct)")
            label = row["label"] == "1"
            if label != (row["left_group"] == row["right_group"]):
                raise LabeledPairError(
                    "duplicate pairs must share a group; distinct pairs must not"
                )
            names: list[str] = []
            for side in ("left", "right"):
                name, resolved = _resolve_image(root, row[side])
                group = row[f"{side}_group"]
                if name not in images:
                    images[name] = _fingerprint(resolved, max_image_pixels)
                fingerprint = images[name]
                if name in group_by_image and group_by_image[name] != group:
                    raise LabeledPairError("an image cannot belong to multiple source groups")
                for content in (fingerprint.byte_sha256, fingerprint.pixel_sha256):
                    if content in group_by_content and group_by_content[content] != group:
                        raise LabeledPairError("identical image content must share a source group")
                    group_by_content[content] = group
                group_by_image[name] = group
                names.append(name)
            key = tuple(sorted(names))
            if names[0] == names[1] or key in seen_pairs:
                raise LabeledPairError("self-pairs and repeated or reversed pairs are not allowed")
            seen_pairs.add((key[0], key[1]))
            left, right = (images[name] for name in names)
            pairs.append(
                PairObservation(
                    names[0],
                    names[1],
                    label,
                    row["left_group"],
                    row["right_group"],
                    left.byte_sha256 == right.byte_sha256,
                    (left.phash ^ right.phash).bit_count(),
                )
            )
        if {pair.label for pair in pairs} != {False, True}:
            raise LabeledPairError("each fold needs both duplicate and distinct labeled pairs")
        return _Fold(
            tuple(pairs),
            images,
            frozenset(group_by_image.values()),
            hashlib.sha256(document).hexdigest(),
        )
    except (OSError, UnicodeError, csv.Error) as exc:
        raise LabeledPairError("could not read labeled-pair CSV") from exc


def _check_independence(calibration: _Fold, evaluation: _Fold) -> None:
    if calibration.groups & evaluation.groups:
        raise LabeledPairError("source groups overlap between calibration and evaluation")
    if calibration.images.keys() & evaluation.images.keys():
        raise LabeledPairError("image paths overlap between calibration and evaluation")
    for attribute in ("byte_sha256", "pixel_sha256"):
        left = {getattr(image, attribute) for image in calibration.images.values()}
        right = {getattr(image, attribute) for image in evaluation.images.values()}
        if left & right:
            raise LabeledPairError("image content overlaps between calibration and evaluation")


def score_pairs(pairs: tuple[PairObservation, ...], radius: int) -> ThresholdResult:
    """Apply the SHA-or-pHash rule and retain complete confusion counts."""
    if isinstance(radius, bool) or not isinstance(radius, int) or not -1 <= radius <= 64:
        raise LabeledPairError("radius must be an integer from -1 through 64")
    tp = fp = fn = tn = 0
    for pair in pairs:
        predicted = pair.exact or pair.distance <= radius
        tp += predicted and pair.label
        fp += predicted and not pair.label
        fn += not predicted and pair.label
        tn += not predicted and not pair.label
    return ThresholdResult(
        radius=radius,
        pair_count=len(pairs),
        true_negatives=tn,
        metrics=binary_metrics(tp, fp, fn),
    )


def select_radius(
    calibration_pairs: tuple[PairObservation, ...], minimum_precision: float
) -> tuple[ThresholdResult, tuple[ThresholdResult, ...]]:
    """Choose once on calibration; evaluation labels are not an argument."""
    if not math.isfinite(minimum_precision) or not 0 < minimum_precision <= 1:
        raise LabeledPairError("minimum precision must be finite and in (0, 1]")
    sweep = tuple(score_pairs(calibration_pairs, radius) for radius in range(-1, 65))
    eligible = tuple(row for row in sweep if row.metrics.precision >= minimum_precision)
    if not eligible:
        raise LabeledPairError(
            "no calibration threshold meets minimum precision; no result written"
        )
    selected = max(
        eligible,
        key=lambda row: (
            row.metrics.recall,
            row.metrics.precision,
            -row.radius,
        ),
    )
    return selected, sweep


def evaluate_labeled_pairs(
    calibration_csv: Path,
    evaluation_csv: Path,
    dataset_root: Path,
    *,
    label_source: str,
    minimum_precision: float = 0.95,
    max_image_pixels: int = 50_000_000,
    repo_root: Path | None = None,
) -> LabeledPairArtifact:
    """Measure an external corpus without selecting thresholds on its held-out fold."""
    if not label_source.strip():
        raise LabeledPairError("label source must describe who labeled the pairs and the protocol")
    if isinstance(max_image_pixels, bool) or max_image_pixels < 1:
        raise LabeledPairError("max image pixels must be positive")
    root = dataset_root.resolve()
    if not root.is_dir():
        raise LabeledPairError("dataset root must be an existing directory")
    calibration = _load_fold(calibration_csv, root, max_image_pixels)
    evaluation = _load_fold(evaluation_csv, root, max_image_pixels)
    _check_independence(calibration, evaluation)
    selected, sweep = select_radius(calibration.pairs, minimum_precision)
    # The only evaluation score is at the already selected calibration radius.
    held_out = score_pairs(evaluation.pairs, selected.radius)
    configuration_sha256 = canonical_sha256(
        {
            "detector": PHASH_ALGORITHM_ID,
            "minimum_precision": minimum_precision,
            "max_image_pixels": max_image_pixels,
            "label_source": label_source.strip(),
        }
    )
    dataset_sha256 = canonical_sha256(
        {
            "calibration_csv": calibration.csv_sha256,
            "evaluation_csv": evaluation.csv_sha256,
            "images": {
                name: image.byte_sha256
                for name, image in sorted((calibration.images | evaluation.images).items())
            },
        }
    )
    return LabeledPairArtifact(
        label_source=label_source.strip(),
        calibration_csv_sha256=calibration.csv_sha256,
        evaluation_csv_sha256=evaluation.csv_sha256,
        minimum_calibration_precision=minimum_precision,
        selected_radius=selected.radius,
        calibration_sweep=sweep,
        calibration=selected,
        evaluation=held_out,
        metadata=collect_run_metadata(
            configuration_sha256, dataset_sha256, (), repo_root=repo_root
        ),
    )
