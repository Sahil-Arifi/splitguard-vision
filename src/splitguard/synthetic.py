"""Deterministic, local-only controlled image corruption generation.

Ground truth in this module is derived from the requested transformation, never
from a detector score.  That separation lets benchmarks measure detector misses
without quietly redefining the expected answer from the detector itself.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import tempfile
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self

from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError
from pydantic import Field, field_validator, model_validator

from splitguard.schemas import ImageRecord, Sha256, Split, StableId, StrictFrozenModel, stable_id

_DRIVE_PATH = re.compile(r"^[A-Za-z]:")
_CORRUPT_PAYLOAD_PREFIX = b"SPLITGUARD-DELIBERATELY-MALFORMED\x00"
_SPLIT_ORDER = {split: index for index, split in enumerate(Split)}


class SyntheticInputError(ValueError):
    """Raised when controlled-corruption inputs are unsafe or inconsistent."""


class CorruptionType(StrEnum):
    """Controlled defect families supported by the offline generator."""

    EXACT_COPY = "exact_copy"
    JPEG_RECOMPRESSION = "jpeg_recompression"
    RESIZE = "resize"
    SMALL_CROP = "small_crop"
    BRIGHTNESS_SHIFT = "brightness_shift"
    GAUSSIAN_BLUR = "gaussian_blur"
    CROSS_SPLIT_COPY = "cross_split_copy"
    CROSS_LABEL_DUPLICATE = "cross_label_duplicate"
    MALFORMED_FILE = "malformed_file"


class ExpectedRelationship(StrEnum):
    """Detector-independent relationship asserted for an injected item."""

    EXACT_DUPLICATE = "exact_duplicate"
    TRANSFORMED_DUPLICATE = "transformed_duplicate"
    INVALID_IMAGE = "invalid_image"
    UNRELATED = "unrelated"


_CORRUPTION_ORDER = {kind: index for index, kind in enumerate(CorruptionType)}
_EXACT_CORRUPTIONS = frozenset(
    {
        CorruptionType.EXACT_COPY,
        CorruptionType.CROSS_SPLIT_COPY,
        CorruptionType.CROSS_LABEL_DUPLICATE,
    }
)
_TRANSFORMED_CORRUPTIONS = frozenset(
    {
        CorruptionType.JPEG_RECOMPRESSION,
        CorruptionType.RESIZE,
        CorruptionType.SMALL_CROP,
        CorruptionType.BRIGHTNESS_SHIFT,
        CorruptionType.GAUSSIAN_BLUR,
    }
)


def _relative_posix_path(value: str) -> str:
    if not value or "\x00" in value or "\\" in value:
        raise ValueError("path must be a non-empty NUL-free POSIX relative path")
    if value.startswith("/") or _DRIVE_PATH.match(value):
        raise ValueError("path must be relative")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("path cannot contain empty, current, or parent segments")
    if PurePosixPath(value).is_absolute():
        raise ValueError("path must be relative")
    return value


class InjectedDefect(StrictFrozenModel):
    """One source-derived pair and its independently declared ground truth."""

    source_id: StableId
    derived_id: StableId
    source_path: Annotated[str, Field(min_length=1, max_length=4096)]
    derived_path: Annotated[str, Field(min_length=1, max_length=4096)]
    source_sha256: Sha256
    derived_sha256: Sha256
    corruption_type: CorruptionType
    source_split: Split
    target_split: Split
    source_label: Annotated[str, Field(min_length=1, max_length=512)] | None = None
    target_label: Annotated[str, Field(min_length=1, max_length=512)] | None = None
    expected_relationship: ExpectedRelationship
    is_duplicate: bool
    exact_bytes: bool
    valid_image: bool

    _source_path = field_validator("source_path")(_relative_posix_path)
    _derived_path = field_validator("derived_path")(_relative_posix_path)

    @model_validator(mode="after")
    def relationship_matches_declared_corruption(self) -> Self:
        if self.source_id == self.derived_id:
            raise ValueError("source and derived IDs must differ")
        if self.derived_id != stable_id("img", self.derived_path):
            raise ValueError("derived_id must be determined by derived_path")

        if self.corruption_type in _EXACT_CORRUPTIONS:
            expected = (ExpectedRelationship.EXACT_DUPLICATE, True, True, True)
        elif self.corruption_type in _TRANSFORMED_CORRUPTIONS:
            expected = (ExpectedRelationship.TRANSFORMED_DUPLICATE, True, False, True)
        else:
            expected = (ExpectedRelationship.INVALID_IMAGE, False, False, False)
        actual = (
            self.expected_relationship,
            self.is_duplicate,
            self.exact_bytes,
            self.valid_image,
        )
        if actual != expected:
            raise ValueError("ground truth does not match the declared corruption")
        if self.exact_bytes != (self.source_sha256 == self.derived_sha256):
            raise ValueError("exact_bytes must agree with the payload SHA-256 values")
        if (
            self.corruption_type is CorruptionType.CROSS_SPLIT_COPY
            and self.source_split is self.target_split
        ):
            raise ValueError("cross-split copy must change the split")
        if (
            self.corruption_type is CorruptionType.CROSS_LABEL_DUPLICATE
            and self.source_label == self.target_label
        ):
            raise ValueError("cross-label duplicate must change the label")
        return self

    @property
    def canonical_pair(self) -> tuple[str, str]:
        """Return the detector-facing unordered pair key."""

        left_id, right_id = sorted((self.source_id, self.derived_id))
        return left_id, right_id


class NegativeControlPair(StrictFrozenModel):
    """An independently labeled source/other-source-derived nonduplicate pair."""

    source_id: StableId
    unrelated_derived_id: StableId
    source_path: Annotated[str, Field(min_length=1, max_length=4096)]
    unrelated_derived_path: Annotated[str, Field(min_length=1, max_length=4096)]
    source_sha256: Sha256
    unrelated_derived_sha256: Sha256
    corruption_type: CorruptionType
    source_split: Split
    comparison_split: Split
    source_label: Annotated[str, Field(min_length=1, max_length=512)] | None = None
    comparison_label: Annotated[str, Field(min_length=1, max_length=512)] | None = None
    expected_relationship: Literal[ExpectedRelationship.UNRELATED] = (
        ExpectedRelationship.UNRELATED
    )
    is_duplicate: Literal[False] = False

    _source_path = field_validator("source_path")(_relative_posix_path)
    _derived_path = field_validator("unrelated_derived_path")(_relative_posix_path)

    @model_validator(mode="after")
    def valid_negative_pair(self) -> Self:
        if self.source_id == self.unrelated_derived_id:
            raise ValueError("negative-control endpoints must differ")
        if self.corruption_type is CorruptionType.MALFORMED_FILE:
            raise ValueError("malformed files are validation cases, not pairwise controls")
        if self.source_sha256 == self.unrelated_derived_sha256:
            raise ValueError("negative-control payloads must be byte-distinct")
        return self

    @property
    def canonical_pair(self) -> tuple[str, str]:
        left_id, right_id = sorted((self.source_id, self.unrelated_derived_id))
        return left_id, right_id


class SyntheticCorruptionSet(StrictFrozenModel):
    """Portable result of one controlled corruption run."""

    generator: str = "splitguard-controlled-corruptions-v1"
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)]
    source_ids: tuple[StableId, ...]
    derived_records: tuple[ImageRecord, ...]
    injections: tuple[InjectedDefect, ...]
    negative_controls: tuple[NegativeControlPair, ...] = ()
    invalid_derived_ids: tuple[StableId, ...]

    @model_validator(mode="after")
    def collections_are_complete_and_canonical(self) -> Self:
        if self.source_ids != tuple(sorted(set(self.source_ids))):
            raise ValueError("source_ids must be sorted and unique")
        record_ids = tuple(record.id for record in self.derived_records)
        if record_ids != tuple(sorted(set(record_ids))):
            raise ValueError("derived_records must have unique IDs in canonical order")
        injection_keys = tuple(
            (item.source_id, _CORRUPTION_ORDER[item.corruption_type])
            for item in self.injections
        )
        if injection_keys != tuple(sorted(set(injection_keys))):
            raise ValueError("injections must be unique and in canonical order")
        if self.invalid_derived_ids != tuple(sorted(set(self.invalid_derived_ids))):
            raise ValueError("invalid_derived_ids must be sorted and unique")

        control_keys = tuple(
            (item.corruption_type.value, *item.canonical_pair)
            for item in self.negative_controls
        )
        if control_keys != tuple(sorted(set(control_keys))):
            raise ValueError("negative_controls must be unique and in canonical order")
        expected_control_keys = (
            {
                (item.source_id, item.corruption_type)
                for item in self.injections
                if item.valid_image
            }
            if len(self.source_ids) >= 2
            else set()
        )
        actual_control_keys = {
            (item.source_id, item.corruption_type) for item in self.negative_controls
        }
        if actual_control_keys != expected_control_keys:
            raise ValueError(
                "negative controls must cover every source and valid corruption exactly once"
            )

        valid_ids = {item.derived_id for item in self.injections if item.valid_image}
        invalid_ids = {item.derived_id for item in self.injections if not item.valid_image}
        if valid_ids != set(record_ids):
            raise ValueError("derived_records must exactly cover valid injections")
        if invalid_ids != set(self.invalid_derived_ids):
            raise ValueError("invalid_derived_ids must exactly cover malformed injections")
        if not {item.source_id for item in self.injections} <= set(self.source_ids):
            raise ValueError("every injection must reference a declared source ID")
        injection_ids = {item.derived_id for item in self.injections}
        if any(
            item.source_id not in self.source_ids
            or item.unrelated_derived_id not in injection_ids
            for item in self.negative_controls
        ):
            raise ValueError("negative controls must reference declared sources and injections")
        source_metadata: dict[str, tuple[str, Split, str | None, str]] = {}
        for injection in self.injections:
            declared = (
                injection.source_path,
                injection.source_split,
                injection.source_label,
                injection.source_sha256,
            )
            previous = source_metadata.setdefault(injection.source_id, declared)
            if previous != declared:
                raise ValueError("one source ID cannot declare inconsistent metadata")
        injections_by_derived = {item.derived_id: item for item in self.injections}
        record_hashes = {record.id: record.byte_sha256 for record in self.derived_records}
        if any(
            record_hashes[item.derived_id] != item.derived_sha256
            for item in self.injections
            if item.valid_image
        ):
            raise ValueError("derived records must preserve injected payload hashes")
        for control in self.negative_controls:
            declared_source = source_metadata.get(control.source_id)
            if declared_source is None:
                raise ValueError("negative controls must reference an injected source")
            source_path, source_split, source_label, source_sha256 = declared_source
            compared = injections_by_derived[control.unrelated_derived_id]
            if compared.source_id == control.source_id:
                raise ValueError("negative controls must originate from another source")
            if not compared.valid_image:
                raise ValueError("negative controls must reference valid derived images")
            if (
                control.source_path != source_path
                or control.source_split is not source_split
                or control.source_label != source_label
                or control.source_sha256 != source_sha256
                or control.corruption_type is not compared.corruption_type
                or control.unrelated_derived_path != compared.derived_path
                or control.comparison_split is not compared.target_split
                or control.comparison_label != compared.target_label
                or control.unrelated_derived_sha256 != compared.derived_sha256
            ):
                raise ValueError("negative-control metadata must match its endpoints")
        return self


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be between 0 and 2**32 - 1")
    return seed


def _canonical_corruptions(
    corruption_types: Iterable[CorruptionType],
) -> tuple[CorruptionType, ...]:
    values = tuple(corruption_types)
    if any(not isinstance(value, CorruptionType) for value in values):
        raise TypeError("corruption_types must contain CorruptionType values")
    if len(values) != len(set(values)):
        raise SyntheticInputError("corruption_types cannot contain duplicates")
    return tuple(sorted(values, key=_CORRUPTION_ORDER.__getitem__))


def _absolute(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return Path(os.path.abspath(candidate))


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validated_roots(
    dataset_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
) -> tuple[Path, Path]:
    source = _absolute(dataset_root)
    output = _absolute(output_root)
    try:
        resolved_source = source.resolve(strict=True)
    except OSError as exc:
        raise SyntheticInputError("source dataset root is unavailable") from exc
    if not resolved_source.is_dir() or source.is_symlink():
        raise SyntheticInputError("source dataset root must be a non-symlink directory")
    resolved_output = output.resolve(strict=False)
    if _is_within(resolved_output, resolved_source) or _is_within(
        resolved_source, resolved_output
    ):
        raise SyntheticInputError("source and output roots must not overlap")
    if output.exists():
        raise SyntheticInputError("output root must not already exist")
    return resolved_source, output


def _read_verified_source(root: Path, record: ImageRecord) -> tuple[bytes, Image.Image]:
    candidate = root.joinpath(*PurePosixPath(record.path).parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise SyntheticInputError(f"source record {record.id} is unavailable") from exc
    if not _is_within(resolved, root):
        raise SyntheticInputError(f"source record {record.id} escapes the dataset root")

    current = root
    for part in PurePosixPath(record.path).parts:
        current = current / part
        if current.is_symlink():
            raise SyntheticInputError(f"source record {record.id} traverses a symbolic link")
    if not resolved.is_file():
        raise SyntheticInputError(f"source record {record.id} is not a file")
    try:
        payload = resolved.read_bytes()
    except OSError as exc:
        raise SyntheticInputError(f"source record {record.id} could not be read") from exc
    if hashlib.sha256(payload).hexdigest() != record.byte_sha256:
        raise SyntheticInputError(f"source record {record.id} content hash changed")
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise SyntheticInputError(f"source record {record.id} is not a decodable image") from exc
    if image.size != (record.width, record.height):
        image.close()
        raise SyntheticInputError(f"source record {record.id} dimensions changed")
    return payload, image


def _derived_parameters(seed: int, source_id: str, corruption: CorruptionType) -> bytes:
    return hashlib.sha256(f"{seed}\0{source_id}\0{corruption.value}".encode()).digest()


def _decoded_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    try:
        digest = hashlib.sha256()
        digest.update(rgb.width.to_bytes(8, "big", signed=False))
        digest.update(rgb.height.to_bytes(8, "big", signed=False))
        digest.update(rgb.tobytes())
        return digest.hexdigest()
    finally:
        rgb.close()


def _decoded_payload_sha256(payload: bytes) -> str:
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            opened.load()
            normalized = ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, ValueError, UnidentifiedImageError) as exc:  # pragma: no cover
        raise SyntheticInputError("an internally generated image could not be decoded") from exc
    try:
        return _decoded_sha256(normalized)
    finally:
        normalized.close()


def _encode_transformation(
    image: Image.Image,
    corruption: CorruptionType,
    parameters: bytes,
) -> tuple[bytes, str]:
    output = io.BytesIO()
    if corruption is CorruptionType.JPEG_RECOMPRESSION:
        quality = 68 + parameters[0] % 11
        image.save(
            output,
            format="JPEG",
            quality=quality,
            optimize=False,
            progressive=False,
            subsampling=2,
        )
        return output.getvalue(), "jpeg"

    transformed: Image.Image
    if corruption is CorruptionType.RESIZE:
        scale = 0.68 + (parameters[0] % 13) / 100.0
        width = max(1, round(image.width * scale))
        height = max(1, round(image.height * scale))
        if image.width > 1 and width == image.width:
            width -= 1
        if image.height > 1 and height == image.height:
            height -= 1
        transformed = image.resize((width, height), Image.Resampling.LANCZOS)
    elif corruption is CorruptionType.SMALL_CROP:
        horizontal = min(max(1, image.width // 16), max(0, (image.width - 1) // 2))
        vertical = min(max(1, image.height // 16), max(0, (image.height - 1) // 2))
        transformed = image.crop(
            (horizontal, vertical, image.width - horizontal, image.height - vertical)
        )
    elif corruption is CorruptionType.BRIGHTNESS_SHIFT:
        factor = 0.78 if parameters[0] % 2 == 0 else 1.22
        transformed = ImageEnhance.Brightness(image).enhance(factor)
    elif corruption is CorruptionType.GAUSSIAN_BLUR:
        radius = 0.8 + (parameters[0] % 8) / 10.0
        transformed = image.filter(ImageFilter.GaussianBlur(radius=radius))
    else:  # pragma: no cover - callers constrain this internal helper
        raise AssertionError(f"unsupported transformed corruption: {corruption}")

    try:
        transformed.save(
            output,
            format="PNG",
            compress_level=1 + parameters[-1] % 8,
            optimize=False,
        )
        return output.getvalue(), "png"
    finally:
        transformed.close()


def _encode_distinct_transformation(
    image: Image.Image,
    corruption: CorruptionType,
    parameters: bytes,
    source_payload: bytes,
) -> tuple[bytes, str]:
    for attempt in range(16):
        attempt_parameters = (
            parameters
            if attempt == 0
            else hashlib.sha256(
                parameters + attempt.to_bytes(2, "big", signed=False)
            ).digest()
        )
        payload, output_format = _encode_transformation(
            image,
            corruption,
            attempt_parameters,
        )
        if payload != source_payload:
            return payload, output_format
    raise SyntheticInputError(
        f"{corruption.value} could not produce a byte-distinct transformed image"
    )


def _next_split(source: Split) -> Split:
    standard = (Split.TRAIN, Split.VAL, Split.TEST)
    if source not in standard:
        return Split.TRAIN
    return standard[(standard.index(source) + 1) % len(standard)]


def _conflicting_label(source_label: str | None) -> str:
    primary = "splitguard_conflict"
    return "splitguard_conflict_alt" if source_label == primary else primary


def _extension_for_exact(record: ImageRecord) -> str:
    return "jpg" if record.format == "jpeg" else record.format


def _derived_path(
    record: ImageRecord,
    corruption: CorruptionType,
    target_split: Split,
    extension: str,
) -> str:
    digest = stable_id("derived", record.id, corruption.value).split("_", maxsplit=1)[1]
    return f"{target_split.value}/synthetic/{digest}-{corruption.value}.{extension}"


def _image_record_from_payload(
    path: str,
    split: Split,
    label: str | None,
    payload: bytes,
    expected_format: str,
) -> ImageRecord:
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            opened.load()
            width, height = opened.size
    except (OSError, ValueError, UnidentifiedImageError) as exc:  # pragma: no cover
        raise SyntheticInputError("an internally generated image could not be decoded") from exc
    return ImageRecord(
        id=stable_id("img", path),
        path=path,
        split=split,
        label=label,
        byte_sha256=hashlib.sha256(payload).hexdigest(),
        byte_size=len(payload),
        width=width,
        height=height,
        format=expected_format,
    )


def generate_controlled_corruptions(
    dataset_root: str | os.PathLike[str],
    records: Iterable[ImageRecord],
    output_root: str | os.PathLike[str],
    *,
    seed: int = 0,
    corruption_types: Iterable[CorruptionType] = tuple(CorruptionType),
) -> SyntheticCorruptionSet:
    """Generate a portable corruption set without modifying source data.

    ``output_root`` must not exist and must be disjoint from ``dataset_root``.
    Work is built in a sibling staging directory and installed with one atomic
    directory rename, so a failed run leaves no partial output tree.
    """

    accepted_seed = _validate_seed(seed)
    accepted_corruptions = _canonical_corruptions(corruption_types)
    source_root, destination = _validated_roots(dataset_root, output_root)
    source_records = tuple(records)
    records_by_id = {record.id: record for record in source_records}
    if len(records_by_id) != len(source_records):
        raise SyntheticInputError("source records must contain unique IDs")
    ordered_records = tuple(sorted(source_records, key=lambda record: record.id))
    source_hashes = tuple(record.byte_sha256 for record in ordered_records)
    if len(source_hashes) != len(set(source_hashes)):
        raise SyntheticInputError(
            "controlled benchmark sources must be clean and byte-distinct"
        )
    source_decoded_hashes: dict[str, str] = {}
    decoded_hash_owner: dict[str, str] = {}
    for record in ordered_records:
        _, source_image = _read_verified_source(source_root, record)
        try:
            decoded_hash = _decoded_sha256(source_image)
        finally:
            source_image.close()
        previous_owner = decoded_hash_owner.setdefault(decoded_hash, record.id)
        if previous_owner != record.id:
            raise SyntheticInputError(
                "controlled benchmark sources must be decoded-distinct"
            )
        source_decoded_hashes[record.id] = decoded_hash

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".splitguard-synthetic-", dir=destination.parent)
    )
    derived_records: list[ImageRecord] = []
    injections: list[InjectedDefect] = []
    invalid_ids: list[str] = []
    derived_decoded_hashes: dict[str, str] = {}
    try:
        for record in ordered_records:
            source_payload, source_image = _read_verified_source(source_root, record)
            try:
                for corruption in accepted_corruptions:
                    target_split = (
                        _next_split(record.split)
                        if corruption is CorruptionType.CROSS_SPLIT_COPY
                        else record.split
                    )
                    target_label = (
                        _conflicting_label(record.label)
                        if corruption is CorruptionType.CROSS_LABEL_DUPLICATE
                        else record.label
                    )
                    parameters = _derived_parameters(accepted_seed, record.id, corruption)

                    if corruption in _EXACT_CORRUPTIONS:
                        payload = source_payload
                        output_format = record.format
                        extension = _extension_for_exact(record)
                        relationship = ExpectedRelationship.EXACT_DUPLICATE
                        is_duplicate, exact_bytes, valid_image = True, True, True
                    elif corruption in _TRANSFORMED_CORRUPTIONS:
                        payload, output_format = _encode_distinct_transformation(
                            source_image,
                            corruption,
                            parameters,
                            source_payload,
                        )
                        extension = "jpg" if output_format == "jpeg" else output_format
                        relationship = ExpectedRelationship.TRANSFORMED_DUPLICATE
                        is_duplicate, exact_bytes, valid_image = True, False, True
                    else:
                        payload = (
                            _CORRUPT_PAYLOAD_PREFIX
                            + parameters
                            + record.id.encode("ascii")
                        )
                        output_format = "bin"
                        extension = "corrupt"
                        relationship = ExpectedRelationship.INVALID_IMAGE
                        is_duplicate, exact_bytes, valid_image = False, False, False

                    relative_path = _derived_path(
                        record,
                        corruption,
                        target_split,
                        extension,
                    )
                    derived_id = stable_id("img", relative_path)
                    derived_sha256 = hashlib.sha256(payload).hexdigest()
                    output_path = staging.joinpath(*PurePosixPath(relative_path).parts)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(payload)

                    injection = InjectedDefect(
                        source_id=record.id,
                        derived_id=derived_id,
                        source_path=record.path,
                        derived_path=relative_path,
                        source_sha256=record.byte_sha256,
                        derived_sha256=derived_sha256,
                        corruption_type=corruption,
                        source_split=record.split,
                        target_split=target_split,
                        source_label=record.label,
                        target_label=target_label,
                        expected_relationship=relationship,
                        is_duplicate=is_duplicate,
                        exact_bytes=exact_bytes,
                        valid_image=valid_image,
                    )
                    injections.append(injection)
                    if valid_image:
                        derived_decoded_hashes[derived_id] = _decoded_payload_sha256(payload)
                        derived_records.append(
                            _image_record_from_payload(
                                relative_path,
                                target_split,
                                target_label,
                                payload,
                                output_format,
                            )
                        )
                    else:
                        invalid_ids.append(derived_id)
            finally:
                source_image.close()

        ordered_injections = tuple(
            sorted(
                injections,
                key=lambda item: (item.source_id, _CORRUPTION_ORDER[item.corruption_type]),
            )
        )
        negative_controls: list[NegativeControlPair] = []
        if len(ordered_records) >= 2:
            injections_by_source_and_type = {
                (item.source_id, item.corruption_type): item for item in ordered_injections
            }
            for index, source_record in enumerate(ordered_records):
                unrelated_source = ordered_records[(index + 1) % len(ordered_records)]
                for corruption in accepted_corruptions:
                    if corruption is CorruptionType.MALFORMED_FILE:
                        continue
                    unrelated_injection = injections_by_source_and_type[
                        (unrelated_source.id, corruption)
                    ]
                    if (
                        source_decoded_hashes[source_record.id]
                        == derived_decoded_hashes[unrelated_injection.derived_id]
                    ):
                        raise SyntheticInputError(
                            "negative-control endpoints must be decoded-distinct"
                        )
                    negative_controls.append(
                        NegativeControlPair(
                            source_id=source_record.id,
                            unrelated_derived_id=unrelated_injection.derived_id,
                            source_path=source_record.path,
                            unrelated_derived_path=unrelated_injection.derived_path,
                            source_sha256=source_record.byte_sha256,
                            unrelated_derived_sha256=unrelated_injection.derived_sha256,
                            corruption_type=corruption,
                            source_split=source_record.split,
                            comparison_split=unrelated_injection.target_split,
                            source_label=source_record.label,
                            comparison_label=unrelated_injection.target_label,
                        )
                    )
        ordered_controls = tuple(
            sorted(
                negative_controls,
                key=lambda item: (item.corruption_type.value, *item.canonical_pair),
            )
        )
        result = SyntheticCorruptionSet(
            seed=accepted_seed,
            source_ids=tuple(sorted(records_by_id)),
            derived_records=tuple(sorted(derived_records, key=lambda record: record.id)),
            injections=ordered_injections,
            negative_controls=ordered_controls,
            invalid_derived_ids=tuple(sorted(invalid_ids)),
        )
        os.replace(staging, destination)
        return result
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "CorruptionType",
    "ExpectedRelationship",
    "InjectedDefect",
    "NegativeControlPair",
    "SyntheticCorruptionSet",
    "SyntheticInputError",
    "generate_controlled_corruptions",
]
