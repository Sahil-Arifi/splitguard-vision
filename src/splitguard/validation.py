"""Local, non-mutating image validation for manifest entries."""

from __future__ import annotations

import hashlib
import stat
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Annotated, Self

from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import Field, model_validator

from splitguard.config import IOConfig
from splitguard.schemas import (
    ImageRecord,
    ManifestEntry,
    StrictFrozenModel,
    ValidationIssue,
    ValidationIssueCode,
)

DEFAULT_MAX_IMAGE_PIXELS = IOConfig().max_image_pixels
_SUPPORTED_FORMATS = {
    "BMP": "bmp",
    "JPEG": "jpeg",
    "PNG": "png",
    "TIFF": "tiff",
    "WEBP": "webp",
}


def _issue_sort_key(issue: ValidationIssue) -> tuple[str, str, str, str, str, str]:
    return (
        issue.record_id or "",
        issue.path or "",
        issue.split.value if issue.split is not None else "",
        issue.label or "",
        issue.code.value,
        issue.message,
    )


class ScanResult(StrictFrozenModel):
    """Immutable result of validating one manifest snapshot."""

    entry_count: Annotated[int, Field(ge=0)]
    records: tuple[ImageRecord, ...]
    issues: tuple[ValidationIssue, ...]
    duration_seconds: Annotated[float, Field(ge=0.0)]

    @property
    def valid_count(self) -> int:
        return len(self.records)

    @property
    def invalid_count(self) -> int:
        return len(self.issues)

    @model_validator(mode="after")
    def validate_canonical_result(self) -> Self:
        record_ids = tuple(record.id for record in self.records)
        if record_ids != tuple(sorted(set(record_ids))):
            raise ValueError("records must have unique IDs in canonical order")
        if self.issues != tuple(sorted(self.issues, key=_issue_sort_key)):
            raise ValueError("issues must be in canonical order")
        if self.entry_count != len(self.records) + len(self.issues):
            raise ValueError("every manifest entry must produce one record or one issue")
        return self


@dataclass(frozen=True, slots=True)
class _DecodedImage:
    width: int
    height: int
    format: str


@dataclass(frozen=True, slots=True)
class _ImageProblem:
    code: ValidationIssueCode
    message: str


def _make_issue(
    entry: ManifestEntry,
    code: ValidationIssueCode,
    message: str,
) -> ValidationIssue:
    """Build an issue without including exception text or resolved host paths."""
    return ValidationIssue(
        record_id=entry.id,
        path=entry.path,
        split=entry.split,
        label=entry.label,
        code=code,
        message=message,
    )


def _decode_image(contents: bytes, max_image_pixels: int) -> _DecodedImage | _ImageProblem:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(contents)) as image:
                source_format = image.format
                canonical_format = _SUPPORTED_FORMATS.get(source_format or "")
                if canonical_format is None:
                    return _ImageProblem(
                        ValidationIssueCode.UNSUPPORTED_FORMAT,
                        "decoded image format is not supported",
                    )

                source_width, source_height = image.size
                if source_width <= 0 or source_height <= 0:
                    return _ImageProblem(
                        ValidationIssueCode.MALFORMED_IMAGE,
                        "image dimensions are invalid",
                    )
                if source_width * source_height > max_image_pixels:
                    return _ImageProblem(
                        ValidationIssueCode.IMAGE_TOO_LARGE,
                        "decoded image exceeds the configured pixel limit",
                    )

                oriented = ImageOps.exif_transpose(image)
                try:
                    rgb_image = oriented.convert("RGB")
                    try:
                        rgb_image.load()
                        width, height = rgb_image.size
                    finally:
                        rgb_image.close()
                finally:
                    oriented.close()

                if width * height > max_image_pixels:
                    return _ImageProblem(
                        ValidationIssueCode.IMAGE_TOO_LARGE,
                        "processed image exceeds the configured pixel limit",
                    )
                return _DecodedImage(width=width, height=height, format=canonical_format)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning, MemoryError):
        return _ImageProblem(
            ValidationIssueCode.IMAGE_TOO_LARGE,
            "image could not be safely decoded within the configured limit",
        )
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        return _ImageProblem(
            ValidationIssueCode.MALFORMED_IMAGE,
            "image could not be decoded",
        )


def _scan_entry(
    dataset_root: Path,
    entry: ManifestEntry,
    max_image_pixels: int,
) -> ImageRecord | ValidationIssue:
    candidate = dataset_root.joinpath(*PurePosixPath(entry.path).parts)
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError):
        return _make_issue(
            entry,
            ValidationIssueCode.IO_ERROR,
            "path could not be resolved safely",
        )

    if not resolved.is_relative_to(dataset_root):
        return _make_issue(
            entry,
            ValidationIssueCode.PATH_ESCAPE,
            "path resolves outside the dataset root",
        )

    try:
        file_status = resolved.stat()
    except (FileNotFoundError, NotADirectoryError):
        return _make_issue(
            entry,
            ValidationIssueCode.MISSING_PATH,
            "path does not exist",
        )
    except OSError:
        return _make_issue(
            entry,
            ValidationIssueCode.IO_ERROR,
            "path metadata could not be read",
        )

    if not stat.S_ISREG(file_status.st_mode):
        return _make_issue(
            entry,
            ValidationIssueCode.NOT_A_FILE,
            "path is not a regular file",
        )

    try:
        contents = resolved.read_bytes()
    except (FileNotFoundError, NotADirectoryError):
        return _make_issue(
            entry,
            ValidationIssueCode.MISSING_PATH,
            "file disappeared before it could be read",
        )
    except OSError:
        return _make_issue(
            entry,
            ValidationIssueCode.IO_ERROR,
            "file could not be read",
        )

    decoded = _decode_image(contents, max_image_pixels)
    if isinstance(decoded, _ImageProblem):
        return _make_issue(entry, decoded.code, decoded.message)

    return ImageRecord(
        id=entry.id,
        path=entry.path,
        split=entry.split,
        label=entry.label,
        byte_sha256=hashlib.sha256(contents).hexdigest(),
        byte_size=len(contents),
        width=decoded.width,
        height=decoded.height,
        format=decoded.format,
        phash=None,
    )


def scan_images(
    dataset_root: str | Path,
    entries: tuple[ManifestEntry, ...] | list[ManifestEntry],
    *,
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> ScanResult:
    """Validate a manifest snapshot without changing or uploading source images."""
    started = time.perf_counter()
    if isinstance(max_image_pixels, bool) or not isinstance(max_image_pixels, int):
        raise TypeError("max_image_pixels must be an integer")
    if max_image_pixels <= 0:
        raise ValueError("max_image_pixels must be positive")
    if not isinstance(entries, (tuple, list)):
        raise TypeError("entries must be a tuple or list of ManifestEntry values")

    snapshot = tuple(entries)
    if any(not isinstance(entry, ManifestEntry) for entry in snapshot):
        raise TypeError("entries must contain only ManifestEntry values")

    try:
        root = Path(dataset_root).resolve(strict=True)
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError):
        raise ValueError("dataset root must be an existing directory") from None
    if not root.is_dir():
        raise ValueError("dataset root must be an existing directory")

    ordered_entries = tuple(
        sorted(
            snapshot,
            key=lambda entry: (
                entry.id,
                entry.path,
                entry.ordinal,
                entry.split.value,
                entry.label or "",
            ),
        )
    )
    id_counts = Counter(entry.id for entry in ordered_entries)
    path_counts = Counter(entry.path for entry in ordered_entries)

    records: list[ImageRecord] = []
    issues: list[ValidationIssue] = []
    for entry in ordered_entries:
        if id_counts[entry.id] > 1:
            issues.append(
                _make_issue(
                    entry,
                    ValidationIssueCode.INVALID_MANIFEST,
                    "record identifier is duplicated in the manifest",
                )
            )
            continue
        if path_counts[entry.path] > 1:
            issues.append(
                _make_issue(
                    entry,
                    ValidationIssueCode.DUPLICATE_PATH,
                    "image path is duplicated in the manifest",
                )
            )
            continue

        outcome = _scan_entry(root, entry, max_image_pixels)
        if isinstance(outcome, ImageRecord):
            records.append(outcome)
        else:
            issues.append(outcome)

    duration = max(0.0, time.perf_counter() - started)
    return ScanResult(
        entry_count=len(snapshot),
        records=tuple(sorted(records, key=lambda record: record.id)),
        issues=tuple(sorted(issues, key=_issue_sort_key)),
        duration_seconds=duration,
    )


__all__ = ["DEFAULT_MAX_IMAGE_PIXELS", "ScanResult", "scan_images"]
