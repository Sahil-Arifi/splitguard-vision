"""Deterministic, decode-agnostic dataset manifest ingestion."""

from __future__ import annotations

import csv
import os
import re
import unicodedata
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from splitguard.schemas import ManifestEntry, Split, stable_id

_REQUIRED_CSV_HEADERS = ("path", "split", "label")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")


class ManifestMode(StrEnum):
    IMAGE_FOLDER = "image_folder"
    CSV = "csv"


class ManifestError(ValueError):
    """Raised when a manifest cannot be interpreted safely."""

    def __init__(self, message: str, *, row_number: int | None = None) -> None:
        self.row_number = row_number
        prefix = f"CSV row {row_number}: " if row_number is not None else ""
        super().__init__(f"{prefix}{message}")


@dataclass(frozen=True, slots=True)
class ManifestResult:
    """Runtime manifest plus its local root; the root is never serialized."""

    root: Path
    entries: tuple[ManifestEntry, ...]
    mode: ManifestMode
    source: Path


def _absolute_lexical(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    # abspath normalizes the lexical path without following symbolic links.
    return Path(os.path.abspath(candidate))


def normalize_manifest_path(raw_path: str, *, row_number: int | None = None) -> str:
    """Normalize a user path to a safe, root-relative POSIX representation."""

    value = unicodedata.normalize("NFC", raw_path.strip()).replace("\\", "/")
    if not value:
        raise ManifestError("path cannot be empty", row_number=row_number)
    if "\x00" in value:
        raise ManifestError("path cannot contain NUL characters", row_number=row_number)
    if value.startswith("/") or _WINDOWS_DRIVE_RE.match(value):
        raise ManifestError("path must be relative to the dataset root", row_number=row_number)

    parts: list[str] = []
    for part in value.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                raise ManifestError("path escapes the dataset root", row_number=row_number)
            parts.pop()
            continue
        parts.append(part)

    if not parts:
        raise ManifestError("path must identify a file", row_number=row_number)
    return "/".join(parts)


def _parse_split(value: str, *, location: str, row_number: int | None = None) -> Split:
    normalized = unicodedata.normalize("NFC", value.strip())
    try:
        return Split(normalized)
    except ValueError as exc:
        supported = ", ".join(split.value for split in Split)
        raise ManifestError(
            f"unsupported split in {location}; expected one of: {supported}",
            row_number=row_number,
        ) from exc


def _path_sort_key(path: str) -> tuple[str, str]:
    return (path.casefold(), path)


def _build_entries(
    rows: Sequence[tuple[str, Split, str | None]],
) -> tuple[ManifestEntry, ...]:
    ordered = sorted(
        rows,
        key=lambda row: (
            *_path_sort_key(row[0]),
            row[1].value,
            "" if row[2] is None else row[2],
        ),
    )
    seen_paths: dict[str, str] = {}
    entries: list[ManifestEntry] = []
    for ordinal, (path, split, label) in enumerate(ordered):
        collision_key = unicodedata.normalize("NFC", path).casefold()
        previous = seen_paths.get(collision_key)
        if previous is not None:
            raise ManifestError(
                f"duplicate logical path after normalization: {previous!r} and {path!r}"
            )
        seen_paths[collision_key] = path
        entries.append(
            ManifestEntry(
                id=stable_id("img", path),
                path=path,
                split=split,
                label=label,
                ordinal=ordinal,
            )
        )
    return tuple(entries)


def _scan_directory(directory: Path, *, description: str) -> list[os.DirEntry[str]]:
    try:
        with os.scandir(directory) as iterator:
            return sorted(iterator, key=lambda item: (item.name.casefold(), item.name))
    except OSError as exc:
        raise ManifestError(f"cannot read {description}") from exc


def _walk_class_files(class_directory: Path, dataset_root: Path) -> Iterator[str]:
    for item in _scan_directory(class_directory, description="class directory"):
        relative = normalize_manifest_path(Path(item.path).relative_to(dataset_root).as_posix())
        if item.is_symlink():
            raise ManifestError(f"symbolic links are not supported: {relative!r}")
        try:
            is_directory = item.is_dir(follow_symlinks=False)
        except OSError as exc:
            raise ManifestError(f"cannot inspect dataset entry: {relative!r}") from exc
        if is_directory:
            yield from _walk_class_files(Path(item.path), dataset_root)
        else:
            # Do not inspect extensions or decode content here. Validation owns that work.
            yield relative


def discover_image_folder(dataset_root: str | os.PathLike[str]) -> ManifestResult:
    """Discover ``split/class/file`` entries without opening image content."""

    root = _absolute_lexical(dataset_root)
    if root.is_symlink():
        raise ManifestError("dataset root cannot be a symbolic link")
    if not root.exists():
        raise ManifestError("dataset root does not exist")
    if not root.is_dir():
        raise ManifestError("dataset root must be a directory")

    rows: list[tuple[str, Split, str | None]] = []
    for split_item in _scan_directory(root, description="dataset root"):
        split_name = split_item.name
        if split_item.is_symlink():
            raise ManifestError(f"symbolic links are not supported at split level: {split_name!r}")
        try:
            is_split_directory = split_item.is_dir(follow_symlinks=False)
        except OSError as exc:
            raise ManifestError(f"cannot inspect top-level entry: {split_name!r}") from exc
        if not is_split_directory:
            raise ManifestError(f"top-level entry must be a split directory: {split_name!r}")
        split = _parse_split(split_name, location="top-level directory")

        split_directory = Path(split_item.path)
        for class_item in _scan_directory(split_directory, description="split directory"):
            class_relative = normalize_manifest_path(
                Path(class_item.path).relative_to(root).as_posix()
            )
            if class_item.is_symlink():
                raise ManifestError(
                    f"symbolic links are not supported at class level: {class_relative!r}"
                )
            try:
                is_class_directory = class_item.is_dir(follow_symlinks=False)
            except OSError as exc:
                raise ManifestError(f"cannot inspect class entry: {class_relative!r}") from exc
            if not is_class_directory:
                raise ManifestError(
                    f"split entries must be class directories: {class_relative!r}"
                )
            label = unicodedata.normalize("NFC", class_item.name.strip())
            if not label:
                raise ManifestError(f"class directory has an empty label: {class_relative!r}")
            rows.extend(
                (path, split, label)
                for path in _walk_class_files(Path(class_item.path), root)
            )

    return ManifestResult(
        root=root,
        entries=_build_entries(rows),
        mode=ManifestMode.IMAGE_FOLDER,
        source=root,
    )


def parse_csv_manifest(
    manifest_path: str | os.PathLike[str],
    *,
    dataset_root: str | os.PathLike[str] | None = None,
) -> ManifestResult:
    """Parse ``path,split,label`` rows without checking or opening referenced files."""

    source = _absolute_lexical(manifest_path)
    if not source.exists():
        raise ManifestError("CSV manifest does not exist")
    if not source.is_file():
        raise ManifestError("CSV manifest must be a file")
    root = source.parent if dataset_root is None else _absolute_lexical(dataset_root)

    rows: list[tuple[str, Split, str | None]] = []
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, skipinitialspace=True)
            if reader.fieldnames is None:
                raise ManifestError("CSV manifest is missing a header row")
            normalized_headers = tuple(header.strip() for header in reader.fieldnames)
            if len(set(normalized_headers)) != len(normalized_headers):
                raise ManifestError("CSV manifest contains duplicate headers")
            missing = [
                header for header in _REQUIRED_CSV_HEADERS if header not in normalized_headers
            ]
            if missing:
                missing_list = ", ".join(missing)
                raise ManifestError(f"CSV manifest is missing required headers: {missing_list}")
            header_names = dict(zip(normalized_headers, reader.fieldnames, strict=True))

            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    raise ManifestError(
                        "row has more values than declared headers",
                        row_number=row_number,
                    )
                if all(value is None or not value.strip() for value in row.values()):
                    continue
                raw_path = row.get(header_names["path"])
                raw_split = row.get(header_names["split"])
                raw_label = row.get(header_names["label"])
                if raw_path is None or raw_split is None or raw_label is None:
                    raise ManifestError("row is incomplete", row_number=row_number)
                path = normalize_manifest_path(raw_path, row_number=row_number)
                split = _parse_split(raw_split, location="CSV field", row_number=row_number)
                label_value = unicodedata.normalize("NFC", raw_label.strip())
                rows.append((path, split, label_value or None))
    except UnicodeDecodeError as exc:
        raise ManifestError("CSV manifest must be UTF-8 encoded") from exc
    except csv.Error as exc:
        raise ManifestError("CSV manifest is malformed") from exc
    except OSError as exc:
        raise ManifestError("CSV manifest could not be read") from exc

    return ManifestResult(
        root=root,
        entries=_build_entries(rows),
        mode=ManifestMode.CSV,
        source=source,
    )


def load_manifest(
    source: str | os.PathLike[str],
    *,
    dataset_root: str | os.PathLike[str] | None = None,
) -> ManifestResult:
    """Load a CSV manifest or discover an ImageFolder-style directory."""

    source_path = _absolute_lexical(source)
    if source_path.suffix.casefold() == ".csv":
        return parse_csv_manifest(source_path, dataset_root=dataset_root)
    if dataset_root is not None:
        raise ManifestError("dataset_root is only valid when loading a CSV manifest")
    return discover_image_folder(source_path)


__all__ = [
    "ManifestError",
    "ManifestMode",
    "ManifestResult",
    "discover_image_folder",
    "load_manifest",
    "normalize_manifest_path",
    "parse_csv_manifest",
]
