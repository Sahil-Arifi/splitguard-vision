"""Tests for local, explicit, and privacy-safe image validation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from splitguard.schemas import ManifestEntry, Split, ValidationIssueCode, stable_id
from splitguard.validation import ScanResult, scan_images


def _entry(path: str, ordinal: int = 0, *, id_part: str | None = None) -> ManifestEntry:
    return ManifestEntry(
        id=stable_id("image", id_part or path),
        path=path,
        split=Split.TRAIN,
        label="example",
        ordinal=ordinal,
    )


def _save_image(path: Path, image_format: str, *, size: tuple[int, int] = (9, 7)) -> None:
    mode = "RGB" if image_format == "JPEG" else "RGBA"
    Image.new(mode, size, color=(25, 50, 75) if mode == "RGB" else (25, 50, 75, 200)).save(
        path,
        format=image_format,
    )


@pytest.mark.parametrize(
    ("suffix", "image_format", "expected_format"),
    [
        (".jpg", "JPEG", "jpeg"),
        (".png", "PNG", "png"),
        (".webp", "WEBP", "webp"),
        (".bmp", "BMP", "bmp"),
        (".tiff", "TIFF", "tiff"),
    ],
)
def test_supported_images_are_fully_read_and_recorded(
    tmp_path: Path,
    suffix: str,
    image_format: str,
    expected_format: str,
) -> None:
    image_path = tmp_path / f"sample{suffix}"
    _save_image(image_path, image_format)
    original_contents = image_path.read_bytes()

    result = scan_images(tmp_path, [_entry(image_path.name)])

    assert result.entry_count == 1
    assert result.valid_count == 1
    assert result.invalid_count == 0
    assert result.duration_seconds >= 0.0
    record = result.records[0]
    assert (record.width, record.height) == (9, 7)
    assert record.format == expected_format
    assert record.byte_size == len(original_contents)
    assert record.byte_sha256 == hashlib.sha256(original_contents).hexdigest()
    assert image_path.read_bytes() == original_contents


def test_exif_orientation_is_applied_before_dimensions_are_recorded(tmp_path: Path) -> None:
    image_path = tmp_path / "oriented.jpg"
    image = Image.new("RGB", (11, 7), color=(1, 2, 3))
    exif = image.getexif()
    exif[274] = 6
    image.save(image_path, format="JPEG", exif=exif)

    result = scan_images(tmp_path, (_entry(image_path.name),))

    assert not result.issues
    assert (result.records[0].width, result.records[0].height) == (7, 11)


def test_result_order_is_deterministic_for_tuple_and_list_inputs(tmp_path: Path) -> None:
    _save_image(tmp_path / "a.png", "PNG")
    _save_image(tmp_path / "b.png", "PNG")
    entries = [_entry("b.png", 1), _entry("a.png", 0)]

    from_list = scan_images(tmp_path, entries)
    from_tuple = scan_images(tmp_path, tuple(reversed(entries)))

    assert from_list.records == from_tuple.records
    assert tuple(record.id for record in from_list.records) == tuple(
        sorted(record.id for record in from_list.records)
    )


def test_scan_result_is_immutable(tmp_path: Path) -> None:
    _save_image(tmp_path / "image.png", "PNG")
    result = scan_images(tmp_path, [_entry("image.png")])

    with pytest.raises(ValidationError, match="frozen_instance"):
        result.entry_count = 2


def test_missing_path_and_directory_are_explicit_issues(tmp_path: Path) -> None:
    (tmp_path / "folder.png").mkdir()

    result = scan_images(
        tmp_path,
        [_entry("missing.png"), _entry("folder.png")],
    )

    assert not result.records
    assert result.entry_count == 2
    assert {issue.code for issue in result.issues} == {
        ValidationIssueCode.MISSING_PATH,
        ValidationIssueCode.NOT_A_FILE,
    }


def test_malformed_and_unsupported_images_are_distinct_and_sanitized(tmp_path: Path) -> None:
    secret_marker = "PRIVATE-IMAGE-CONTENT"
    (tmp_path / "malformed.jpg").write_bytes(secret_marker.encode())
    Image.new("P", (4, 4)).save(tmp_path / "unsupported.gif", format="GIF")

    result = scan_images(
        tmp_path,
        [_entry("unsupported.gif"), _entry("malformed.jpg")],
    )

    assert not result.records
    assert {issue.code for issue in result.issues} == {
        ValidationIssueCode.MALFORMED_IMAGE,
        ValidationIssueCode.UNSUPPORTED_FORMAT,
    }
    for issue in result.issues:
        assert str(tmp_path.resolve()) not in issue.message
        assert secret_marker not in issue.message


def test_decoded_pixel_limit_is_enforced_before_processing(tmp_path: Path) -> None:
    _save_image(tmp_path / "large.png", "PNG", size=(11, 10))

    result = scan_images(
        tmp_path,
        [_entry("large.png")],
        max_image_pixels=100,
    )

    assert not result.records
    assert result.issues[0].code is ValidationIssueCode.IMAGE_TOO_LARGE


@pytest.mark.parametrize("pixel_limit", [0, -1])
def test_pixel_limit_must_be_positive(pixel_limit: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        scan_images(Path.cwd(), [], max_image_pixels=pixel_limit)


def test_symlink_escape_is_rejected_when_symlinks_are_available(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    outside = tmp_path / "outside.png"
    _save_image(outside, "PNG")
    link = dataset_root / "linked.png"
    try:
        link.symlink_to(outside)
    except (NotImplementedError, OSError):
        pytest.skip("file symlinks are not available in this environment")

    result = scan_images(dataset_root, [_entry("linked.png")])

    assert not result.records
    assert result.issues[0].code is ValidationIssueCode.PATH_ESCAPE
    assert str(outside.resolve()) not in result.issues[0].message


def test_duplicate_ids_and_paths_are_never_silently_scanned(tmp_path: Path) -> None:
    _save_image(tmp_path / "a.png", "PNG")
    _save_image(tmp_path / "b.png", "PNG")
    duplicate_id = [_entry("a.png", id_part="same"), _entry("b.png", id_part="same")]
    duplicate_path = [_entry("a.png", 0, id_part="one"), _entry("a.png", 1, id_part="two")]

    id_result = scan_images(tmp_path, duplicate_id)
    path_result = scan_images(tmp_path, duplicate_path)

    assert len(id_result.issues) == 2
    assert {issue.code for issue in id_result.issues} == {
        ValidationIssueCode.INVALID_MANIFEST
    }
    assert len(path_result.issues) == 2
    assert {issue.code for issue in path_result.issues} == {
        ValidationIssueCode.DUPLICATE_PATH
    }


def test_dataset_root_validation_does_not_expose_host_path(tmp_path: Path) -> None:
    missing_root = tmp_path / "private" / "missing"

    with pytest.raises(ValueError, match="dataset root must be an existing directory") as error:
        scan_images(missing_root, [])

    assert str(missing_root) not in str(error.value)


def test_scan_result_rejects_noncanonical_or_incomplete_results(tmp_path: Path) -> None:
    _save_image(tmp_path / "a.png", "PNG")
    valid = scan_images(tmp_path, [_entry("a.png")])

    with pytest.raises(ValidationError, match="one record or one issue"):
        ScanResult(
            entry_count=2,
            records=valid.records,
            issues=(),
            duration_seconds=0.0,
        )
