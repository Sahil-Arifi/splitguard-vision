from __future__ import annotations

from pathlib import Path

import pytest

from splitguard.manifest import (
    ManifestError,
    ManifestMode,
    discover_image_folder,
    load_manifest,
    normalize_manifest_path,
    parse_csv_manifest,
)
from splitguard.schemas import Split, stable_id


def write_file(path: Path, content: bytes = b"not necessarily an image") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def create_image_folder(root: Path) -> None:
    # Deliberately create files out of lexical order; discovery must canonicalize it.
    write_file(root / "val" / "dog" / "z.bin")
    write_file(root / "train" / "cat" / "nested" / "b.jpg")
    write_file(root / "train" / "cat" / "a.png")
    write_file(root / "custom" / "bird" / "broken.anything")


def test_discovers_image_folder_deterministically_without_decoding(tmp_path: Path) -> None:
    create_image_folder(tmp_path)

    result = discover_image_folder(tmp_path)

    assert result.mode is ManifestMode.IMAGE_FOLDER
    assert result.root == tmp_path
    assert [entry.path for entry in result.entries] == [
        "custom/bird/broken.anything",
        "train/cat/a.png",
        "train/cat/nested/b.jpg",
        "val/dog/z.bin",
    ]
    assert [entry.ordinal for entry in result.entries] == [0, 1, 2, 3]
    assert [entry.split for entry in result.entries] == [
        Split.CUSTOM,
        Split.TRAIN,
        Split.TRAIN,
        Split.VAL,
    ]
    assert [entry.label for entry in result.entries] == ["bird", "cat", "cat", "dog"]
    assert result.entries[0].id == stable_id("img", result.entries[0].path)


def test_image_folder_ids_and_entries_are_stable_across_roots(tmp_path: Path) -> None:
    first_root = tmp_path / "one"
    second_root = tmp_path / "elsewhere" / "two"
    create_image_folder(first_root)
    create_image_folder(second_root)

    first = discover_image_folder(first_root)
    second = discover_image_folder(second_root)

    assert first.entries == second.entries
    assert first.root != second.root


def test_image_folder_rejects_unsupported_split(tmp_path: Path) -> None:
    write_file(tmp_path / "validation" / "cat" / "a.png")

    with pytest.raises(ManifestError, match="unsupported split"):
        discover_image_folder(tmp_path)


def test_image_folder_requires_split_and_class_directories(tmp_path: Path) -> None:
    write_file(tmp_path / "root-file.png")
    with pytest.raises(ManifestError, match="split directory"):
        discover_image_folder(tmp_path)

    (tmp_path / "root-file.png").unlink()
    write_file(tmp_path / "train" / "not-a-class.png")
    with pytest.raises(ManifestError, match="class directories"):
        discover_image_folder(tmp_path)


def test_parse_csv_uses_manifest_parent_and_normalizes_paths(tmp_path: Path) -> None:
    manifest = tmp_path / "input.csv"
    manifest.write_text(
        "path,split,label\n"
        ".\\images\\cat.png,train,cat\n"
        "images/nested/../dog.png,test,dog\n"
        "images/unlabeled.png,custom,\n",
        encoding="utf-8",
    )

    result = parse_csv_manifest(manifest)

    assert result.mode is ManifestMode.CSV
    assert result.root == tmp_path
    assert [entry.path for entry in result.entries] == [
        "images/cat.png",
        "images/dog.png",
        "images/unlabeled.png",
    ]
    assert [entry.label for entry in result.entries] == ["cat", "dog", None]
    assert [entry.split for entry in result.entries] == [
        Split.TRAIN,
        Split.TEST,
        Split.CUSTOM,
    ]


def test_parse_csv_accepts_explicit_dataset_root_without_touching_images(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    manifest = manifest_dir / "input.csv"
    manifest.write_text("path,split,label\nmissing/a.jpg,val,cat\n", encoding="utf-8")
    dataset_root = tmp_path / "dataset-that-does-not-exist"

    result = parse_csv_manifest(manifest, dataset_root=dataset_root)

    assert result.root == dataset_root
    assert result.entries[0].path == "missing/a.jpg"
    assert result.entries[0].split is Split.VAL


@pytest.mark.parametrize(
    "csv_text,match",
    [
        ("path,split\na.png,train\n", "missing required headers: label"),
        ("path,split,label\na.png,validation,cat\n", "unsupported split"),
        ("path,path,split,label\na.png,a.png,train,cat\n", "duplicate headers"),
    ],
)
def test_parse_csv_rejects_invalid_schema(
    tmp_path: Path, csv_text: str, match: str
) -> None:
    manifest = tmp_path / "input.csv"
    manifest.write_text(csv_text, encoding="utf-8")

    with pytest.raises(ManifestError, match=match):
        parse_csv_manifest(manifest)


def test_duplicate_logical_paths_are_rejected_after_normalization(tmp_path: Path) -> None:
    manifest = tmp_path / "input.csv"
    manifest.write_text(
        "path,split,label\n"
        "images/nested/../A.png,train,cat\n"
        "images/a.png,test,cat\n",
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="duplicate logical path"):
        parse_csv_manifest(manifest)


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../outside.png",
        "images/../../outside.png",
        "/absolute/image.png",
        "C:\\private\\image.png",
        "\\\\server\\share\\image.png",
    ],
)
def test_csv_rejects_path_escape_and_absolute_paths(
    tmp_path: Path, unsafe_path: str
) -> None:
    manifest = tmp_path / "input.csv"
    manifest.write_text(
        f"path,split,label\n{unsafe_path},train,cat\n",
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match=r"path escapes|must be relative"):
        parse_csv_manifest(manifest)


def test_normalization_is_unicode_stable_and_rejects_empty_targets() -> None:
    assert normalize_manifest_path("cafe\u0301/./image.png") == "caf\u00e9/image.png"
    with pytest.raises(ManifestError, match="identify a file"):
        normalize_manifest_path("images/..")


def test_load_manifest_dispatches_and_rejects_root_for_image_folder(tmp_path: Path) -> None:
    create_image_folder(tmp_path / "dataset")
    csv_path = tmp_path / "manifest.csv"
    csv_path.write_text("path,split,label\na.png,train,cat\n", encoding="utf-8")

    assert load_manifest(csv_path).mode is ManifestMode.CSV
    assert load_manifest(tmp_path / "dataset").mode is ManifestMode.IMAGE_FOLDER
    with pytest.raises(ManifestError, match="only valid"):
        load_manifest(tmp_path / "dataset", dataset_root=tmp_path)
