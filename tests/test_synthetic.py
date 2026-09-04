from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

import splitguard.synthetic as synthetic_module
from splitguard.schemas import ImageRecord, Split, stable_id
from splitguard.synthetic import (
    CorruptionType,
    ExpectedRelationship,
    InjectedDefect,
    SyntheticCorruptionSet,
    SyntheticInputError,
    generate_controlled_corruptions,
)


def make_source(root: Path, name: str, split: Split, label: str) -> ImageRecord:
    relative = f"{split.value}/{label}/{name}.png"
    path = root / split.value / label / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (48, 36))
    pixels = image.load()
    assert pixels is not None
    name_offset = sum(name.encode("utf-8")) % 256
    for y in range(image.height):
        for x in range(image.width):
            pixels[x, y] = (
                (x * 7 + y * 3 + name_offset) % 256,
                (x * 2 + y * 11 + name_offset * 3) % 256,
                (x * 13 + y * 5 + name_offset * 5) % 256,
            )
    image.save(path, format="PNG", compress_level=6)
    image.close()
    payload = path.read_bytes()
    return ImageRecord(
        id=stable_id("img", relative),
        path=relative,
        split=split,
        label=label,
        byte_sha256=hashlib.sha256(payload).hexdigest(),
        byte_size=len(payload),
        width=48,
        height=36,
        format="png",
    )


def test_generates_every_declared_corruption_with_independent_truth(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source = make_source(source_root, "pattern", Split.TRAIN, "cat")
    original_payload = (source_root / source.path).read_bytes()
    output = tmp_path / "generated"

    result = generate_controlled_corruptions(
        source_root,
        (source,),
        output,
        seed=17,
    )

    assert tuple(item.corruption_type for item in result.injections) == tuple(CorruptionType)
    assert len(result.derived_records) == len(CorruptionType) - 1
    assert len(result.invalid_derived_ids) == 1
    assert result.negative_controls == ()
    assert (source_root / source.path).read_bytes() == original_payload

    by_type = {item.corruption_type: item for item in result.injections}
    for corruption in (
        CorruptionType.EXACT_COPY,
        CorruptionType.CROSS_SPLIT_COPY,
        CorruptionType.CROSS_LABEL_DUPLICATE,
    ):
        injection = by_type[corruption]
        assert injection.expected_relationship is ExpectedRelationship.EXACT_DUPLICATE
        assert (output / injection.derived_path).read_bytes() == original_payload
        assert injection.exact_bytes is True

    for corruption in (
        CorruptionType.JPEG_RECOMPRESSION,
        CorruptionType.RESIZE,
        CorruptionType.SMALL_CROP,
        CorruptionType.BRIGHTNESS_SHIFT,
        CorruptionType.GAUSSIAN_BLUR,
    ):
        injection = by_type[corruption]
        assert injection.expected_relationship is ExpectedRelationship.TRANSFORMED_DUPLICATE
        assert (output / injection.derived_path).read_bytes() != original_payload
        with Image.open(output / injection.derived_path) as transformed:
            transformed.load()

    cross_split = by_type[CorruptionType.CROSS_SPLIT_COPY]
    assert (cross_split.source_split, cross_split.target_split) == (Split.TRAIN, Split.VAL)
    cross_label = by_type[CorruptionType.CROSS_LABEL_DUPLICATE]
    assert cross_label.source_label != cross_label.target_label

    malformed = by_type[CorruptionType.MALFORMED_FILE]
    assert malformed.expected_relationship is ExpectedRelationship.INVALID_IMAGE
    assert malformed.is_duplicate is False
    with pytest.raises(UnidentifiedImageError):
        Image.open(output / malformed.derived_path)


def test_outputs_and_metadata_are_reproducible_across_locations_and_input_order(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    first = make_source(source_root, "first", Split.TRAIN, "cat")
    second = make_source(source_root, "second", Split.TEST, "dog")
    selected = (
        CorruptionType.JPEG_RECOMPRESSION,
        CorruptionType.BRIGHTNESS_SHIFT,
        CorruptionType.CROSS_SPLIT_COPY,
    )

    forward = generate_controlled_corruptions(
        source_root,
        (first, second),
        tmp_path / "one",
        seed=91,
        corruption_types=selected,
    )
    reverse = generate_controlled_corruptions(
        source_root,
        (second, first),
        tmp_path / "two",
        seed=91,
        corruption_types=reversed(selected),
    )

    assert forward == reverse
    assert len(forward.negative_controls) == len(selected) * 2
    injections_by_id = {item.derived_id: item for item in forward.injections}
    for control in forward.negative_controls:
        compared_injection = injections_by_id[control.unrelated_derived_id]
        assert control.expected_relationship is ExpectedRelationship.UNRELATED
        assert control.is_duplicate is False
        assert control.source_id != compared_injection.source_id
        assert control.corruption_type is compared_injection.corruption_type
    for injection in forward.injections:
        assert (tmp_path / "one" / injection.derived_path).read_bytes() == (
            tmp_path / "two" / injection.derived_path
        ).read_bytes()
    assert all("\\" not in item.derived_path for item in forward.injections)
    assert all(not Path(item.derived_path).is_absolute() for item in forward.injections)

    forged = forward.model_dump()
    control = forged["negative_controls"][0]
    compared = next(
        item
        for item in forward.injections
        if item.derived_id == control["unrelated_derived_id"]
    )
    control["source_id"] = compared.source_id
    forged["negative_controls"] = (control,)
    with pytest.raises(ValidationError, match="negative control"):
        SyntheticCorruptionSet.model_validate(forged)


def test_changed_source_hash_fails_without_installing_partial_output(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source = make_source(source_root, "pattern", Split.TRAIN, "cat")
    changed = source.model_copy(update={"byte_sha256": "0" * 64})
    output = tmp_path / "generated"

    with pytest.raises(SyntheticInputError, match="content hash changed"):
        generate_controlled_corruptions(source_root, (changed,), output)

    assert not output.exists()
    assert not tuple(tmp_path.glob(".splitguard-synthetic-*"))


@pytest.mark.parametrize("output", ["same", "child", "parent"])
def test_source_and_output_roots_must_be_disjoint(tmp_path: Path, output: str) -> None:
    source_root = tmp_path / "dataset"
    source = make_source(source_root, "pattern", Split.TRAIN, "cat")
    destinations = {
        "same": source_root,
        "child": source_root / "generated",
        "parent": tmp_path,
    }

    with pytest.raises(SyntheticInputError, match="must not overlap"):
        generate_controlled_corruptions(source_root, (source,), destinations[output])


def test_existing_destination_and_duplicate_inputs_are_rejected(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source = make_source(source_root, "pattern", Split.TRAIN, "cat")
    output = tmp_path / "generated"
    output.mkdir()

    with pytest.raises(SyntheticInputError, match="must not already exist"):
        generate_controlled_corruptions(source_root, (source,), output)
    with pytest.raises(SyntheticInputError, match="unique IDs"):
        generate_controlled_corruptions(
            source_root,
            (source, source),
            tmp_path / "duplicates",
        )
    with pytest.raises(SyntheticInputError, match="cannot contain duplicates"):
        generate_controlled_corruptions(
            source_root,
            (source,),
            tmp_path / "corruption-duplicates",
            corruption_types=(CorruptionType.RESIZE, CorruptionType.RESIZE),
        )


def test_byte_identical_sources_are_rejected_before_negative_controls(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source = make_source(source_root, "pattern", Split.TRAIN, "cat")
    duplicate_path = "test/dog/copy.png"
    duplicate = source.model_copy(
        update={
            "id": stable_id("img", duplicate_path),
            "path": duplicate_path,
            "split": Split.TEST,
            "label": "dog",
        }
    )
    output = tmp_path / "generated"

    with pytest.raises(SyntheticInputError, match="byte-distinct"):
        generate_controlled_corruptions(source_root, (source, duplicate), output)

    assert not output.exists()


def test_differently_encoded_but_decoded_identical_sources_are_rejected(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    first = make_source(source_root, "first", Split.TRAIN, "cat")
    second_relative = "test/dog/second.png"
    second_path = source_root / second_relative
    second_path.parent.mkdir(parents=True)
    with Image.open(source_root / first.path) as image:
        image.save(second_path, format="PNG", compress_level=9)
    second_payload = second_path.read_bytes()
    assert second_payload != (source_root / first.path).read_bytes()
    second = ImageRecord(
        id=stable_id("img", second_relative),
        path=second_relative,
        split=Split.TEST,
        label="dog",
        byte_sha256=hashlib.sha256(second_payload).hexdigest(),
        byte_size=len(second_payload),
        width=first.width,
        height=first.height,
        format="png",
    )
    output = tmp_path / "generated"

    with pytest.raises(SyntheticInputError, match="decoded-distinct"):
        generate_controlled_corruptions(source_root, (first, second), output)

    assert not output.exists()


def test_byte_identical_transformation_retries_then_fails_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source = make_source(source_root, "pattern", Split.TRAIN, "cat")
    source_payload = (source_root / source.path).read_bytes()
    output = tmp_path / "generated"

    monkeypatch.setattr(
        synthetic_module,
        "_encode_transformation",
        lambda *_args: (source_payload, "png"),
    )

    with pytest.raises(SyntheticInputError, match="byte-distinct"):
        generate_controlled_corruptions(
            source_root,
            (source,),
            output,
            corruption_types=(CorruptionType.RESIZE,),
        )

    assert not output.exists()
    assert not tuple(tmp_path.glob(".splitguard-synthetic-*"))


def test_injected_defect_contract_rejects_detector_relabeling(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source = make_source(source_root, "pattern", Split.TRAIN, "cat")
    result = generate_controlled_corruptions(
        source_root,
        (source,),
        tmp_path / "generated",
        corruption_types=(CorruptionType.RESIZE,),
    )
    payload = result.injections[0].model_dump()
    payload["expected_relationship"] = ExpectedRelationship.EXACT_DUPLICATE

    with pytest.raises(ValidationError, match="ground truth"):
        InjectedDefect.model_validate(payload)


def test_empty_source_set_produces_empty_installed_dataset(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    output = tmp_path / "generated"

    result = generate_controlled_corruptions(source_root, (), output, seed=4)

    assert output.is_dir()
    assert result.source_ids == ()
    assert result.derived_records == ()
    assert result.injections == ()
    assert result.negative_controls == ()
