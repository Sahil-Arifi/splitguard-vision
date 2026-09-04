from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps
from pydantic import ValidationError
from scipy.fft import dctn

from splitguard.hashing import (
    PHASH_ALGORITHM_ID,
    BKTree,
    ExactDuplicateCluster,
    PhashCandidatePair,
    brute_force_phash_pairs,
    compute_phash,
    fingerprint_records,
    group_exact_duplicates,
    hamming_distance,
    indexed_phash_pairs,
)
from splitguard.schemas import ImageRecord, Split, SplitBoundary, stable_id


def record(
    path: str,
    *,
    sha: str,
    split: Split = Split.TRAIN,
    label: str | None = "cat",
    byte_size: int = 12,
    phash: int | None = None,
) -> ImageRecord:
    return ImageRecord(
        id=stable_id("img", path),
        path=path,
        split=split,
        label=label,
        byte_sha256=sha,
        byte_size=byte_size,
        width=8,
        height=8,
        format="png",
        phash=phash,
    )


def reference_phash_for_opaque_image(image: Image.Image) -> int:
    """Straightforward reference implementation for the public v1 contract."""

    grayscale = ImageOps.exif_transpose(image).convert("RGB").convert("L")
    resized = grayscale.resize((32, 32), resample=Image.Resampling.LANCZOS)
    coefficients = dctn(
        np.asarray(resized, dtype=np.float64),
        type=2,
        axes=(0, 1),
        norm="ortho",
    )
    low_frequency = coefficients[:8, :8].reshape(-1)
    bits = low_frequency > np.median(low_frequency[1:])
    bits[0] = False
    return int.from_bytes(np.packbits(bits, bitorder="big").tobytes(), byteorder="big")


def test_group_exact_duplicates_ignores_unique_images() -> None:
    duplicate_sha = "a" * 64
    unique_sha = "b" * 64
    first = record("train/cat/a.png", sha=duplicate_sha)
    second = record("train/cat/b.png", sha=duplicate_sha)
    unique = record("train/cat/unique.png", sha=unique_sha)

    clusters = group_exact_duplicates((unique, second, first))

    assert len(clusters) == 1
    assert clusters[0].byte_sha256 == duplicate_sha
    assert clusters[0].member_ids == tuple(sorted((first.id, second.id)))
    assert unique.id not in clusters[0].member_ids


def test_exact_cluster_identifiers_and_order_are_deterministic() -> None:
    low_sha = "1" * 64
    high_sha = "f" * 64
    records = (
        record("train/cat/high-b.png", sha=high_sha),
        record("train/cat/low-b.png", sha=low_sha),
        record("train/cat/high-a.png", sha=high_sha),
        record("train/cat/low-a.png", sha=low_sha),
    )

    forward = group_exact_duplicates(records)
    reverse = group_exact_duplicates(reversed(records))

    assert forward == reverse
    assert tuple(cluster.cluster_id for cluster in forward) == (
        f"exact_{low_sha}",
        f"exact_{high_sha}",
    )
    assert all(cluster.member_ids == tuple(sorted(cluster.member_ids)) for cluster in forward)


def test_exact_group_reports_cross_split_and_cross_label_metadata() -> None:
    sha = "c" * 64
    records = (
        record("test/dog/c.png", sha=sha, split=Split.TEST, label="dog", byte_size=20),
        record("train/cat/a.png", sha=sha, split=Split.TRAIN, label="cat", byte_size=20),
        record("val/cat/b.png", sha=sha, split=Split.VAL, label="cat", byte_size=20),
        record("custom/unlabeled/d.png", sha=sha, split=Split.CUSTOM, label=None, byte_size=20),
    )

    (cluster,) = group_exact_duplicates(records)

    assert cluster.member_count == 4
    assert cluster.splits == (Split.TRAIN, Split.VAL, Split.TEST, Split.CUSTOM)
    assert cluster.labels == ("cat", "dog")
    assert cluster.boundaries == (
        SplitBoundary(left=Split.TRAIN, right=Split.VAL),
        SplitBoundary(left=Split.TRAIN, right=Split.TEST),
        SplitBoundary(left=Split.VAL, right=Split.TEST),
    )
    assert cluster.crosses_evaluation_boundary is True
    assert cluster.total_duplicate_bytes == 80


@pytest.mark.parametrize(
    "first_split,second_split",
    [
        (Split.TRAIN, Split.TRAIN),
        (Split.TRAIN, Split.CUSTOM),
        (Split.VAL, Split.CUSTOM),
        (Split.TEST, Split.CUSTOM),
    ],
)
def test_custom_or_same_split_duplicates_do_not_cross_an_evaluation_boundary(
    first_split: Split,
    second_split: Split,
) -> None:
    sha = "d" * 64
    records = (
        record("first/cat/a.png", sha=sha, split=first_split),
        record("second/cat/b.png", sha=sha, split=second_split),
    )

    (cluster,) = group_exact_duplicates(records)

    assert cluster.splits == tuple(
        split for split in Split if split in {first_split, second_split}
    )
    assert cluster.boundaries == ()
    assert cluster.crosses_evaluation_boundary is False


def test_exact_duplicate_cluster_is_strict_and_immutable() -> None:
    sha = "e" * 64
    members = tuple(
        sorted((stable_id("img", "train/a.png"), stable_id("img", "train/b.png")))
    )
    cluster = ExactDuplicateCluster(
        cluster_id=f"exact_{sha}",
        byte_sha256=sha,
        member_ids=members,
        member_count=2,
        splits=(Split.TRAIN,),
        total_duplicate_bytes=24,
        crosses_evaluation_boundary=False,
    )

    with pytest.raises(ValidationError, match="frozen"):
        cluster.member_count = 3
    with pytest.raises(ValidationError):
        ExactDuplicateCluster.model_validate(
            {
                **cluster.model_dump(),
                "member_count": "2",
            }
        )


def test_compute_phash_has_known_tie_behavior_and_is_deterministic() -> None:
    black = Image.new("RGB", (41, 19), color=(0, 0, 0))

    first = compute_phash(black)
    second = compute_phash(black)

    # Every AC coefficient equals its median. The strict > tie rule therefore
    # clears all AC bits, while the ignored DC bit is fixed to zero as well.
    assert first == 0
    assert second == first
    assert 0 <= first < 1 << 64
    assert first & (1 << 63) == 0


def test_compute_phash_matches_reference_and_published_algorithm_id() -> None:
    rows, columns = np.indices((27, 35))
    pixels = np.stack(
        (
            (columns * 17 + rows * 3) % 256,
            (columns * 5 + rows * 29) % 256,
            (columns * 11 + rows * 7 + 43) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    image = Image.fromarray(pixels)

    assert PHASH_ALGORITHM_ID == "phash64-dct-v1"
    assert compute_phash(image) == 0x52851C952B953F6A
    assert compute_phash(image) == reference_phash_for_opaque_image(image)


def test_compute_phash_is_invariant_to_brightness_offset_and_grayscale_mode() -> None:
    pixels = ((np.arange(32 * 32, dtype=np.uint16) * 37) % 150).astype(np.uint8)
    pixels = pixels.reshape((32, 32))
    grayscale = Image.fromarray(pixels)
    rgb = Image.merge("RGB", (grayscale, grayscale, grayscale))
    brighter = Image.fromarray((pixels.astype(np.uint16) + 40).astype(np.uint8))

    assert compute_phash(grayscale) == compute_phash(rgb)
    assert compute_phash(grayscale) == compute_phash(brighter)


def test_compute_phash_applies_exif_orientation_before_fingerprinting() -> None:
    pixels = np.zeros((18, 30, 3), dtype=np.uint8)
    pixels[:6, :10] = (240, 10, 30)
    pixels[9:, 17:] = (20, 190, 70)
    displayed = Image.fromarray(pixels)
    stored = displayed.transpose(Image.Transpose.ROTATE_180)
    exif = stored.getexif()
    exif[274] = 3
    stored.info["exif"] = exif.tobytes()

    assert compute_phash(stored) == compute_phash(displayed)


def test_compute_phash_composites_alpha_over_fixed_white_background() -> None:
    first = np.zeros((24, 24, 4), dtype=np.uint8)
    second = np.zeros((24, 24, 4), dtype=np.uint8)
    first[..., :3] = (255, 0, 0)
    second[..., :3] = (0, 0, 255)
    first[4:13, 7:18] = (10, 80, 30, 255)
    second[4:13, 7:18] = (10, 80, 30, 255)

    assert compute_phash(Image.fromarray(first)) == compute_phash(Image.fromarray(second))


def test_compute_phash_does_not_mutate_input_image() -> None:
    pixels = np.arange(20 * 16 * 4, dtype=np.uint16).reshape((20, 16, 4)) % 256
    image = Image.fromarray(pixels.astype(np.uint8))
    exif = image.getexif()
    exif[274] = 3
    image.info["exif"] = exif.tobytes()
    before = (image.mode, image.size, image.tobytes(), image.getexif().tobytes())

    compute_phash(image)

    assert (image.mode, image.size, image.tobytes(), image.getexif().tobytes()) == before


def test_compute_phash_rejects_non_image_input() -> None:
    with pytest.raises(TypeError, match="Pillow Image"):
        compute_phash(np.zeros((32, 32), dtype=np.uint8))  # type: ignore[arg-type]


def test_hamming_distance_validates_uint64_and_counts_bits() -> None:
    assert hamming_distance(0, 0) == 0
    assert hamming_distance(0, (1 << 64) - 1) == 64
    assert hamming_distance(0b1010, 0b0011) == 2
    assert hamming_distance(123, 456) == hamming_distance(456, 123)

    for invalid in (-1, 1 << 64):
        with pytest.raises(ValueError, match="unsigned 64-bit"):
            hamming_distance(invalid, 0)
    for non_integer in (True, 1.5, "1"):
        with pytest.raises(TypeError, match="integer"):
            hamming_distance(non_integer, 0)  # type: ignore[arg-type]


def test_bk_tree_supports_duplicate_hashes_and_canonical_query_results() -> None:
    first = stable_id("img", "first")
    second = stable_id("img", "second")
    third = stable_id("img", "third")
    tree = BKTree()

    assert tree.search(0, radius=0) == ()
    tree.add(0b1010, second)
    tree.add(0b1010, first)
    tree.add(0b1011, third)

    matches = tree.search(0b1010, radius=1)

    assert len(tree) == 3
    assert [(match.distance, match.record_id) for match in matches] == [
        (0, min(first, second)),
        (0, max(first, second)),
        (1, third),
    ]
    assert [match.phash for match in matches] == [0b1010, 0b1010, 0b1011]

    with pytest.raises(ValueError, match="already indexed"):
        tree.add(0b1010, first)
    with pytest.raises(ValueError, match="already indexed"):
        tree.add(0b1111, first)


def test_bk_tree_bulk_construction_is_permutation_independent_and_excludes_id() -> None:
    items = [
        (0, stable_id("img", "zero")),
        (1, stable_id("img", "one")),
        (3, stable_id("img", "three")),
        (3, stable_id("img", "also-three")),
        ((1 << 64) - 1, stable_id("img", "maximum")),
    ]
    baseline = BKTree.from_items(items)
    expected = baseline.search(3, radius=8)

    rng = random.Random(412)
    for _ in range(8):
        permuted = list(items)
        rng.shuffle(permuted)
        tree = BKTree.from_items(permuted)
        assert len(tree) == len(items)
        assert tree.search(3, radius=8) == expected

    excluded_id = stable_id("img", "three")
    excluded = baseline.search(3, radius=0, exclude_id=excluded_id)
    assert [match.record_id for match in excluded] == [stable_id("img", "also-three")]

    with pytest.raises(ValueError, match="duplicated"):
        BKTree.from_items((*items, items[0]))


def test_bk_tree_search_includes_triangle_boundaries() -> None:
    root_id = stable_id("img", "root")
    child_id = stable_id("img", "child")
    tree = BKTree()
    tree.add(0, root_id)
    tree.add((1 << 9) - 1, child_id)

    matches = tree.search(1 << 8, radius=8)

    assert {match.record_id for match in matches} == {root_id, child_id}
    assert next(match for match in matches if match.record_id == child_id).distance == 8


@pytest.mark.parametrize("radius", [-1, 65])
def test_bk_tree_rejects_out_of_range_radius(radius: int) -> None:
    with pytest.raises(ValueError, match="between 0 and 64"):
        BKTree().search(0, radius)


def test_bk_tree_rejects_non_integer_radius_and_noncanonical_id() -> None:
    with pytest.raises(TypeError, match="radius must be an integer"):
        BKTree().search(0, True)
    with pytest.raises(ValueError, match="canonical stable identifier"):
        BKTree().add(0, "not-canonical")


def test_indexed_phash_pairs_equal_brute_force_on_controlled_hashes() -> None:
    rng = random.Random(20260903)
    hashes = [rng.getrandbits(64) for _ in range(24)]
    hashes.extend((hashes[0], hashes[0] ^ 1, hashes[1] ^ 0b11, hashes[2] ^ 0b1111))
    records = tuple(
        record(
            f"train/cat/{index}.png",
            sha=f"{index:064x}",
            phash=phash,
        )
        for index, phash in enumerate(hashes)
    )

    for radius in (0, 1, 8, 63, 64):
        expected = brute_force_phash_pairs(records, radius)
        assert indexed_phash_pairs(records, radius) == expected
        for _ in range(3):
            permuted = list(records)
            rng.shuffle(permuted)
            assert indexed_phash_pairs(permuted, radius) == expected


def test_indexed_pairs_are_canonical_unique_and_exclude_self() -> None:
    shared_hash = 0x0123456789ABCDEF
    hashed_records = (
        record("train/cat/c.png", sha="3" * 64, phash=shared_hash),
        record("train/cat/a.png", sha="1" * 64, phash=shared_hash),
        record("train/cat/b.png", sha="2" * 64, phash=shared_hash),
        record("train/cat/unhashed.png", sha="4" * 64),
    )

    forward = indexed_phash_pairs(hashed_records, radius=0)
    reverse = indexed_phash_pairs(reversed(hashed_records), radius=0)

    assert forward == reverse
    assert len(forward) == 3
    assert all(pair.left_id < pair.right_id for pair in forward)
    assert all(pair.left_id != pair.right_id for pair in forward)
    assert forward == tuple(
        sorted(forward, key=lambda pair: (pair.left_id, pair.right_id, pair.distance))
    )
    assert all(pair.distance == 0 for pair in forward)
    assert len({(pair.left_id, pair.right_id) for pair in forward}) == len(forward)


def test_phash_candidate_pair_rejects_noncanonical_member_order() -> None:
    left, right = sorted((stable_id("img", "left"), stable_id("img", "right")))
    with pytest.raises(ValidationError, match="left_id < right_id"):
        PhashCandidatePair(left_id=right, right_id=left, distance=0)


def test_pair_search_rejects_duplicate_record_ids() -> None:
    duplicate = record("train/cat/a.png", sha="a" * 64, phash=123)

    with pytest.raises(ValueError, match="record_id is duplicated"):
        indexed_phash_pairs((duplicate, duplicate), radius=0)
    with pytest.raises(ValueError, match="record_id is duplicated"):
        brute_force_phash_pairs((duplicate, duplicate), radius=0)


def test_fingerprint_records_attaches_hashes_without_changing_source(tmp_path: Path) -> None:
    image_path = tmp_path / "train" / "cat" / "sample.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (12, 10), (25, 100, 220)).save(image_path)
    original_bytes = image_path.read_bytes()
    source = record(
        "train/cat/sample.png",
        sha=hashlib.sha256(original_bytes).hexdigest(),
        byte_size=len(original_bytes),
    )

    (fingerprinted,) = fingerprint_records(tmp_path, (source,))

    assert fingerprinted.phash is not None
    assert fingerprinted.byte_sha256 == source.byte_sha256
    assert source.phash is None
    assert image_path.read_bytes() == original_bytes


def test_fingerprint_records_rejects_changed_content_without_path_leakage(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "train" / "cat" / "sample.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), "red").save(image_path)
    source = record(
        "train/cat/sample.png",
        sha=hashlib.sha256(image_path.read_bytes()).hexdigest(),
    )
    Image.new("RGB", (8, 8), "blue").save(image_path)

    with pytest.raises(RuntimeError, match="content changed") as error:
        fingerprint_records(tmp_path, (source,))

    assert str(tmp_path) not in str(error.value)
