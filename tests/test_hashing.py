from __future__ import annotations

import pytest
from pydantic import ValidationError

from splitguard.hashing import ExactDuplicateCluster, group_exact_duplicates
from splitguard.schemas import ImageRecord, Split, SplitBoundary, stable_id


def record(
    path: str,
    *,
    sha: str,
    split: Split = Split.TRAIN,
    label: str | None = "cat",
    byte_size: int = 12,
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
    )


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
        cluster.member_count = 3  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ExactDuplicateCluster.model_validate(
            {
                **cluster.model_dump(),
                "member_count": "2",
            }
        )
