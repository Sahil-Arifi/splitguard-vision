"""Offline fixtures test protocol enforcement, not real-world detector performance."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from splitguard.cli import app
from splitguard.labeled_pairs import (
    LabeledPairError,
    PairObservation,
    evaluate_labeled_pairs,
    score_pairs,
    select_radius,
)


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["left", "right", "label", "left_group", "right_group"])
        writer.writerows(rows)


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[Path, Path, Path]:
    images = tmp_path / "images"
    images.mkdir()
    for index, name in enumerate(("a", "b", "c", "d")):
        pixels = np.random.default_rng(index).integers(0, 256, (48, 48, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(images / f"{name}.png")
        (images / f"{name}-copy.png").write_bytes((images / f"{name}.png").read_bytes())
    calibration, evaluation = tmp_path / "calibration.csv", tmp_path / "evaluation.csv"
    _write_csv(
        calibration,
        [
            ["a.png", "a-copy.png", "1", "source-a", "source-a"],
            ["a.png", "b.png", "0", "source-a", "source-b"],
        ],
    )
    _write_csv(
        evaluation,
        [
            ["c.png", "c-copy.png", "1", "source-c", "source-c"],
            ["c.png", "d.png", "0", "source-c", "source-d"],
        ],
    )
    return calibration, evaluation, images


def _run(corpus: tuple[Path, Path, Path]) -> object:
    return evaluate_labeled_pairs(*corpus, label_source="Synthetic unit-test fixtures only")


def test_frozen_threshold_does_not_optimize_held_out_labels() -> None:
    calibration = (
        PairObservation("a", "b", True, "a", "a", False, 2),
        PairObservation("c", "d", False, "c", "d", False, 5),
    )
    selected, sweep = select_radius(calibration, 1.0)
    assert selected.radius == 2
    assert len(sweep) == 66
    evaluation = (
        PairObservation("e", "f", True, "e", "e", False, 4),
        PairObservation("g", "h", False, "g", "h", False, 1),
    )
    result = score_pairs(evaluation, selected.radius)
    assert result.metrics.false_positives == 1
    assert result.metrics.false_negatives == 1
    assert result.metrics.f1 == 0.0
    assert result.true_negatives == 0


def test_no_threshold_meeting_precision_fails() -> None:
    pairs = (
        PairObservation("a", "b", True, "a", "a", False, 8),
        PairObservation("c", "d", False, "c", "d", False, 2),
    )
    with pytest.raises(LabeledPairError, match="no calibration threshold"):
        select_radius(pairs, 0.95)


@pytest.mark.parametrize("precision", [0.0, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_precision(precision: float) -> None:
    with pytest.raises(LabeledPairError, match="minimum precision"):
        select_radius((), precision)


@pytest.mark.parametrize("radius", [-2, 65, True])
def test_invalid_radius(radius: int) -> None:
    with pytest.raises(LabeledPairError, match="radius"):
        score_pairs((), radius)


def test_independent_external_corpus_has_hashed_provenance(
    corpus: tuple[Path, Path, Path],
) -> None:
    artifact = evaluate_labeled_pairs(*corpus, label_source="Synthetic test fixtures")
    assert artifact.selected_radius == -1  # prefer SHA-only when it gives the same quality
    assert artifact.calibration.metrics.true_positives == 1
    assert artifact.evaluation.metrics.true_positives == 1
    assert artifact.evaluation.true_negatives == 1
    assert artifact.calibration_csv_sha256 != artifact.evaluation_csv_sha256
    assert len(artifact.metadata.dataset_manifest_sha256) == 64
    serialized = artifact.model_dump_json()
    assert str(corpus[2]) not in serialized
    assert artifact.evaluation.pair_count == 2


@pytest.mark.parametrize("bad_path", ["../a.png", "/a.png", "a\\b.png", "C:/a.png", "./a.png"])
def test_rejects_unsafe_paths(corpus: tuple[Path, Path, Path], bad_path: str) -> None:
    calibration, _, _ = corpus
    _write_csv(calibration, [[bad_path, "a-copy.png", "1", "a", "a"]])
    with pytest.raises(LabeledPairError, match="relative POSIX"):
        _run(corpus)


@pytest.mark.parametrize(
    "bad_row,match",
    [
        (["a.png", "a-copy.png", "yes", "a", "a"], "label must"),
        (["a.png", "a-copy.png", "1", "a", "b"], "must share"),
        (["a.png", "b.png", "0", "a", "a"], "must share"),
        (["a.png", "b.png", "0", "a", ""], "five nonempty"),
        (["missing.png", "a.png", "1", "a", "a"], "existing file"),
        (["a.png", "a.png", "1", "a", "a"], "self-pairs"),
        (["a.png", "a-copy.png", "0", "a", "b"], "identical image content"),
    ],
)
def test_invalid_rows(corpus: tuple[Path, Path, Path], bad_row: list[str], match: str) -> None:
    _write_csv(corpus[0], [bad_row])
    with pytest.raises(LabeledPairError, match=match):
        _run(corpus)


def test_reversed_duplicate_pair_rejected(corpus: tuple[Path, Path, Path]) -> None:
    with corpus[0].open("a", encoding="utf-8") as handle:
        handle.write("a-copy.png,a.png,1,source-a,source-a\n")
    with pytest.raises(LabeledPairError, match="repeated or reversed"):
        _run(corpus)


def test_inconsistent_group_for_same_image_rejected(corpus: tuple[Path, Path, Path]) -> None:
    corpus[0].write_text(corpus[0].read_text().replace("0,source-a", "0,new-a"))
    with pytest.raises(LabeledPairError, match="multiple source groups"):
        _run(corpus)


@pytest.mark.parametrize("mode", ["groups", "bytes", "pixels", "path", "alpha"])
def test_rejects_fold_contamination(corpus: tuple[Path, Path, Path], mode: str) -> None:
    _, evaluation, images = corpus
    if mode == "groups":
        evaluation.write_text(evaluation.read_text().replace("source-c", "source-a"))
    elif mode == "path":
        evaluation.write_text(
            evaluation.read_text().replace("c.png", "a.png").replace("c-copy.png", "a-copy.png")
        )
    else:
        if mode == "bytes":
            data = (images / "a.png").read_bytes()
            (images / "c.png").write_bytes(data)
        elif mode == "alpha":
            with Image.open(images / "a.png") as original:
                rgba = original.convert("RGBA")
            rgba.putalpha(128)
            rgba.save(images / "a.png")
            (images / "a-copy.png").write_bytes((images / "a.png").read_bytes())
            background = Image.new("RGBA", rgba.size, color=(255, 255, 255, 255))
            Image.alpha_composite(background, rgba).convert("RGB").save(images / "c.png")
        else:
            with Image.open(images / "a.png") as image:
                image.save(images / "c.png", compress_level=0)
            assert (images / "c.png").read_bytes() != (images / "a.png").read_bytes()
        (images / "c-copy.png").write_bytes((images / "c.png").read_bytes())
    with pytest.raises(LabeledPairError, match="overlap"):
        _run(corpus)


def test_bad_csv_image_and_missing_class_fail(corpus: tuple[Path, Path, Path]) -> None:
    calibration, _, images = corpus
    calibration.write_text("left,right,label\na,b,1\n")
    with pytest.raises(LabeledPairError, match="CSV header"):
        _run(corpus)
    _write_csv(calibration, [["a.png", "a-copy.png", "1", "a", "a"]])
    with pytest.raises(LabeledPairError, match="both duplicate and distinct"):
        _run(corpus)
    (images / "a.png").write_bytes(b"not an image")
    with pytest.raises(LabeledPairError, match="decode"):
        _run(corpus)


def test_pixel_limit_and_source_validation(corpus: tuple[Path, Path, Path]) -> None:
    with pytest.raises(LabeledPairError, match="decoded-pixel"):
        evaluate_labeled_pairs(*corpus, label_source="test", max_image_pixels=1)
    with pytest.raises(LabeledPairError, match="label source"):
        evaluate_labeled_pairs(*corpus, label_source=" ")
    with pytest.raises(LabeledPairError, match="must be positive"):
        evaluate_labeled_pairs(*corpus, label_source="test", max_image_pixels=0)


def test_cli_writes_held_out_json_and_preserves_inputs(corpus: tuple[Path, Path, Path]) -> None:
    calibration, evaluation, images = corpus
    output = calibration.parent / "results.json"
    before = {path: path.read_bytes() for path in images.iterdir()}
    args = [
        "evaluate-labeled-pairs",
        str(calibration),
        str(evaluation),
        "--dataset-root",
        str(images),
        "--label-source",
        "Synthetic test fixtures",
        "--output",
        str(output),
    ]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    artifact = json.loads(output.read_text())
    assert artifact["selected_radius"] == -1
    assert artifact["evaluation"]["metrics"]["precision"] == 1.0
    assert all(path.read_bytes() == content for path, content in before.items())
    for unsafe_output in (calibration, evaluation, images / "a.png"):
        result = CliRunner().invoke(app, [*args[:-1], str(unsafe_output)])
        assert result.exit_code != 0
        assert "output must be outside" in result.output
