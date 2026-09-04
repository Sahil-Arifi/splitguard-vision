"""CLI bootstrap tests."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from splitguard.cli import app

runner = CliRunner()


def test_root_help_is_useful() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Audit and repair image dataset split leakage" in result.stdout
    assert "version" in result.stdout


def test_version_supports_short_output() -> None:
    result = runner.invoke(app, ["version", "--short"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"


def test_scan_reports_exact_duplicates_and_invalid_images(tmp_path: Path) -> None:
    train = tmp_path / "train" / "cat"
    test = tmp_path / "test" / "cat"
    train.mkdir(parents=True)
    test.mkdir(parents=True)
    image_path = train / "source.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image_path)
    (test / "copy.png").write_bytes(image_path.read_bytes())
    (test / "broken.png").write_bytes(b"not an image")

    result = runner.invoke(app, ["scan", str(tmp_path)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["mode"] == "image_folder"
    assert payload["manifest_entries"] == 3
    assert payload["valid_images"] == 2
    assert payload["invalid_images"] == 1
    assert payload["exact_duplicate_groups"] == 1
    assert payload["exact_cross_split_groups"] == 1
    assert payload["invalid_records"][0]["path"] == "test/cat/broken.png"
    assert str(tmp_path) not in result.stdout


def test_scan_reports_manifest_errors_without_traceback(tmp_path: Path) -> None:
    result = runner.invoke(app, ["scan", str(tmp_path / "missing")])

    assert result.exit_code == 2
    assert "dataset root does not exist" in result.output
    assert "Traceback" not in result.output


def test_audit_writes_private_path_free_artifact_without_network(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    train = dataset / "train" / "cat"
    test = dataset / "test" / "dog"
    train.mkdir(parents=True)
    test.mkdir(parents=True)
    source = train / "source.png"
    Image.new("RGB", (12, 12), (40, 80, 120)).save(source)
    (test / "copy.png").write_bytes(source.read_bytes())
    (test / "broken.png").write_bytes(b"broken")
    output = tmp_path / "results" / "audit.json"

    result = runner.invoke(
        app,
        [
            "audit",
            str(dataset),
            "--config",
            str(Path(__file__).parents[1] / "configs" / "default.yaml"),
            "--output",
            str(output),
            "--no-embeddings",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "audit"
    assert payload["summary"]["valid_image_count"] == 2
    assert payload["summary"]["invalid_image_count"] == 1
    assert payload["summary"]["leakage_group_count"] == 1
    assert payload["summary"]["exact_leakage_group_count"] == 1
    assert payload["summary"]["cross_label_conflict_count"] == 1
    assert str(tmp_path) not in output.read_text(encoding="utf-8")
    assert "audit.json" in result.stdout
