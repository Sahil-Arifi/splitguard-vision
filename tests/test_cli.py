"""CLI bootstrap tests."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Literal

import pytest
from PIL import Image
from pytest import MonkeyPatch
from typer.testing import CliRunner

import splitguard.cli as cli_module
from splitguard.cli import app
from splitguard.config import load_config
from splitguard.repair import MaterializationResult
from splitguard.schemas import AuditArtifact, RepairArtifact, Split, canonical_sha256

runner = CliRunner()


def test_root_help_is_useful() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Audit and repair image dataset split leakage" in result.stdout
    assert "version" in result.stdout
    assert "repair" in result.stdout
    assert "materialize" in result.stdout


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


def create_audit_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset = tmp_path / "private-dataset"
    train = dataset / "train" / "cat"
    test = dataset / "test" / "dog"
    train.mkdir(parents=True)
    test.mkdir(parents=True)
    source = train / "source.png"
    Image.new("RGB", (10, 10), (15, 30, 45)).save(source)
    (test / "copy.png").write_bytes(source.read_bytes())
    (test / "broken.png").write_bytes(b"not an image")
    audit_path = tmp_path / "audit" / "audit.json"
    config_path = Path(__file__).parents[1] / "configs" / "default.yaml"

    result = runner.invoke(
        app,
        [
            "audit",
            str(dataset),
            "--config",
            str(config_path),
            "--output",
            str(audit_path),
            "--no-embeddings",
        ],
    )

    assert result.exit_code == 0, result.output
    return dataset, audit_path, config_path


def test_repair_writes_complete_artifact_and_manifest_with_provenance(
    tmp_path: Path,
) -> None:
    _, audit_path, config_path = create_audit_fixture(tmp_path)
    repair_path = tmp_path / "repair-output" / "repair.json"
    manifest_path = tmp_path / "repair-output" / "repaired_manifest.csv"

    result = runner.invoke(
        app,
        [
            "repair",
            str(audit_path),
            "--ratios",
            "0.8,0.1,0.1",
            "--config",
            str(config_path),
            "--output",
            str(repair_path),
        ],
    )

    assert result.exit_code == 0, result.output
    artifact = RepairArtifact.model_validate_json(repair_path.read_bytes())
    source_audit = AuditArtifact.model_validate_json(audit_path.read_bytes())
    config = load_config(config_path)
    expected_configuration_sha256 = canonical_sha256(
        {
            "base_configuration_sha256": config.config_hash,
            "repair": {
                "class_balance_weight": artifact.class_balance_weight,
                "local_improvement_iterations": artifact.local_improvement_iterations,
                "random_seed": artifact.random_seed,
                "split_size_weight": artifact.split_size_weight,
            },
            "requested_ratios": [
                {"ratio": item.ratio, "split": item.split.value}
                for item in artifact.requested_ratios
            ],
        }
    )
    manifest_bytes = manifest_path.read_bytes()
    rows = list(csv.DictReader(manifest_path.read_text(encoding="utf-8").splitlines()))
    repaired_splits = {row["split"] for row in rows}

    assert artifact.artifact_type == "repair"
    assert tuple(item.ratio for item in artifact.requested_ratios) == (0.8, 0.1, 0.1)
    assert artifact.integer_targets == (
        (Split.TRAIN, 2),
        (Split.VAL, 0),
        (Split.TEST, 0),
    )
    assert artifact.split_size_weight == config.repair.split_size_weight
    assert artifact.class_balance_weight == config.repair.class_balance_weight
    assert artifact.random_seed == config.repair.random_seed
    assert (
        artifact.local_improvement_iterations
        == config.repair.local_improvement_iterations
    )
    assert artifact.metadata.configuration_sha256 == expected_configuration_sha256
    assert (
        artifact.metadata.dataset_manifest_sha256
        == source_audit.metadata.dataset_manifest_sha256
    )
    assert artifact.metadata.random_seeds == (config.repair.random_seed,)
    assert artifact.repaired_manifest_sha256 == hashlib.sha256(manifest_bytes).hexdigest()
    assert artifact.summary.definite_leakage_groups_before == 1
    assert artifact.summary.definite_leakage_groups_after == 0
    assert artifact.summary.hard_group_invariant_satisfied is True
    assert len(rows) == 2
    assert len(repaired_splits) == 1
    assert artifact.excluded_invalid_ids

    success = json.loads(result.stdout)
    assert success["source_audit_sha256"] == canonical_sha256(source_audit)
    assert success["audit_configuration_sha256"] == source_audit.metadata.configuration_sha256
    assert success["configuration_sha256"] == expected_configuration_sha256
    assert success["repaired_manifest_sha256"] == artifact.repaired_manifest_sha256
    assert success["warnings"] == list(artifact.infeasibility_warnings)
    assert str(tmp_path) not in result.stdout
    assert str(tmp_path) not in repair_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in manifest_path.read_text(encoding="utf-8")


def test_repair_strictly_rejects_malformed_audit_without_partial_outputs(
    tmp_path: Path,
) -> None:
    audit_path = tmp_path / "private" / "malformed-audit.json"
    audit_path.parent.mkdir()
    audit_path.write_text('{"artifact_type":"audit"}', encoding="utf-8")
    repair_path = tmp_path / "results" / "repair.json"
    manifest_path = tmp_path / "results" / "repaired.csv"

    result = runner.invoke(
        app,
        [
            "repair",
            str(audit_path),
            "--ratios",
            "0.8,0.1,0.1",
            "--output",
            str(repair_path),
            "--manifest-output",
            str(manifest_path),
        ],
    )

    assert result.exit_code == 2
    assert "failed strict" in result.output
    assert "validation" in result.output
    assert "Traceback" not in result.output
    assert str(tmp_path) not in result.output
    assert not repair_path.exists()
    assert not manifest_path.exists()


@pytest.mark.parametrize(
    ("ratios", "message"),
    (
        ("0.8,0.2", "exactly three"),
        ("0.8,0.2,not-a-number", "exactly three"),
        ("nan,0.5,0.5", "finite"),
        ("-0.1,0.5,0.6", "cannot be negative"),
        ("0.8,0.1,0.2", "sum to one"),
    ),
)
def test_repair_rejects_invalid_ratio_option_before_writing(
    tmp_path: Path,
    ratios: str,
    message: str,
) -> None:
    audit_path = tmp_path / "audit.json"
    audit_path.write_text("{}", encoding="utf-8")

    result = runner.invoke(
        app,
        ["repair", str(audit_path), "--ratios", ratios],
    )

    assert result.exit_code == 2
    assert message in result.output
    assert "Traceback" not in result.output


def test_repair_refuses_to_overwrite_its_source_audit(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.json"
    audit_path.write_text("{}", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "repair",
            str(audit_path),
            "--ratios",
            "0.8,0.1,0.1",
            "--output",
            str(audit_path),
        ],
    )

    assert result.exit_code == 2
    assert "must not" in result.output
    assert "overwrite" in result.output
    assert audit_path.read_text(encoding="utf-8") == "{}"


def test_materialize_copy_round_trip_is_verified_and_privacy_safe(tmp_path: Path) -> None:
    dataset, audit_path, config_path = create_audit_fixture(tmp_path)
    repair_path = tmp_path / "repair" / "repair.json"
    manifest_path = tmp_path / "repair" / "repaired_manifest.csv"
    repair_result = runner.invoke(
        app,
        [
            "repair",
            str(audit_path),
            "--ratios",
            "0.8,0.1,0.1",
            "--config",
            str(config_path),
            "--output",
            str(repair_path),
            "--manifest-output",
            str(manifest_path),
        ],
    )
    assert repair_result.exit_code == 0, repair_result.output
    source_hashes = {
        path.relative_to(dataset).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.rglob("*.png")
    }
    output_dir = tmp_path / "materialized-private-output"

    result = runner.invoke(
        app,
        [
            "materialize",
            str(manifest_path),
            str(output_dir),
            "--dataset-root",
            str(dataset),
            "--mode",
            "copy",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    materialized_images = tuple((output_dir / "images").iterdir())
    assert payload["mode"] == "copy"
    assert payload["output"] == output_dir.name
    assert payload["record_count"] == 2
    assert payload["verified_file_count"] == 2
    assert len(materialized_images) == 2
    assert (output_dir / "manifest.csv").is_file()
    assert source_hashes == {
        path.relative_to(dataset).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.rglob("*.png")
    }
    assert str(tmp_path) not in result.stdout
    assert str(tmp_path) not in (output_dir / "manifest.csv").read_text(encoding="utf-8")


def test_materialize_passes_explicit_symlink_mode_to_helper(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    calls: list[tuple[Path, Path, Path, str]] = []

    def fake_materialize(
        manifest_path: str | Path,
        dataset_root: str | Path,
        output_dir: str | Path,
        *,
        mode: Literal["copy", "symlink"] = "copy",
    ) -> MaterializationResult:
        calls.append(
            (Path(manifest_path), Path(dataset_root), Path(output_dir), mode)
        )
        return MaterializationResult(
            mode=mode,
            record_count=1,
            verified_file_count=1,
            manifest_sha256="a" * 64,
        )

    monkeypatch.setattr(cli_module, "materialize_repaired_manifest", fake_materialize)
    manifest_path = tmp_path / "repaired.csv"
    dataset_root = tmp_path / "source"
    output_dir = tmp_path / "output"

    result = runner.invoke(
        app,
        [
            "materialize",
            str(manifest_path),
            str(output_dir),
            "--dataset-root",
            str(dataset_root),
            "--mode",
            "symlink",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [(manifest_path, dataset_root, output_dir, "symlink")]
    assert json.loads(result.stdout)["mode"] == "symlink"


def test_materialize_reports_safe_domain_error_without_traceback(tmp_path: Path) -> None:
    manifest_path = tmp_path / "private" / "missing.csv"
    dataset_root = tmp_path / "private" / "missing-dataset"
    output_dir = tmp_path / "private" / "output"

    result = runner.invoke(
        app,
        [
            "materialize",
            str(manifest_path),
            str(output_dir),
            "--dataset-root",
            str(dataset_root),
        ],
    )

    assert result.exit_code == 2
    assert "dataset root must be" in result.output
    assert "directory" in result.output
    assert "Traceback" not in result.output
    assert str(tmp_path) not in result.output
    assert not output_dir.exists()
