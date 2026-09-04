"""Offline integration and publication-safety tests for the generated demo."""

from __future__ import annotations

import hashlib
import socket
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, cast

import pytest

import splitguard.demo as demo_module
from splitguard.demo import DemoInputError, DemoRunError, run_offline_demo
from splitguard.reporting import ReportResult
from splitguard.schemas import (
    AuditArtifact,
    RepairArtifact,
    ValidationIssueCode,
    canonical_json,
    canonical_sha256,
)

_SOURCE_PATH = "train/cat/source.png"
_EXACT_PATH = "test/cat/exact-duplicate.png"
_JPEG_PATH = "test/cat/jpeg-recompressed.jpg"
_RESIZE_PATH = "val/cat/resized.png"
_CONFLICT_PATH = "val/dog/conflicting-label.png"
_CORRUPT_PATH = "test/cat/corrupt.png"


def _deny_network(*_args: object, **_kwargs: object) -> NoReturn:
    raise AssertionError("the offline demo attempted to create a network socket")


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file()
    }


def _assert_portable_relative_path(path: str) -> None:
    parsed = PurePosixPath(path)
    assert not parsed.is_absolute()
    assert "\\" not in path
    assert ".." not in parsed.parts


def test_full_demo_is_offline_deterministic_and_privacy_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket, "socket", _deny_network)
    monkeypatch.setattr(socket, "create_connection", _deny_network)

    first_workspace = tmp_path / "first-workspace"
    first_output = tmp_path / "first-output"
    second_workspace = tmp_path / "second-workspace"
    second_output = tmp_path / "second-output"
    first = run_offline_demo(first_workspace, first_output, seed=71)
    second = run_offline_demo(second_workspace, second_output, seed=71)

    assert first.dataset_sha256 == second.dataset_sha256
    assert _file_hashes(first_workspace / first.dataset_path) == _file_hashes(
        second_workspace / second.dataset_path
    )
    assert first.dataset_sha256 == canonical_sha256(
        _file_hashes(first_workspace / first.dataset_path)
    )
    assert first.source_bytes_unchanged is True
    assert first.valid_image_count == 5
    assert first.invalid_image_count == 1
    assert first.leakage_group_count == 1
    assert first.cross_label_conflict_count == 1

    relative_paths = (
        first.dataset_path,
        first.audit_path,
        first.repair_path,
        first.repaired_manifest_path,
        first.report_html_path,
        first.report_markdown_path,
        *first.chart_paths,
    )
    for relative_path in relative_paths:
        _assert_portable_relative_path(relative_path)
        assert str(tmp_path) not in relative_path

    audit_path = first_output / first.audit_path
    repair_path = first_output / first.repair_path
    manifest_path = first_output / first.repaired_manifest_path
    audit = AuditArtifact.model_validate_json(audit_path.read_bytes())
    repair = RepairArtifact.model_validate_json(repair_path.read_bytes())

    assert audit_path.read_text(encoding="utf-8") == canonical_json(audit) + "\n"
    assert repair_path.read_text(encoding="utf-8") == canonical_json(repair) + "\n"
    assert audit.metadata.dataset_manifest_sha256 == repair.metadata.dataset_manifest_sha256
    assert audit.summary.valid_image_count == 5
    assert audit.summary.invalid_image_count == 1
    assert audit.summary.exact_leakage_group_count == 1
    assert audit.summary.perceptual_leakage_group_count == 0
    assert audit.summary.cross_label_conflict_count == 1
    assert {
        issue.path: issue.code for issue in audit.invalid_records
    } == {_CORRUPT_PATH: ValidationIssueCode.MALFORMED_IMAGE}

    records_by_path = {record.path: record for record in audit.records}
    assert set(records_by_path) == {
        _SOURCE_PATH,
        _EXACT_PATH,
        _JPEG_PATH,
        _RESIZE_PATH,
        _CONFLICT_PATH,
    }
    source_sha = records_by_path[_SOURCE_PATH].byte_sha256
    assert records_by_path[_EXACT_PATH].byte_sha256 == source_sha
    assert records_by_path[_CONFLICT_PATH].byte_sha256 == source_sha
    assert records_by_path[_JPEG_PATH].byte_sha256 != source_sha
    assert records_by_path[_RESIZE_PATH].byte_sha256 != source_sha
    assert records_by_path[_RESIZE_PATH].width < records_by_path[_SOURCE_PATH].width
    assert records_by_path[_RESIZE_PATH].height < records_by_path[_SOURCE_PATH].height

    expected_ids = {record.id for record in records_by_path.values()}
    assert any(expected_ids <= set(family.member_ids) for family in audit.families)
    assert repair.summary.hard_group_invariant_satisfied is True
    assert repair.summary.definite_leakage_groups_before == 1
    assert repair.summary.definite_leakage_groups_after == 0
    assert len(repair.excluded_invalid_ids) == 1
    assert repair.repaired_manifest_sha256 == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    manifest_lines = manifest_path.read_text(encoding="utf-8").splitlines()
    assert manifest_lines[0] == "path,split,label"
    assert len(manifest_lines) == 6
    assert all(_CORRUPT_PATH not in line for line in manifest_lines)

    report_files = (
        first_output / first.report_html_path,
        first_output / first.report_markdown_path,
        *(first_output / path for path in first.chart_paths),
    )
    assert all(path.is_file() for path in report_files)
    assert all(
        (first_output / chart_path).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        for chart_path in first.chart_paths
    )
    assert tuple((first_output / "report" / "report_thumbnails").glob("*.png"))

    private_markers = {str(tmp_path), tmp_path.as_posix()}
    for path in (audit_path, repair_path, manifest_path, *report_files[:2]):
        shareable_text = path.read_text(encoding="utf-8")
        assert all(marker not in shareable_text for marker in private_markers)


def test_source_tampering_aborts_and_cleans_both_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_generate_report = demo_module.generate_report

    def mutating_report(*args: Any, **kwargs: Any) -> ReportResult:
        result = original_generate_report(*args, **kwargs)
        dataset_root = Path(cast(str | Path, kwargs["dataset_root"]))
        (dataset_root / _SOURCE_PATH).write_bytes(b"tampered after report generation")
        return result

    monkeypatch.setattr(demo_module, "generate_report", mutating_report)
    workspace = tmp_path / "workspace"
    output = tmp_path / "output"

    with pytest.raises(DemoRunError, match="source data changed"):
        run_offline_demo(workspace, output, seed=5)

    assert not workspace.exists()
    assert not output.exists()
    assert not tuple(tmp_path.glob(".splitguard-demo-workspace-*"))
    assert not tuple(tmp_path.glob(".splitguard-demo-output-*"))


@pytest.mark.parametrize(
    ("workspace_relative", "output_relative"),
    (
        ("same", "same"),
        ("workspace", "workspace/output"),
        ("output/workspace", "output"),
        ("scope/../workspace", "workspace/nested/output"),
    ),
)
def test_overlapping_destinations_are_rejected_before_creation(
    tmp_path: Path,
    workspace_relative: str,
    output_relative: str,
) -> None:
    workspace = tmp_path / workspace_relative
    output = tmp_path / output_relative

    with pytest.raises(DemoInputError, match="must not overlap"):
        run_offline_demo(workspace, output)

    assert not workspace.exists()
    assert not output.exists()


def test_existing_destination_is_never_reused(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()

    with pytest.raises(DemoInputError, match="must not already exist"):
        run_offline_demo(existing, tmp_path / "output")
    with pytest.raises(DemoInputError, match="must not already exist"):
        run_offline_demo(tmp_path / "workspace", existing)

    assert tuple(existing.iterdir()) == ()


def test_symlink_alias_cannot_hide_destination_overlap(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(real_parent, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")

    workspace = alias / "workspace"
    output = real_parent / "workspace" / "output"
    with pytest.raises(DemoInputError, match="must not overlap"):
        run_offline_demo(workspace, output)

    assert not workspace.exists()
    assert not output.exists()
