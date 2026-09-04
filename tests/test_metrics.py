"""Tests for reproducibility metadata and pure metric calculations."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import splitguard.metrics as metrics_module
from splitguard.metrics import (
    PRODUCTION_DISTRIBUTIONS,
    binary_metrics,
    collect_run_metadata,
    manifest_snapshot_hash,
)
from splitguard.schemas import (
    ImageRecord,
    Split,
    ValidationIssue,
    ValidationIssueCode,
    stable_id,
)

ZERO_SHA256 = "0" * 64
ONE_SHA256 = "1" * 64


def _record(path: str, byte_sha256: str = ZERO_SHA256) -> ImageRecord:
    return ImageRecord(
        id=stable_id("image", path),
        path=path,
        split=Split.TRAIN,
        label="cat",
        byte_sha256=byte_sha256,
        byte_size=123,
        width=12,
        height=8,
        format="png",
    )


def _issue(path: str, code: ValidationIssueCode) -> ValidationIssue:
    return ValidationIssue(
        record_id=stable_id("image", path),
        path=path,
        split=Split.TEST,
        label="dog",
        code=code,
        message="portable validation message",
    )


def _completed(
    arguments: list[str],
    *,
    returncode: int,
    stdout: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(arguments, returncode, stdout=stdout, stderr="")


@pytest.mark.parametrize(("status_output", "expected_dirty"), [("", False), (" M file\n", True)])
def test_git_metadata_for_clean_and_dirty_repository(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status_output: str,
    expected_dirty: bool,
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((arguments, kwargs))
        if "rev-parse" in arguments:
            return _completed(arguments, returncode=0, stdout=f"{'a' * 40}\n")
        return _completed(arguments, returncode=0, stdout=status_output)

    monkeypatch.setattr(metrics_module.subprocess, "run", fake_run)

    commit_sha, dirty = metrics_module._git_details(tmp_path)

    assert commit_sha == "a" * 40
    assert dirty is expected_dirty
    assert len(calls) == 2
    for arguments, kwargs in calls:
        assert arguments[0] == "git"
        assert kwargs["cwd"] == tmp_path
        assert kwargs["timeout"] == 3.0
        assert kwargs["shell"] is False


def test_git_metadata_gracefully_handles_non_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_run(arguments: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return _completed(arguments, returncode=128, stdout="")

    monkeypatch.setattr(metrics_module.subprocess, "run", fake_run)

    assert metrics_module._git_details(None) == (None, None)
    assert calls == 1


def test_git_metadata_sanitizes_invalid_output_and_timeouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid_sha(arguments: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return _completed(arguments, returncode=0, stdout="/private/not-a-sha\n")

    monkeypatch.setattr(metrics_module.subprocess, "run", invalid_sha)
    assert metrics_module._git_details(None) == (None, None)

    def timeout(_arguments: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired("git", 3.0)

    monkeypatch.setattr(metrics_module.subprocess, "run", timeout)
    assert metrics_module._git_details(None) == (None, None)


def test_cuda_metadata_requires_an_available_usable_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usable_cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        current_device=lambda: 0,
        get_device_name=lambda _index: "Test GPU",
    )
    monkeypatch.setattr(
        metrics_module,
        "_load_torch",
        lambda: SimpleNamespace(cuda=usable_cuda),
    )
    assert metrics_module._cuda_details() == (True, "Test GPU")

    unavailable_cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setattr(
        metrics_module,
        "_load_torch",
        lambda: SimpleNamespace(cuda=unavailable_cuda),
    )
    assert metrics_module._cuda_details() == (False, None)

    broken_cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: (_ for _ in ()).throw(AssertionError("no CUDA runtime")),
    )
    monkeypatch.setattr(
        metrics_module,
        "_load_torch",
        lambda: SimpleNamespace(cuda=broken_cuda),
    )
    assert metrics_module._cuda_details() == (False, None)


def test_cuda_metadata_does_not_publish_a_path_like_device_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        current_device=lambda: 0,
        get_device_name=lambda _index: r"C:\Users\private\gpu",
    )
    monkeypatch.setattr(metrics_module, "_load_torch", lambda: SimpleNamespace(cuda=cuda))

    assert metrics_module._cuda_details() == (True, None)


def test_required_package_versions_are_complete_sorted_and_explicit_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_version(distribution_name: str) -> str:
        if distribution_name == "pillow":
            raise metrics_module.metadata.PackageNotFoundError(distribution_name)
        return f"1.0+{distribution_name}"

    monkeypatch.setattr(metrics_module.metadata, "version", fake_version)

    packages = metrics_module._package_versions()

    assert tuple(package.name for package in packages) == PRODUCTION_DISTRIBUTIONS
    assert tuple(package.name for package in packages) == tuple(
        sorted(PRODUCTION_DISTRIBUTIONS, key=str.casefold)
    )
    assert next(package.version for package in packages if package.name == "pillow") == (
        "not-installed"
    )


def test_collect_run_metadata_is_canonical_and_privacy_safe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    private_fragment = "private-user"
    monkeypatch.setattr(metrics_module, "_git_details", lambda _root: ("b" * 40, True))
    monkeypatch.setattr(metrics_module, "_cuda_details", lambda: (False, None))
    monkeypatch.setattr(metrics_module, "_package_versions", lambda: ())
    monkeypatch.setattr(metrics_module.platform, "system", lambda: "TestOS")
    monkeypatch.setattr(metrics_module.platform, "release", lambda: "1.2")
    monkeypatch.setattr(metrics_module.platform, "machine", lambda: "x86_test")
    monkeypatch.setattr(
        metrics_module.platform,
        "processor",
        lambda: f"C:\\Users\\{private_fragment}\\cpu",
    )
    monkeypatch.setattr(metrics_module.platform, "python_version", lambda: "3.11.9")

    result = collect_run_metadata(
        ZERO_SHA256,
        ONE_SHA256,
        random_seeds=[11, 7, 11],
        repo_root=tmp_path / private_fragment,
    )
    serialized = result.model_dump_json()

    assert result.git_commit_sha == "b" * 40
    assert result.git_dirty is True
    assert result.timestamp.tzinfo is not None
    assert result.timestamp.utcoffset() is not None
    assert result.python_version == "3.11.9"
    assert result.os == "TestOS 1.2 (x86_test)"
    assert result.cpu == "x86_test"
    assert result.random_seeds == (7, 11)
    assert private_fragment not in serialized
    assert str(tmp_path) not in serialized


def test_collect_run_metadata_rejects_non_integer_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(metrics_module, "_git_details", lambda _root: (None, None))
    monkeypatch.setattr(metrics_module, "_cuda_details", lambda: (False, None))
    monkeypatch.setattr(metrics_module, "_package_versions", lambda: ())

    with pytest.raises(TypeError, match="only integers"):
        collect_run_metadata(ZERO_SHA256, ONE_SHA256, random_seeds=[True])


def test_manifest_snapshot_hash_is_order_independent() -> None:
    records = (_record("train/a.png"), _record("train/b.png", ONE_SHA256))
    issues = (
        _issue("test/missing.png", ValidationIssueCode.MISSING_PATH),
        _issue("test/bad.png", ValidationIssueCode.MALFORMED_IMAGE),
    )

    forward = manifest_snapshot_hash(records, issues)
    reversed_order = manifest_snapshot_hash(tuple(reversed(records)), tuple(reversed(issues)))

    assert forward == reversed_order
    assert len(forward) == 64


def test_manifest_snapshot_hash_uses_full_content_hash_and_issue_identity() -> None:
    first_hash = "a" * 63 + "0"
    second_hash = "a" * 63 + "1"
    base = manifest_snapshot_hash((_record("train/a.png", first_hash),), ())
    changed_content = manifest_snapshot_hash((_record("train/a.png", second_hash),), ())
    missing = manifest_snapshot_hash(
        (),
        (_issue("test/a.png", ValidationIssueCode.MISSING_PATH),),
    )
    malformed = manifest_snapshot_hash(
        (),
        (_issue("test/a.png", ValidationIssueCode.MALFORMED_IMAGE),),
    )

    assert base != changed_content
    assert missing != malformed


def test_manifest_snapshot_payload_contains_no_absolute_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def capture(payload: Any) -> str:
        captured["payload"] = payload
        return "c" * 64

    monkeypatch.setattr(metrics_module, "canonical_sha256", capture)

    result = manifest_snapshot_hash(
        (_record("train/cat/image.png"),),
        (_issue("test/dog/missing.png", ValidationIssueCode.MISSING_PATH),),
    )
    serialized = json.dumps(captured)

    assert result == "c" * 64
    assert ZERO_SHA256 in serialized
    assert "train/cat/image.png" in serialized
    assert str(tmp_path) not in serialized
    assert "portable validation message" not in serialized


def test_binary_metrics_handles_regular_perfect_and_empty_counts() -> None:
    regular = binary_metrics(8, 2, 4)
    perfect = binary_metrics(5, 0, 0)
    empty = binary_metrics(0, 0, 0)

    assert regular.precision == pytest.approx(0.8)
    assert regular.recall == pytest.approx(8 / 12)
    assert regular.f1 == pytest.approx(8 / 11)
    assert (perfect.precision, perfect.recall, perfect.f1) == (1.0, 1.0, 1.0)
    assert (empty.precision, empty.recall, empty.f1) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("counts", [(-1, 0, 0), (0, -1, 0), (0, 0, -1)])
def test_binary_metrics_rejects_negative_counts(counts: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        binary_metrics(*counts)


@pytest.mark.parametrize("counts", [(True, 0, 0), (1.0, 0, 0)])
def test_binary_metrics_rejects_non_integer_counts(counts: tuple[object, object, object]) -> None:
    with pytest.raises(TypeError, match="must be integers"):
        binary_metrics(*counts)  # type: ignore[arg-type]
