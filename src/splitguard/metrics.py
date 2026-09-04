"""Pure metrics and privacy-safe reproducibility metadata utilities."""

from __future__ import annotations

import platform
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from importlib import import_module, metadata
from pathlib import Path
from types import ModuleType
from typing import Any

from splitguard.schemas import (
    BinaryMetrics,
    ImageRecord,
    PackageVersion,
    RunMetadata,
    ValidationIssue,
    canonical_sha256,
)

PRODUCTION_DISTRIBUTIONS = tuple(
    sorted(
        (
            "faiss-cpu",
            "jinja2",
            "matplotlib",
            "numpy",
            "pandas",
            "pillow",
            "psutil",
            "pydantic",
            "pydantic-settings",
            "pyyaml",
            "rich",
            "scikit-learn",
            "scipy",
            "torch",
            "torchvision",
            "transformers",
            "typer",
        ),
        key=str.casefold,
    )
)

_GIT_TIMEOUT_SECONDS = 3.0
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")
_DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _safe_component(value: object, *, fallback: str, max_length: int) -> str:
    """Collapse harmless system text and reject path-like or control-bearing values."""
    if not isinstance(value, str):
        return fallback
    collapsed = " ".join(value.split())
    if not collapsed or "\x00" in collapsed:
        return fallback
    if "/" in collapsed or "\\" in collapsed or _DRIVE_PATH_RE.match(collapsed):
        return fallback
    return collapsed[:max_length]


def _run_git(
    arguments: Sequence[str],
    repo_root: Path | None,
) -> subprocess.CompletedProcess[str] | None:
    """Run a fixed Git query with no shell, bounded time, and captured output."""
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_GIT_TIMEOUT_SECONDS,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _git_details(repo_root: str | Path | None) -> tuple[str | None, bool | None]:
    root = Path(repo_root) if repo_root is not None else None
    revision = _run_git(("rev-parse", "--verify", "HEAD"), root)
    if revision is None or revision.returncode != 0:
        return None, None

    commit_sha = revision.stdout.strip().lower()
    if _GIT_SHA_RE.fullmatch(commit_sha) is None:
        return None, None

    status = _run_git(
        ("status", "--porcelain=v1", "--untracked-files=normal"),
        root,
    )
    git_dirty = None if status is None or status.returncode != 0 else bool(status.stdout)
    return commit_sha, git_dirty


def _load_torch() -> ModuleType:
    """Import torch lazily so metadata collection remains cheap when CUDA is absent."""
    return import_module("torch")


def _cuda_details() -> tuple[bool, str | None]:
    try:
        torch_module = _load_torch()
        cuda: Any = torch_module.cuda
        if not bool(cuda.is_available()) or int(cuda.device_count()) <= 0:
            return False, None
        device_index = int(cuda.current_device())
        raw_name = cuda.get_device_name(device_index)
    except (
        AssertionError,
        AttributeError,
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return False, None

    gpu_model = _safe_component(raw_name, fallback="", max_length=512)
    return True, gpu_model or None


def _package_versions() -> tuple[PackageVersion, ...]:
    versions: list[PackageVersion] = []
    for distribution_name in PRODUCTION_DISTRIBUTIONS:
        try:
            raw_version = metadata.version(distribution_name)
        except metadata.PackageNotFoundError:
            raw_version = "not-installed"
        except (OSError, ValueError):
            raw_version = "unknown"
        version = _safe_component(raw_version, fallback="unknown", max_length=256)
        versions.append(PackageVersion(name=distribution_name, version=version))
    return tuple(versions)


def _canonical_seeds(random_seeds: Iterable[int]) -> tuple[int, ...]:
    seeds = tuple(random_seeds)
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise TypeError("random_seeds must contain only integers")
    return tuple(sorted(set(seeds)))


def _os_description() -> str:
    system = _safe_component(platform.system(), fallback="unknown-os", max_length=80)
    release = _safe_component(platform.release(), fallback="unknown-release", max_length=80)
    machine = _safe_component(platform.machine(), fallback="unknown-architecture", max_length=80)
    return f"{system} {release} ({machine})"


def _cpu_description() -> str:
    machine = _safe_component(platform.machine(), fallback="unknown-cpu", max_length=512)
    return _safe_component(platform.processor(), fallback=machine, max_length=512)


def collect_run_metadata(
    configuration_sha256: str,
    dataset_manifest_sha256: str,
    random_seeds: Iterable[int] = (),
    repo_root: str | Path | None = None,
) -> RunMetadata:
    """Collect portable runtime provenance without hostnames, users, or paths."""
    git_commit_sha, git_dirty = _git_details(repo_root)
    cuda_available, gpu_model = _cuda_details()
    python_version = _safe_component(
        platform.python_version(),
        fallback=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        max_length=128,
    )
    return RunMetadata(
        timestamp=datetime.now(UTC),
        git_commit_sha=git_commit_sha,
        git_dirty=git_dirty,
        python_version=python_version,
        os=_os_description(),
        cpu=_cpu_description(),
        cuda_available=cuda_available,
        gpu_model=gpu_model,
        package_versions=_package_versions(),
        configuration_sha256=configuration_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
        random_seeds=_canonical_seeds(random_seeds),
    )


def manifest_snapshot_hash(
    records: Sequence[ImageRecord],
    issues: Sequence[ValidationIssue],
) -> str:
    """Hash stable validated-manifest fields without a dataset root or derived detectors."""
    record_rows = [
        {
            "id": record.id,
            "path": record.path,
            "split": record.split.value,
            "label": record.label,
            "byte_sha256": record.byte_sha256,
            "byte_size": record.byte_size,
            "width": record.width,
            "height": record.height,
            "format": record.format,
        }
        for record in records
    ]
    record_rows.sort(
        key=lambda row: (
            str(row["id"]),
            str(row["path"]),
            str(row["byte_sha256"]),
        )
    )

    issue_rows = [
        {
            "record_id": issue.record_id,
            "path": issue.path,
            "split": issue.split.value if issue.split is not None else None,
            "label": issue.label,
            "code": issue.code.value,
        }
        for issue in issues
    ]
    issue_rows.sort(
        key=lambda row: (
            str(row["record_id"] or ""),
            str(row["path"] or ""),
            str(row["split"] or ""),
            str(row["label"] or ""),
            str(row["code"]),
        )
    )
    return canonical_sha256(
        {
            "snapshot_schema": "splitguard-manifest-v1",
            "records": record_rows,
            "issues": issue_rows,
        }
    )


def binary_metrics(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> BinaryMetrics:
    """Derive precision, recall, and F1 solely from integer confusion counts."""
    counts = (true_positives, false_positives, false_negatives)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
        raise TypeError("metric counts must be integers")
    return BinaryMetrics.from_counts(*counts)


__all__ = [
    "PRODUCTION_DISTRIBUTIONS",
    "binary_metrics",
    "collect_run_metadata",
    "manifest_snapshot_hash",
]
