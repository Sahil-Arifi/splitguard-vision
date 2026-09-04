"""Deterministic, offline end-to-end SplitGuard demonstration.

The demo builds generated images in a private staging directory, exercises the
same validation, fingerprinting, graph, repair, and reporting functions used by
the production commands, and publishes only after verifying that every source
byte is unchanged.  FakeEmbedder is deliberately used so the workflow never
downloads a model or implies that its vectors are DINOv2 results.
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from PIL import Image

from splitguard.conflicts import analyze_conflicts
from splitguard.embeddings import embed_records
from splitguard.graph import EvidencePolicy, build_evidence_graph
from splitguard.hashing import (
    fingerprint_records,
    group_exact_duplicates,
    indexed_phash_pairs,
)
from splitguard.leakage import analyze_leakage
from splitguard.manifest import discover_image_folder
from splitguard.metrics import collect_run_metadata, manifest_snapshot_hash
from splitguard.models.embedder import FakeEmbedder
from splitguard.neighbors import NeighborCandidate, build_neighbor_index
from splitguard.repair import repair_splits, write_repaired_manifest
from splitguard.reporting import generate_report
from splitguard.schemas import (
    AuditArtifact,
    AuditSummary,
    RepairArtifact,
    ValidationIssueCode,
    canonical_json,
    canonical_sha256,
    stable_id,
)
from splitguard.validation import scan_images

_SOURCE_PATH: Final = "train/cat/source.png"
_EXACT_PATH: Final = "test/cat/exact-duplicate.png"
_JPEG_PATH: Final = "test/cat/jpeg-recompressed.jpg"
_RESIZE_PATH: Final = "val/cat/resized.png"
_CONFLICT_PATH: Final = "val/dog/conflicting-label.png"
_CORRUPT_PATH: Final = "test/cat/corrupt.png"
_FIXTURE_PATHS: Final = (
    _SOURCE_PATH,
    _EXACT_PATH,
    _JPEG_PATH,
    _RESIZE_PATH,
    _CONFLICT_PATH,
    _CORRUPT_PATH,
)
_EXPECTED_DUPLICATE_PATHS: Final = (
    _SOURCE_PATH,
    _EXACT_PATH,
    _JPEG_PATH,
    _RESIZE_PATH,
    _CONFLICT_PATH,
)
_CHART_PATHS: Final = (
    "report/detection_pr_curve.png",
    "report/runtime_scaling.png",
    "report/split_distribution.png",
    "report/evaluation_comparison.png",
)
_PHASH_THRESHOLD: Final = 16
_COSINE_THRESHOLD: Final = 0.99
_MAX_IMAGE_PIXELS: Final = 10_000_000
_REPAIR_RATIOS: Final = (0.6, 0.2, 0.2)
_REPAIR_ITERATIONS: Final = 50


class DemoInputError(ValueError):
    """Raised before work starts when demo destinations are unsafe."""


class DemoRunError(RuntimeError):
    """Raised when staged demo generation cannot be safely published."""


@dataclass(frozen=True, slots=True)
class DemoResult:
    """Privacy-safe, root-relative outputs from one completed demo run.

    ``dataset_path`` is relative to the caller-provided workspace.  Every other
    path is relative to the caller-provided output directory.
    """

    seed: int
    dataset_sha256: str
    dataset_path: str
    audit_path: str
    repair_path: str
    repaired_manifest_path: str
    report_html_path: str
    report_markdown_path: str
    chart_paths: tuple[str, ...]
    valid_image_count: int
    invalid_image_count: int
    leakage_group_count: int
    cross_label_conflict_count: int
    source_bytes_unchanged: bool


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be between 0 and 2**32 - 1")
    return seed


def _absolute_lexical(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return Path(os.path.abspath(candidate))


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _prospective_resolve(path: Path) -> Path:
    missing_parts: list[str] = []
    cursor = path
    while not _path_lexists(cursor):
        missing_parts.append(cursor.name)
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent
    try:
        resolved = cursor.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise DemoInputError("demo destination parent is unavailable") from exc
    for part in reversed(missing_parts):
        resolved /= part
    return resolved


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _validated_destinations(
    workspace: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
) -> tuple[Path, Path]:
    workspace_path = _absolute_lexical(workspace)
    output_path = _absolute_lexical(output_dir)
    if _path_lexists(workspace_path) or _path_lexists(output_path):
        raise DemoInputError("demo workspace and output must not already exist")

    prospective_workspace = _prospective_resolve(workspace_path)
    prospective_output = _prospective_resolve(output_path)
    if _paths_overlap(prospective_workspace, prospective_output):
        raise DemoInputError("demo workspace and output must not overlap")

    try:
        prospective_workspace.parent.mkdir(parents=True, exist_ok=True)
        prospective_output.parent.mkdir(parents=True, exist_ok=True)
        workspace_parent = prospective_workspace.parent.resolve(strict=True)
        output_parent = prospective_output.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise DemoInputError("demo destination parent is unavailable") from exc

    resolved_workspace = workspace_parent / prospective_workspace.name
    resolved_output = output_parent / prospective_output.name
    if _path_lexists(resolved_workspace) or _path_lexists(resolved_output):
        raise DemoInputError("demo workspace and output must not already exist")
    if _paths_overlap(resolved_workspace, resolved_output):
        raise DemoInputError("demo workspace and output must not overlap")
    return resolved_workspace, resolved_output


def _base_image(seed: int) -> Image.Image:
    width, height = 96, 72
    seed_digest = hashlib.sha256(seed.to_bytes(4, "big", signed=False)).digest()
    pixels = bytearray(width * height * 3)
    offset = 0
    for y in range(height):
        for x in range(width):
            checker = 43 if ((x // 9) + (y // 7)) % 2 else 0
            ring = ((x - 48) ** 2 + (y - 36) ** 2) // 31
            pixels[offset] = (3 * x + 5 * y + checker + seed_digest[0]) % 256
            pixels[offset + 1] = (7 * x + 2 * y + ring + seed_digest[7]) % 256
            pixels[offset + 2] = (x + 11 * y + checker + seed_digest[19]) % 256
            offset += 3
    return Image.frombytes("RGB", (width, height), bytes(pixels))


def _encoded_image(image: Image.Image, image_format: str, **options: object) -> bytes:
    stream = io.BytesIO()
    image.save(stream, format=image_format, **options)
    return stream.getvalue()


def _write_fixture(dataset_root: Path, seed: int) -> None:
    for relative_path in _FIXTURE_PATHS:
        (dataset_root / Path(relative_path).parent).mkdir(parents=True, exist_ok=True)

    image = _base_image(seed)
    try:
        source_payload = _encoded_image(
            image,
            "PNG",
            compress_level=6,
            optimize=False,
        )
        jpeg_payload = _encoded_image(
            image,
            "JPEG",
            quality=72,
            optimize=False,
            progressive=False,
            subsampling=2,
        )
        resized = image.resize((72, 54), Image.Resampling.LANCZOS)
        try:
            resized_payload = _encoded_image(
                resized,
                "PNG",
                compress_level=6,
                optimize=False,
            )
        finally:
            resized.close()
    finally:
        image.close()

    if source_payload in {jpeg_payload, resized_payload}:
        raise DemoRunError("generated transformations must be byte-distinct")

    (dataset_root / _SOURCE_PATH).write_bytes(source_payload)
    (dataset_root / _EXACT_PATH).write_bytes(source_payload)
    (dataset_root / _CONFLICT_PATH).write_bytes(source_payload)
    (dataset_root / _JPEG_PATH).write_bytes(jpeg_payload)
    (dataset_root / _RESIZE_PATH).write_bytes(resized_payload)
    malformed = (
        b"SPLITGUARD-DELIBERATELY-MALFORMED\x00"
        + seed.to_bytes(4, "big", signed=False)
        + hashlib.sha256(source_payload).digest()
    )
    (dataset_root / _CORRUPT_PATH).write_bytes(malformed)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _dataset_snapshot(dataset_root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in sorted(dataset_root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise DemoRunError("generated demo data unexpectedly contains a symbolic link")
        if path.is_file():
            relative = path.relative_to(dataset_root).as_posix()
            snapshot[relative] = _sha256_file(path)
    if tuple(sorted(snapshot)) != tuple(sorted(_FIXTURE_PATHS)):
        raise DemoRunError("generated demo data does not match the fixed fixture contract")
    return snapshot


def _atomic_write_artifact(path: Path, artifact: AuditArtifact | RepairArtifact) -> None:
    payload = (canonical_json(artifact) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_audit_contract(audit: AuditArtifact) -> None:
    records_by_path = {record.path: record for record in audit.records}
    if set(records_by_path) != set(_EXPECTED_DUPLICATE_PATHS):
        raise DemoRunError("demo validation did not preserve the fixed valid-image set")
    malformed_issues = tuple(
        issue
        for issue in audit.invalid_records
        if issue.path == _CORRUPT_PATH
        and issue.code is ValidationIssueCode.MALFORMED_IMAGE
    )
    if len(malformed_issues) != 1:
        raise DemoRunError("demo validation did not identify the fixed corrupt image")

    expected_ids = {
        stable_id("img", relative_path) for relative_path in _EXPECTED_DUPLICATE_PATHS
    }
    if not any(expected_ids <= set(family.member_ids) for family in audit.families):
        raise DemoRunError("demo detectors did not recover the fixed duplicate family")
    if audit.summary.exact_leakage_group_count < 1:
        raise DemoRunError("demo audit did not identify the fixed exact leakage")
    if audit.summary.cross_label_conflict_count < 1:
        raise DemoRunError("demo audit did not identify the fixed label conflict")


def _install_new_directory(staging: Path, destination: Path) -> None:
    if _path_lexists(destination):
        raise DemoRunError("a demo destination changed during publication")
    os.rename(staging, destination)


def _remove_owned_directory(path: Path) -> bool:
    if not _path_lexists(path):
        return True
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    except OSError:
        return False
    return True


def run_offline_demo(
    workspace: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    seed: int = 20260903,
) -> DemoResult:
    """Run and atomically publish a complete generated local demo.

    Both destinations must be new and non-overlapping.  The workspace receives
    ``dataset`` and the local embedding cache.  The output receives canonical
    JSON/CSV artifacts and a fully local report.  Returned paths never contain
    either caller-provided absolute root.
    """

    accepted_seed = _validate_seed(seed)
    workspace_path, output_path = _validated_destinations(workspace, output_dir)
    workspace_stage: Path | None = None
    output_stage: Path | None = None
    workspace_installed = False
    output_installed = False

    try:
        workspace_stage = Path(
            tempfile.mkdtemp(
                prefix=".splitguard-demo-workspace-",
                dir=workspace_path.parent,
            )
        )
        output_stage = Path(
            tempfile.mkdtemp(
                prefix=".splitguard-demo-output-",
                dir=output_path.parent,
            )
        )
        dataset_root = workspace_stage / "dataset"
        _write_fixture(dataset_root, accepted_seed)
        source_snapshot = _dataset_snapshot(dataset_root)
        dataset_tree_sha256 = canonical_sha256(source_snapshot)

        manifest = discover_image_folder(dataset_root)
        scan_result = scan_images(
            manifest.root,
            manifest.entries,
            max_image_pixels=_MAX_IMAGE_PIXELS,
        )
        records = fingerprint_records(manifest.root, scan_result.records)
        exact_clusters = group_exact_duplicates(records)
        phash_candidates = indexed_phash_pairs(records, _PHASH_THRESHOLD)

        embedder = FakeEmbedder(
            dimension=16,
            seed=accepted_seed,
            preprocessing_version="demo-fake-rgb-pixels-v1",
        )
        embedding_result = embed_records(
            manifest.root,
            records,
            embedder=embedder,
            cache_dir=workspace_stage / "cache",
            batch_size=8,
        )
        neighbor_candidates: tuple[NeighborCandidate, ...] = ()
        if len(records) > 1:
            neighbor_index = build_neighbor_index(
                embedding_result.record_ids,
                embedding_result.vectors,
                kind="flat_ip",
                threads=1,
            )
            neighbor_candidates = neighbor_index.search(
                embedding_result.record_ids,
                embedding_result.vectors,
                k=min(4, len(records) - 1),
                cosine_threshold=_COSINE_THRESHOLD,
            )

        policy = EvidencePolicy()
        graph = build_evidence_graph(
            records,
            exact_clusters,
            phash_candidates,
            neighbor_candidates,
            phash_threshold=_PHASH_THRESHOLD,
            cosine_threshold=_COSINE_THRESHOLD,
            policy=policy,
        )
        leakage = analyze_leakage(
            records,
            graph.families,
            graph.definite_edges,
            review_edges=graph.review_edges,
        )
        conflicts = analyze_conflicts(
            records,
            graph.families,
            graph.definite_edges,
            review_edges=graph.review_edges,
        )
        dataset_manifest_sha256 = manifest_snapshot_hash(records, scan_result.issues)
        audit_configuration_sha256 = canonical_sha256(
            {
                "demo_schema": "splitguard-offline-demo-v1",
                "fixture_tree_sha256": dataset_tree_sha256,
                "seed": accepted_seed,
                "validation": {"max_image_pixels": _MAX_IMAGE_PIXELS},
                "phash": {"hamming_threshold": _PHASH_THRESHOLD},
                "embedding": {
                    "backend": "fake_pixel_derived_offline",
                    "dimension": embedder.dimension,
                    "model_identity": embedder.model_identity,
                    "preprocessing_version": embedder.preprocessing_version,
                },
                "neighbors": {
                    "cosine_threshold": _COSINE_THRESHOLD,
                    "index": "flat_ip",
                    "k": min(4, max(0, len(records) - 1)),
                    "threads": 1,
                },
                "policy": policy.model_dump(mode="json"),
            }
        )
        audit_metadata = collect_run_metadata(
            audit_configuration_sha256,
            dataset_manifest_sha256,
            (accepted_seed,),
            repo_root=Path(__file__).parents[2],
        )
        audit = AuditArtifact(
            metadata=audit_metadata,
            records=records,
            invalid_records=scan_result.issues,
            edges=graph.edges,
            families=graph.families,
            leakage_groups=leakage.leakage_groups,
            label_conflicts=conflicts.conflicts,
            summary=AuditSummary(
                valid_image_count=len(records),
                invalid_image_count=len(scan_result.issues),
                leakage_group_count=leakage.leakage_group_count,
                contaminated_image_count=leakage.contaminated_image_count,
                evaluation_image_count=leakage.evaluation_image_count,
                contaminated_evaluation_fraction=(
                    leakage.contaminated_evaluation_fraction
                ),
                exact_leakage_group_count=leakage.exact_leakage_group_count,
                perceptual_leakage_group_count=(
                    leakage.perceptual_leakage_group_count
                ),
                embedding_only_review_count=leakage.embedding_only_review_count,
                cross_label_conflict_count=conflicts.cross_label_conflict_count,
            ),
        )
        _validate_audit_contract(audit)

        audit_path = output_stage / "audit.json"
        repair_path = output_stage / "repair.json"
        repaired_manifest_path = output_stage / "repaired_manifest.csv"
        _atomic_write_artifact(audit_path, audit)

        repair_plan = repair_splits(
            audit.records,
            audit.families,
            target_ratios=_REPAIR_RATIOS,
            split_size_weight=1.0,
            class_balance_weight=1.0,
            seed=accepted_seed,
            local_iterations=_REPAIR_ITERATIONS,
        )
        if (
            not repair_plan.hard_group_invariant_satisfied
            or repair_plan.definite_leakage_groups_after != 0
        ):
            raise DemoRunError("demo repair did not satisfy the hard family invariant")
        repaired_manifest_sha256 = write_repaired_manifest(
            repaired_manifest_path,
            audit.records,
            repair_plan.assignments,
        )
        repair_configuration_sha256 = canonical_sha256(
            {
                "base_configuration_sha256": audit_configuration_sha256,
                "repair": {
                    "class_balance_weight": repair_plan.class_balance_weight,
                    "local_improvement_iterations": repair_plan.local_iterations,
                    "random_seed": repair_plan.seed,
                    "split_size_weight": repair_plan.split_size_weight,
                },
                "requested_ratios": [
                    {"ratio": item.ratio, "split": item.split.value}
                    for item in repair_plan.requested_ratios
                ],
            }
        )
        repair_metadata = collect_run_metadata(
            repair_configuration_sha256,
            dataset_manifest_sha256,
            (accepted_seed,),
            repo_root=Path(__file__).parents[2],
        )
        repair = RepairArtifact(
            metadata=repair_metadata,
            requested_ratios=repair_plan.requested_ratios,
            integer_targets=repair_plan.integer_targets,
            assignments=repair_plan.assignments,
            excluded_invalid_ids=tuple(
                sorted(
                    {
                        issue.record_id
                        for issue in audit.invalid_records
                        if issue.record_id is not None
                    }
                )
            ),
            infeasibility_warnings=repair_plan.infeasibility_warnings,
            split_size_weight=repair_plan.split_size_weight,
            class_balance_weight=repair_plan.class_balance_weight,
            random_seed=repair_plan.seed,
            local_improvement_iterations=repair_plan.local_iterations,
            repaired_manifest_sha256=repaired_manifest_sha256,
            before_split_statistics=repair_plan.before_split_statistics,
            after_split_statistics=repair_plan.after_split_statistics,
            summary=repair_plan.summary,
        )
        _atomic_write_artifact(repair_path, repair)

        report_result = generate_report(
            audit_path,
            output_stage / "report",
            repair_path=repair_path,
            dataset_root=dataset_root,
            thumbnail_size=96,
            max_thumbnails=16,
        )
        expected_report_files = (
            output_stage / "report" / "report.html",
            output_stage / "report" / "report.md",
            *(output_stage / relative_path for relative_path in _CHART_PATHS),
        )
        if (
            report_result.output_dir != (output_stage / "report").resolve(strict=False)
            or any(not path.is_file() for path in expected_report_files)
        ):
            raise DemoRunError("demo report did not produce its complete local artifact set")

        if _dataset_snapshot(dataset_root) != source_snapshot:
            raise DemoRunError("generated source data changed during the demo workflow")

        _install_new_directory(workspace_stage, workspace_path)
        workspace_installed = True
        workspace_stage = None
        try:
            _install_new_directory(output_stage, output_path)
        except Exception:
            if not _remove_owned_directory(workspace_path):
                raise DemoRunError("demo publication rollback could not be completed") from None
            workspace_installed = False
            raise
        output_installed = True
        output_stage = None

        return DemoResult(
            seed=accepted_seed,
            dataset_sha256=dataset_tree_sha256,
            dataset_path="dataset",
            audit_path="audit.json",
            repair_path="repair.json",
            repaired_manifest_path="repaired_manifest.csv",
            report_html_path="report/report.html",
            report_markdown_path="report/report.md",
            chart_paths=_CHART_PATHS,
            valid_image_count=audit.summary.valid_image_count,
            invalid_image_count=audit.summary.invalid_image_count,
            leakage_group_count=audit.summary.leakage_group_count,
            cross_label_conflict_count=audit.summary.cross_label_conflict_count,
            source_bytes_unchanged=True,
        )
    except (DemoInputError, DemoRunError):
        raise
    except Exception as exc:
        raise DemoRunError(
            "offline demo failed before verified publication"
        ) from exc
    finally:
        if workspace_stage is not None:
            _remove_owned_directory(workspace_stage)
        if output_stage is not None:
            _remove_owned_directory(output_stage)
        if output_installed and not workspace_installed:
            _remove_owned_directory(output_path)


__all__ = ["DemoInputError", "DemoResult", "DemoRunError", "run_offline_demo"]
