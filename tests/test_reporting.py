"""Offline tests for privacy-safe static report generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import splitguard.reporting as reporting_module
from splitguard.reporting import ReportInputError, generate_report
from splitguard.schemas import (
    AccuracyMetric,
    AuditArtifact,
    AuditSummary,
    BinaryMetrics,
    ConflictKind,
    DetectionBenchmarkArtifact,
    DetectionMetricRow,
    DuplicateClassification,
    DuplicateEdge,
    DuplicateEvidence,
    DuplicateFamily,
    EdgeDecision,
    EmbeddingProvenance,
    ImageRecord,
    LabelConflict,
    LeakageGroup,
    NamedAccuracy,
    RepairArtifact,
    RepairAssignment,
    RepairSummary,
    RunMetadata,
    ScalingBenchmarkArtifact,
    ScalingMetricRow,
    Split,
    SplitBoundary,
    SplitRatio,
    SplitStatistics,
    TrainingArtifact,
    TrainingCondition,
    TrainingConditionSummary,
    TrainingContaminationGroundTruth,
    TrainingExperimentSummary,
    TrainingRun,
    ValidationIssue,
    ValidationIssueCode,
    canonical_json,
    canonical_sha256,
    family_id_for,
    stable_id,
)


@dataclass(frozen=True)
class ArtifactFixture:
    dataset_root: Path
    output_dir: Path
    audit_path: Path
    repair_path: Path
    detection_path: Path
    scaling_path: Path
    training_path: Path


def _metadata(dataset_hash: str, *, seed: int = 7) -> RunMetadata:
    return RunMetadata(
        timestamp=datetime(2026, 9, 3, 12, 0, tzinfo=UTC),
        git_commit_sha="a" * 40,
        git_dirty=False,
        python_version="3.11.9",
        os="test-os",
        cpu="test-cpu",
        cuda_available=False,
        package_versions=(),
        configuration_sha256="c" * 64,
        dataset_manifest_sha256=dataset_hash,
        random_seeds=(seed,),
    )


def _write_image(root: Path, relative: str, color: tuple[int, int, int]) -> ImageRecord:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.zeros((18, 24, 3), dtype=np.uint8)
    pixels[:, :] = color
    Image.fromarray(pixels, mode="RGB").save(path, format="PNG")
    payload = path.read_bytes()
    return ImageRecord(
        id=stable_id("img", relative),
        path=relative,
        split=Split(relative.split("/", maxsplit=1)[0]),
        label="cat",
        byte_sha256=hashlib.sha256(payload).hexdigest(),
        byte_size=len(payload),
        width=24,
        height=18,
        format="png",
        phash=0,
    )


def _accuracy(correct: int, total: int) -> AccuracyMetric:
    return AccuracyMetric(
        correct=correct,
        total=total,
        accuracy=correct / total if total else 0.0,
    )


def _write_json(path: Path, artifact: object) -> None:
    path.write_text(canonical_json(artifact) + "\n", encoding="utf-8")  # type: ignore[arg-type]


def _artifact_fixture(tmp_path: Path) -> ArtifactFixture:
    dataset_root = tmp_path / "private-dataset"
    first = _write_image(dataset_root, "train/cat/source.png", (25, 80, 130))
    copy_path = dataset_root / "test" / "dog" / "copy.png"
    copy_path.parent.mkdir(parents=True)
    copy_path.write_bytes((dataset_root / first.path).read_bytes())
    copy_payload = copy_path.read_bytes()
    second = ImageRecord(
        id=stable_id("img", "test/dog/copy.png"),
        path="test/dog/copy.png",
        split=Split.TEST,
        label="dog | <script>alert(1)</script>",
        byte_sha256=hashlib.sha256(copy_payload).hexdigest(),
        byte_size=len(copy_payload),
        width=24,
        height=18,
        format="png",
        phash=0,
    )
    third = _write_image(dataset_root, "val/cat/resized.png", (80, 20, 140))
    fourth = _write_image(dataset_root, "test/cat/resized-copy.png", (82, 22, 142))
    records = tuple(sorted((first, second, third, fourth), key=lambda item: item.id))

    exact_members = tuple(sorted((first.id, second.id)))
    near_members = tuple(sorted((third.id, fourth.id)))
    exact_family = DuplicateFamily(
        family_id=family_id_for(exact_members),
        member_ids=exact_members,
        edge_count=1,
    )
    near_family = DuplicateFamily(
        family_id=family_id_for(near_members),
        member_ids=near_members,
        edge_count=1,
    )

    exact_left, exact_right = exact_members
    near_left, near_right = near_members
    semantic_left, semantic_right = sorted((first.id, third.id))
    edges = tuple(
        sorted(
            (
                DuplicateEdge(
                    left_id=exact_left,
                    right_id=exact_right,
                    evidence=DuplicateEvidence(
                        exact_match=True,
                        phash_distance=0,
                        cosine_similarity=1.0,
                    ),
                    classification=DuplicateClassification.EXACT,
                    decision=EdgeDecision.DEFINITE,
                    confidence=1.0,
                ),
                DuplicateEdge(
                    left_id=near_left,
                    right_id=near_right,
                    evidence=DuplicateEvidence(
                        phash_distance=2,
                        cosine_similarity=0.98,
                    ),
                    classification=DuplicateClassification.TRANSFORMED_DUPLICATE,
                    decision=EdgeDecision.DEFINITE,
                    confidence=0.96,
                ),
                DuplicateEdge(
                    left_id=semantic_left,
                    right_id=semantic_right,
                    evidence=DuplicateEvidence(cosine_similarity=0.91),
                    classification=DuplicateClassification.SEMANTIC_CANDIDATE,
                    decision=EdgeDecision.REVIEW,
                    confidence=0.91,
                ),
            ),
            key=lambda item: (item.left_id, item.right_id),
        )
    )
    exact_labels = tuple(sorted(("cat", second.label or "")))
    leakage_groups = tuple(
        sorted(
            (
                LeakageGroup(
                    family_id=exact_family.family_id,
                    member_ids=exact_members,
                    splits=(Split.TRAIN, Split.TEST),
                    boundaries=(SplitBoundary(left=Split.TRAIN, right=Split.TEST),),
                    labels=exact_labels,
                    evidence_types=(DuplicateClassification.EXACT,),
                    strongest_evidence=DuplicateClassification.EXACT,
                    label_conflict=True,
                ),
                LeakageGroup(
                    family_id=near_family.family_id,
                    member_ids=near_members,
                    splits=(Split.VAL, Split.TEST),
                    boundaries=(SplitBoundary(left=Split.VAL, right=Split.TEST),),
                    labels=("cat",),
                    evidence_types=(DuplicateClassification.TRANSFORMED_DUPLICATE,),
                    strongest_evidence=DuplicateClassification.TRANSFORMED_DUPLICATE,
                    label_conflict=False,
                ),
            ),
            key=lambda item: item.family_id,
        )
    )
    audit = AuditArtifact(
        metadata=_metadata("d" * 64),
        records=records,
        invalid_records=(
            ValidationIssue(
                path="test/broken.png",
                split=Split.TEST,
                label="<svg onload=alert(2)>",
                code=ValidationIssueCode.MALFORMED_IMAGE,
                message=(
                    r"decode failed at C:\private\patient.png "
                    "and /srv/private/patient.png "
                    "<img src=x onerror=alert(1)>"
                ),
            ),
        ),
        edges=edges,
        families=tuple(sorted((exact_family, near_family), key=lambda item: item.family_id)),
        leakage_groups=leakage_groups,
        label_conflicts=(
            LabelConflict(
                family_id=exact_family.family_id,
                member_ids=exact_members,
                labels=exact_labels,
                kind=ConflictKind.EXACT_DUPLICATE,
            ),
        ),
        summary=AuditSummary(
            valid_image_count=4,
            invalid_image_count=1,
            leakage_group_count=2,
            contaminated_image_count=3,
            evaluation_image_count=3,
            contaminated_evaluation_fraction=1.0,
            exact_leakage_group_count=1,
            perceptual_leakage_group_count=1,
            embedding_only_review_count=1,
            cross_label_conflict_count=1,
        ),
    )

    family_by_id = {
        member_id: family.family_id
        for family in (exact_family, near_family)
        for member_id in family.member_ids
    }
    repaired_split = {
        first.id: Split.TRAIN,
        second.id: Split.TRAIN,
        third.id: Split.TEST,
        fourth.id: Split.TEST,
    }
    repair = RepairArtifact(
        metadata=_metadata("d" * 64),
        requested_ratios=(
            SplitRatio(split=Split.TRAIN, ratio=0.5),
            SplitRatio(split=Split.VAL, ratio=0.25),
            SplitRatio(split=Split.TEST, ratio=0.25),
        ),
        integer_targets=((Split.TRAIN, 2), (Split.VAL, 1), (Split.TEST, 1)),
        assignments=tuple(
            RepairAssignment(
                record_id=record.id,
                family_id=family_by_id[record.id],
                original_split=record.split,
                repaired_split=repaired_split[record.id],
            )
            for record in records
        ),
        split_size_weight=1.0,
        class_balance_weight=1.0,
        random_seed=7,
        local_improvement_iterations=20,
        repaired_manifest_sha256="4" * 64,
        before_split_statistics=(
            SplitStatistics(split=Split.TRAIN, image_count=1, class_counts=(("cat", 1),)),
            SplitStatistics(split=Split.VAL, image_count=1, class_counts=(("cat", 1),)),
            SplitStatistics(
                split=Split.TEST,
                image_count=2,
                class_counts=(("cat", 1), (second.label or "", 1)),
            ),
        ),
        after_split_statistics=(
            SplitStatistics(
                split=Split.TRAIN,
                image_count=2,
                class_counts=(("cat", 1), (second.label or "", 1)),
            ),
            SplitStatistics(split=Split.VAL, image_count=0),
            SplitStatistics(split=Split.TEST, image_count=2, class_counts=(("cat", 2),)),
        ),
        summary=RepairSummary(
            objective_value=0.15,
            split_size_error_before=0.25,
            split_size_error_after=0.125,
            class_divergence_before=0.2,
            class_divergence_after=0.1,
            definite_leakage_groups_before=2,
            definite_leakage_groups_after=0,
            moved_image_count=2,
            hard_group_invariant_satisfied=True,
        ),
    )
    detection = DetectionBenchmarkArtifact(
        metadata=_metadata("e" * 64),
        embedding_provenance=EmbeddingProvenance(
            backend="fake",
            model_identity="fake:sha256-expand-v1:seed=7:dimension=16:l2=float32-v1",
            preprocessing_version="rgb-direct-v1",
            device="cpu",
            is_synthetic=True,
        ),
        rows=(
            DetectionMetricRow(
                detector="phash_bktree",
                corruption_type="jpeg_recompression",
                threshold=2.0,
                metrics=BinaryMetrics.from_counts(1, 0, 1),
            ),
            DetectionMetricRow(
                detector="phash_bktree",
                corruption_type="jpeg_recompression",
                threshold=8.0,
                metrics=BinaryMetrics.from_counts(2, 1, 0),
            ),
            DetectionMetricRow(
                detector="synthetic_fake_embedding_cosine_not_dinov2",
                corruption_type="resize",
                threshold=0.95,
                metrics=BinaryMetrics.from_counts(1, 1, 1),
            ),
        ),
    )
    scaling = ScalingBenchmarkArtifact(
        metadata=_metadata("e" * 64),
        rows=(
            ScalingMetricRow(
                dataset_size=100,
                stage="phash_query",
                mode="bktree",
                duration_seconds=0.01,
                peak_memory_bytes=4096,
                memory_measurement_scope="python_allocations_via_tracemalloc",
            ),
            ScalingMetricRow(
                dataset_size=1000,
                stage="phash_query",
                mode="bktree",
                duration_seconds=0.08,
                peak_memory_bytes=8192,
                memory_measurement_scope="python_allocations_via_tracemalloc",
            ),
            ScalingMetricRow(
                dataset_size=100,
                stage="faiss_query",
                mode="hnsw_synthetic_embeddings_not_dinov2",
                duration_seconds=0.02,
                peak_memory_bytes=8192,
                memory_measurement_scope="python_allocations_via_tracemalloc",
                recall_at_k=0.9,
            ),
        ),
    )
    contamination_ground_truth = (
        TrainingContaminationGroundTruth(
            source_record_id=first.id,
            derived_record_id=second.id,
            source_train_position=0,
            contaminated_test_position=2,
            label=0,
            corruption="resize",
            source_sha256="a" * 64,
            derived_sha256="b" * 64,
        ),
        TrainingContaminationGroundTruth(
            source_record_id=third.id,
            derived_record_id=fourth.id,
            source_train_position=1,
            contaminated_test_position=3,
            label=1,
            corruption="resize",
            source_sha256="c" * 64,
            derived_sha256="d" * 64,
        ),
    )
    experiment_summary = TrainingExperimentSummary(
        dataset_source="provided_arrays",
        num_classes=2,
        corruption="resize",
        injected_family_count=2,
        sampling_seed=19,
        repair_seed=7,
        training_seeds=(7,),
        requested_device="cpu",
        resolved_device="cpu",
        ground_truth=contamination_ground_truth,
        ground_truth_sha256=canonical_sha256(
            tuple(item.model_dump(mode="json") for item in contamination_ground_truth)
        ),
        repair_plan_sha256="3" * 64,
        shared_clean_holdout_sha256="5" * 64,
        shared_clean_holdout_count=2,
        condition_summaries=(
            TrainingConditionSummary(
                condition=TrainingCondition.CONTAMINATED,
                split_manifest_sha256="1" * 64,
                train_image_count=4,
                validation_image_count=2,
                test_image_count=4,
                injected_derivative_test_count=2,
                non_injected_test_count=2,
            ),
            TrainingConditionSummary(
                condition=TrainingCondition.REPAIRED,
                split_manifest_sha256="2" * 64,
                train_image_count=4,
                validation_image_count=2,
                test_image_count=4,
                injected_derivative_test_count=0,
                non_injected_test_count=4,
            ),
        ),
        repair_summary=repair.summary,
        repair_split_size_weight=repair.split_size_weight,
        repair_class_balance_weight=repair.class_balance_weight,
        repair_local_iterations=repair.local_improvement_iterations,
        repair_warnings=(r"review C:\private\repair.txt <script>alert(3)</script>",),
    )
    contaminated_class_rows = (
        NamedAccuracy(name="class_0", metric=_accuracy(2, 2)),
        NamedAccuracy(name="class_1", metric=_accuracy(1, 2)),
    )
    repaired_class_rows = (
        NamedAccuracy(name="class_0", metric=_accuracy(1, 2)),
        NamedAccuracy(name="class_1", metric=_accuracy(1, 2)),
    )
    training = TrainingArtifact(
        metadata=_metadata("f" * 64, seed=7),
        summary=experiment_summary,
        runs=(
            TrainingRun(
                seed=7,
                condition=TrainingCondition.CONTAMINATED,
                resolved_device="cpu",
                split_manifest_sha256="1" * 64,
                train_accuracy=_accuracy(3, 4),
                validation_accuracy=_accuracy(1, 2),
                test_accuracy=_accuracy(3, 4),
                per_class_test_accuracy=contaminated_class_rows,
                contaminated_example_accuracy=_accuracy(2, 2),
                shared_clean_holdout_accuracy=_accuracy(1, 2),
                non_injected_test_accuracy=_accuracy(1, 2),
                duration_seconds=1.25,
            ),
            TrainingRun(
                seed=7,
                condition=TrainingCondition.REPAIRED,
                resolved_device="cpu",
                split_manifest_sha256="2" * 64,
                train_accuracy=_accuracy(3, 4),
                validation_accuracy=_accuracy(1, 2),
                test_accuracy=_accuracy(2, 4),
                per_class_test_accuracy=repaired_class_rows,
                contaminated_example_accuracy=None,
                shared_clean_holdout_accuracy=_accuracy(1, 2),
                non_injected_test_accuracy=_accuracy(2, 4),
                duration_seconds=1.1,
            ),
        ),
    )

    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    paths = {
        "audit_path": output_dir / "audit.json",
        "repair_path": output_dir / "repair.json",
        "detection_path": output_dir / "detection_benchmark.json",
        "scaling_path": output_dir / "scaling_benchmark.json",
        "training_path": output_dir / "training_results.json",
    }
    for key, artifact in zip(
        paths,
        (audit, repair, detection, scaling, training),
        strict=True,
    ):
        _write_json(paths[key], artifact)
    (output_dir / ".gitkeep").write_text("keep\n", encoding="utf-8")
    return ArtifactFixture(dataset_root=dataset_root, output_dir=output_dir, **paths)


def test_generate_full_local_report_from_validated_raw_artifacts(tmp_path: Path) -> None:
    fixture = _artifact_fixture(tmp_path)
    preserved = {
        path.name: path.read_bytes()
        for path in (
            fixture.audit_path,
            fixture.repair_path,
            fixture.detection_path,
            fixture.scaling_path,
            fixture.training_path,
        )
    }
    source_hashes = {
        path.relative_to(fixture.dataset_root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in fixture.dataset_root.rglob("*.png")
    }

    result = generate_report(
        fixture.audit_path,
        fixture.output_dir,
        repair_path=fixture.repair_path,
        detection_benchmark_path=fixture.detection_path,
        scaling_benchmark_path=fixture.scaling_path,
        training_results_path=fixture.training_path,
        dataset_root=fixture.dataset_root,
        max_thumbnails=3,
    )

    assert result.output_dir == fixture.output_dir.resolve()
    assert result.html_path.is_file()
    assert result.markdown_path.is_file()
    assert len(result.chart_paths) == 4
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in result.chart_paths)
    assert len(result.thumbnail_paths) == 3
    assert all(path.is_file() for path in result.thumbnail_paths)
    assert result.thumbnails_skipped == 0

    html = result.html_path.read_text(encoding="utf-8")
    markdown = result.markdown_path.read_text(encoding="utf-8")
    expected_sections = (
        "Summary",
        "Split statistics",
        "Class distribution",
        "Invalid files",
        "Exact duplicate groups",
        "Near-duplicate groups",
        "Cross-split leakage",
        "Label conflicts",
        "Similarity evidence",
        "Representative thumbnails",
        "Repair summary",
        "Detection benchmark",
        "Runtime scaling",
        "Evaluation-integrity experiment",
        "Provenance",
        "Privacy",
    )
    assert all(section in html for section in expected_sections)
    assert all(section in markdown for section in expected_sections)
    assert "Shared clean holdout accuracy (paired)" in html
    assert "All condition-test accuracy" in html
    assert "Non-injected condition-test accuracy" in html
    assert "Injected derivatives in actual test" in html
    assert "fair primary paired comparison" in html
    assert "no causal effect is inferred" in html
    assert "Clean-only accuracy" not in html
    assert "causal inflation" not in html.lower()
    repaired_row = next(
        row.split("</tr>", maxsplit=1)[0]
        for row in html.split("<tr>")
        if "<td>repaired</td><td>7</td><td>cpu</td>" in row
    )
    assert "<td>—</td>" in repaired_row
    training = TrainingArtifact.model_validate_json(fixture.training_path.read_bytes())
    assert training.summary.ground_truth_sha256 in html
    assert training.summary.repair_plan_sha256 in html
    assert training.summary.shared_clean_holdout_sha256 in html
    assert training.summary.ground_truth[0].source_record_id in html
    assert training.summary.ground_truth[0].derived_sha256 in html
    assert "Experiment design and repair evidence" in html
    assert "Condition composition" in html
    assert "Detector-independent injected-family ground truth" in html
    assert "Definite leakage groups (before → after)" in html
    assert "python_allocations_via_tracemalloc" in html
    assert "fake:sha256-expand-v1:seed=7:dimension=16:l2=float32-v1" in html
    assert "test-os" in html and "test-cpu" in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&lt;script&gt;alert(3)&lt;/script&gt;" in html
    assert "<script>" not in html
    assert "<img src=x" not in html
    assert r"C:\private\patient.png" not in html
    assert r"C:\private\repair.txt" not in html
    assert "/srv/private/patient.png" not in html
    assert str(tmp_path) not in html
    assert str(tmp_path) not in markdown
    assert "http://" not in html and "https://" not in html
    assert "http://" not in markdown and "https://" not in markdown
    assert 'src="/' not in html and 'src="../' not in html
    assert "file:" not in html.lower()
    assert 'src="report_thumbnails/' in html
    for chart_name in (
        "detection_pr_curve.png",
        "runtime_scaling.png",
        "split_distribution.png",
        "evaluation_comparison.png",
    ):
        assert f'src="{chart_name}"' in html
        assert f"]({chart_name})" in markdown
    assert "Chart bars are means" in markdown
    assert "Shared clean holdout accuracy (paired)" in markdown
    assert "Injected derivatives in actual test" in markdown
    assert "no causal effect is inferred" in markdown

    assert (fixture.output_dir / ".gitkeep").read_text(encoding="utf-8") == "keep\n"
    assert preserved == {
        path.name: path.read_bytes()
        for path in (
            fixture.audit_path,
            fixture.repair_path,
            fixture.detection_path,
            fixture.scaling_path,
            fixture.training_path,
        )
    }
    assert source_hashes == {
        path.relative_to(fixture.dataset_root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in fixture.dataset_root.rglob("*.png")
    }
    assert not tuple(tmp_path.glob(".splitguard-report-*"))


def test_no_thumbnails_replaces_only_owned_thumbnail_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _artifact_fixture(tmp_path)
    first = generate_report(
        fixture.audit_path,
        fixture.output_dir,
        dataset_root=fixture.dataset_root,
    )
    assert first.thumbnail_paths
    unrelated = fixture.output_dir / "unrelated.txt"
    unrelated.write_text("preserve", encoding="utf-8")

    def unexpected_source_read(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("source images must not be opened or hashed")

    monkeypatch.setattr(reporting_module, "_sha256_file", unexpected_source_read)
    monkeypatch.setattr(reporting_module.Image, "open", unexpected_source_read)

    result = generate_report(
        fixture.audit_path,
        fixture.output_dir,
        dataset_root=tmp_path / "missing-private-dataset",
        no_thumbnails=True,
    )

    assert result.thumbnail_paths == ()
    assert result.thumbnails_skipped == 0
    assert (fixture.output_dir / "report_thumbnails").is_dir()
    assert not tuple((fixture.output_dir / "report_thumbnails").iterdir())
    assert unrelated.read_text(encoding="utf-8") == "preserve"
    assert "no source images were read" in result.html_path.read_text(encoding="utf-8")


def test_thumbnail_generation_rejects_changed_source_bytes_and_removes_stale_assets(
    tmp_path: Path,
) -> None:
    fixture = _artifact_fixture(tmp_path)
    initial = generate_report(
        fixture.audit_path,
        fixture.output_dir,
        dataset_root=fixture.dataset_root,
        max_thumbnails=4,
    )
    assert len(initial.thumbnail_paths) == 4

    for source in fixture.dataset_root.rglob("*.png"):
        source.write_bytes(source.read_bytes() + b"changed-after-audit")

    result = generate_report(
        fixture.audit_path,
        fixture.output_dir,
        dataset_root=fixture.dataset_root,
        max_thumbnails=4,
    )

    assert result.thumbnail_paths == ()
    assert result.thumbnails_skipped == 4
    assert not tuple((fixture.output_dir / "report_thumbnails").iterdir())


def test_late_publication_failure_restores_every_owned_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _artifact_fixture(tmp_path)
    generate_report(
        fixture.audit_path,
        fixture.output_dir,
        repair_path=fixture.repair_path,
        detection_benchmark_path=fixture.detection_path,
        scaling_benchmark_path=fixture.scaling_path,
        training_results_path=fixture.training_path,
        dataset_root=fixture.dataset_root,
        max_thumbnails=3,
    )
    unrelated = fixture.output_dir / "unrelated.txt"
    unrelated.write_text("preserve", encoding="utf-8")
    before = {
        path.relative_to(fixture.output_dir).as_posix(): path.read_bytes()
        for path in fixture.output_dir.rglob("*")
        if path.is_file()
    }

    original_replace = reporting_module.os.replace
    failed = False

    def fail_once_during_publish(source: object, destination: object) -> None:
        nonlocal failed
        source_path = Path(source)  # type: ignore[arg-type]
        if (
            not failed
            and source_path.name == "runtime_scaling.png"
            and source_path.parent.name.startswith(".splitguard-report-")
            and not source_path.parent.name.startswith(".splitguard-report-backup-")
        ):
            failed = True
            raise OSError("simulated late publication failure")
        original_replace(source, destination)  # type: ignore[arg-type]

    monkeypatch.setattr(reporting_module.os, "replace", fail_once_during_publish)

    with pytest.raises(ReportInputError, match="without exposing source paths"):
        generate_report(
            fixture.audit_path,
            fixture.output_dir,
            no_thumbnails=True,
        )

    assert failed
    assert before == {
        path.relative_to(fixture.output_dir).as_posix(): path.read_bytes()
        for path in fixture.output_dir.rglob("*")
        if path.is_file()
    }
    assert unrelated.read_text(encoding="utf-8") == "preserve"
    assert not tuple(tmp_path.glob(".splitguard-report-*"))


def test_report_rejects_invalid_json_without_leaking_or_replacing_files(
    tmp_path: Path,
) -> None:
    invalid = tmp_path / "private" / "audit.json"
    invalid.parent.mkdir()
    invalid.write_text(
        '{"artifact_type":"audit","source":"C:\\\\private\\\\patient.png"}',
        encoding="utf-8",
    )
    output = tmp_path / "artifacts"
    output.mkdir()
    existing = output / "report.html"
    existing.write_text("existing", encoding="utf-8")

    with pytest.raises(ReportInputError, match="strict validation") as caught:
        generate_report(invalid, output)

    assert str(tmp_path) not in str(caught.value)
    assert existing.read_text(encoding="utf-8") == "existing"
    assert not (output / "report.md").exists()


def test_report_rejects_mismatched_repair_and_dataset_overlap(tmp_path: Path) -> None:
    fixture = _artifact_fixture(tmp_path)
    repair = RepairArtifact.model_validate_json(fixture.repair_path.read_bytes())
    mismatched = repair.model_copy(update={"metadata": _metadata("9" * 64)})
    mismatch_path = fixture.output_dir / "mismatched-repair.json"
    _write_json(mismatch_path, mismatched)

    with pytest.raises(ReportInputError, match="different datasets"):
        generate_report(
            fixture.audit_path,
            fixture.output_dir,
            repair_path=mismatch_path,
            no_thumbnails=True,
        )
    with pytest.raises(ReportInputError, match="must not overlap"):
        generate_report(
            fixture.audit_path,
            fixture.dataset_root / "reports",
            dataset_root=fixture.dataset_root,
            no_thumbnails=True,
        )
