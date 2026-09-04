"""Privacy-safe static reports generated only from validated result artifacts."""

from __future__ import annotations

import hashlib
import html
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TypeVar

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ValidationError

from splitguard.schemas import (
    AuditArtifact,
    DetectionBenchmarkArtifact,
    DuplicateClassification,
    RepairArtifact,
    ScalingBenchmarkArtifact,
    Split,
    TrainingArtifact,
)

_ArtifactT = TypeVar("_ArtifactT", bound=BaseModel)
_CHART_NAMES = (
    "detection_pr_curve.png",
    "runtime_scaling.png",
    "split_distribution.png",
    "evaluation_comparison.png",
)
_REPORT_FILE_NAMES = (*_CHART_NAMES, "report.md", "report.html")
_THUMBNAIL_DIRECTORY = "report_thumbnails"
_EVALUATION_NOTE = (
    "The shared clean holdout is the fair primary paired comparison because the same "
    "records are scored under both conditions. All condition-test, non-injected "
    "condition-test, and injected-derivative metrics describe condition-specific "
    "cohorts; no causal effect is inferred from their differences. A dash means that "
    "no injected derivative remained in that condition's test split."
)
_SPLIT_ORDER = {split: index for index, split in enumerate(Split)}
_WINDOWS_ABSOLUTE_PATH = re.compile(r"(?i)(?:[a-z]:[\\/]|\\\\[^\\/\s]+[\\/][^\\/\s]+)[^\s<>\"']*")
_POSIX_ABSOLUTE_PATH = re.compile(r"(^|[\s(\[=:\"'])/(?:[^\s<>\"']+/)*[^\s<>\"']+")


class ReportInputError(ValueError):
    """Raised when report inputs or destinations fail safe validation."""


@dataclass(frozen=True, slots=True)
class ReportResult:
    """Paths and counts for one successfully published static report."""

    output_dir: Path
    html_path: Path
    markdown_path: Path
    chart_paths: tuple[Path, ...]
    thumbnail_paths: tuple[Path, ...]
    thumbnails_skipped: int


@dataclass(frozen=True, slots=True)
class _Artifacts:
    audit: AuditArtifact
    repair: RepairArtifact | None
    detection: DetectionBenchmarkArtifact | None
    scaling: ScalingBenchmarkArtifact | None
    training: TrainingArtifact | None


@dataclass(frozen=True, slots=True)
class _Thumbnail:
    record_id: str
    path: str
    split: str
    label: str
    asset_path: str


def _load_artifact(path: str | Path, model: type[_ArtifactT], kind: str) -> _ArtifactT:
    try:
        document = Path(path).read_bytes()
    except OSError as exc:
        raise ReportInputError(f"could not read the {kind} artifact") from exc
    try:
        return model.model_validate_json(document)
    except (ValidationError, ValueError) as exc:
        raise ReportInputError(f"the {kind} artifact failed strict validation") from exc


def _load_optional_artifact(
    path: str | Path | None,
    model: type[_ArtifactT],
    kind: str,
) -> _ArtifactT | None:
    return None if path is None else _load_artifact(path, model, kind)


def _safe_text(value: object) -> str:
    text = str(value).replace("\x00", "").replace("\r", " ").replace("\n", " ")
    text = _WINDOWS_ABSOLUTE_PATH.sub("[private path]", text)
    return _POSIX_ABSOLUTE_PATH.sub(
        lambda match: f"{match.group(1)}[private path]",
        text,
    )


def _html_text(value: object) -> str:
    return html.escape(_safe_text(value), quote=True)


def _markdown_text(value: object) -> str:
    text = _safe_text(value)
    return (
        text.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("`", "\\`")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _metric(value: float | int | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.6f}"


def _html_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    materialized = tuple(tuple(row) for row in rows)
    header = "".join(f'<th scope="col">{_html_text(item)}</th>' for item in headers)
    if materialized:
        body = "".join(
            "<tr>" + "".join(f"<td>{_html_text(cell)}</td>" for cell in row) + "</tr>"
            for row in materialized
        )
    else:
        body = f'<tr><td class="empty" colspan="{len(headers)}">None recorded.</td></tr>'
    return (
        '<div class="table-wrap"><table><thead><tr>'
        f"{header}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    materialized = tuple(tuple(row) for row in rows)
    lines = [
        "| " + " | ".join(_markdown_text(item) for item in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    if materialized:
        lines.extend(
            "| " + " | ".join(_markdown_text(cell) for cell in row) + " |" for row in materialized
        )
    else:
        lines.append("| " + " | ".join(("None recorded.", *("" for _ in headers[1:]))) + " |")
    return "\n".join(lines)


def _split_rows(audit: AuditArtifact) -> tuple[tuple[object, ...], ...]:
    counts = Counter(record.split for record in audit.records)
    return tuple((split.value, counts[split]) for split in Split if counts[split] > 0)


def _class_rows(audit: AuditArtifact) -> tuple[tuple[object, ...], ...]:
    counts = Counter(
        (record.split, record.label if record.label is not None else "(unlabeled)")
        for record in audit.records
    )
    return tuple(
        (split.value, label, count)
        for (split, label), count in sorted(
            counts.items(),
            key=lambda item: (_SPLIT_ORDER[item[0][0]], item[0][1]),
        )
    )


def _family_rows(
    audit: AuditArtifact,
    classification: DuplicateClassification,
) -> tuple[tuple[object, ...], ...]:
    member_to_family = {
        member_id: family.family_id for family in audit.families for member_id in family.member_ids
    }
    evidence_by_family: dict[str, set[DuplicateClassification]] = defaultdict(set)
    for edge in audit.edges:
        family_id = member_to_family.get(edge.left_id)
        if family_id is not None and family_id == member_to_family.get(edge.right_id):
            evidence_by_family[family_id].add(edge.classification)
    records = {record.id: record for record in audit.records}
    rows: list[tuple[object, ...]] = []
    for family in audit.families:
        if classification not in evidence_by_family[family.family_id]:
            continue
        members = tuple(records[item] for item in family.member_ids if item in records)
        splits = sorted({record.split for record in members}, key=_SPLIT_ORDER.__getitem__)
        labels = sorted({record.label for record in members if record.label is not None})
        evidence = sorted(item.value for item in evidence_by_family[family.family_id])
        rows.append(
            (
                family.family_id,
                len(family.member_ids),
                ", ".join(split.value for split in splits),
                ", ".join(labels) if labels else "(unlabeled)",
                ", ".join(evidence),
                ", ".join(family.member_ids),
            )
        )
    return tuple(rows)


def _leakage_rows(audit: AuditArtifact) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            group.family_id,
            ", ".join(f"{item.left.value}-{item.right.value}" for item in group.boundaries),
            group.strongest_evidence.value,
            "yes" if group.label_conflict else "no",
            ", ".join(group.labels) if group.labels else "(unlabeled)",
            ", ".join(group.member_ids),
        )
        for group in audit.leakage_groups
    )


def _conflict_rows(audit: AuditArtifact) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            conflict.family_id,
            conflict.kind.value,
            ", ".join(conflict.labels),
            ", ".join(conflict.member_ids),
        )
        for conflict in audit.label_conflicts
    )


def _evidence_rows(audit: AuditArtifact) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            edge.left_id,
            edge.right_id,
            edge.classification.value,
            edge.decision.value,
            "yes" if edge.evidence.exact_match else "no",
            _metric(edge.evidence.phash_distance),
            _metric(edge.evidence.cosine_similarity),
            _metric(edge.confidence),
        )
        for edge in audit.edges
    )


def _repair_split_rows(repair: RepairArtifact | None) -> tuple[tuple[object, ...], ...]:
    if repair is None:
        return ()
    before = {item.split: item for item in repair.before_split_statistics}
    after = {item.split: item for item in repair.after_split_statistics}
    return tuple(
        (
            split.value,
            before[split].image_count if split in before else 0,
            after[split].image_count if split in after else 0,
            dict(before[split].class_counts) if split in before else {},
            dict(after[split].class_counts) if split in after else {},
        )
        for split in Split
        if split in before or split in after
    )


def _detection_rows(
    detection: DetectionBenchmarkArtifact | None,
) -> tuple[tuple[object, ...], ...]:
    if detection is None:
        return ()
    return tuple(
        (
            row.detector,
            row.corruption_type,
            _metric(row.threshold),
            _metric(row.metrics.precision),
            _metric(row.metrics.recall),
            _metric(row.metrics.f1),
            row.metrics.true_positives,
            row.metrics.false_positives,
            row.metrics.false_negatives,
        )
        for row in detection.rows
    )


def _detection_provenance_rows(
    detection: DetectionBenchmarkArtifact | None,
) -> tuple[tuple[object, ...], ...]:
    if detection is None:
        return ()
    provenance = detection.embedding_provenance
    return (
        ("Backend", provenance.backend),
        ("Model identity", provenance.model_identity),
        ("Immutable model revision", provenance.model_revision or "not applicable"),
        ("Preprocessing", provenance.preprocessing_version),
        ("Resolved device", provenance.device),
        ("Synthetic evidence", "yes" if provenance.is_synthetic else "no"),
    )


def _scaling_rows(
    scaling: ScalingBenchmarkArtifact | None,
) -> tuple[tuple[object, ...], ...]:
    if scaling is None:
        return ()
    return tuple(
        (
            row.dataset_size,
            row.stage,
            row.mode,
            _metric(row.duration_seconds),
            _metric(row.peak_memory_bytes),
            row.memory_measurement_scope or "—",
            _metric(row.recall_at_k),
        )
        for row in scaling.rows
    )


def _training_rows(training: TrainingArtifact | None) -> tuple[tuple[object, ...], ...]:
    if training is None:
        return ()
    return tuple(
        (
            run.condition.value,
            run.seed,
            run.resolved_device,
            _metric(run.train_accuracy.accuracy),
            _metric(run.validation_accuracy.accuracy),
            _metric(run.shared_clean_holdout_accuracy.accuracy),
            _metric(run.test_accuracy.accuracy),
            _metric(run.non_injected_test_accuracy.accuracy),
            _metric(
                run.contaminated_example_accuracy.accuracy
                if run.contaminated_example_accuracy is not None
                else None
            ),
            _metric(run.duration_seconds),
            run.split_manifest_sha256,
        )
        for run in training.runs
    )


def _training_summary_rows(
    training: TrainingArtifact | None,
) -> tuple[tuple[object, ...], ...]:
    if training is None:
        return ()
    summary = training.summary
    repair = summary.repair_summary
    return (
        ("Dataset", summary.dataset_name),
        ("Dataset source", summary.dataset_source),
        ("Classes", summary.num_classes),
        ("Injected transformation", summary.corruption),
        ("Injected families", summary.injected_family_count),
        ("Native selected-set duplicates", summary.baseline_native_duplicate_count),
        ("Sampling seed", summary.sampling_seed),
        ("Repair seed", summary.repair_seed),
        ("Training seeds", ", ".join(str(seed) for seed in summary.training_seeds)),
        ("Requested device", summary.requested_device),
        ("Resolved device", summary.resolved_device),
        ("Ground-truth SHA-256", summary.ground_truth_sha256),
        ("Repair-plan SHA-256", summary.repair_plan_sha256),
        ("Shared clean holdout SHA-256", summary.shared_clean_holdout_sha256),
        ("Shared clean holdout images", summary.shared_clean_holdout_count),
        ("Repair split-size weight", _metric(summary.repair_split_size_weight)),
        ("Repair class-balance weight", _metric(summary.repair_class_balance_weight)),
        ("Repair local iterations", summary.repair_local_iterations),
        (
            "Repair split-size error (before → after)",
            f"{_metric(repair.split_size_error_before)} → {_metric(repair.split_size_error_after)}",
        ),
        (
            "Repair class divergence (before → after)",
            f"{_metric(repair.class_divergence_before)} → {_metric(repair.class_divergence_after)}",
        ),
        (
            "Definite leakage groups (before → after)",
            f"{repair.definite_leakage_groups_before} → {repair.definite_leakage_groups_after}",
        ),
        ("Repair objective", _metric(repair.objective_value)),
        ("Repair moved images", repair.moved_image_count),
        (
            "Hard family invariant",
            "satisfied" if repair.hard_group_invariant_satisfied else "failed",
        ),
        ("Repair warnings", "; ".join(summary.repair_warnings) or "none"),
    )


def _training_condition_rows(
    training: TrainingArtifact | None,
) -> tuple[tuple[object, ...], ...]:
    if training is None:
        return ()
    return tuple(
        (
            condition.condition.value,
            condition.split_manifest_sha256,
            condition.train_image_count,
            condition.validation_image_count,
            condition.test_image_count,
            condition.non_injected_test_count,
            condition.injected_derivative_test_count,
        )
        for condition in training.summary.condition_summaries
    )


def _training_ground_truth_rows(
    training: TrainingArtifact | None,
) -> tuple[tuple[object, ...], ...]:
    if training is None:
        return ()
    return tuple(
        (
            row.source_record_id,
            row.derived_record_id,
            row.source_train_position,
            row.contaminated_test_position,
            row.label,
            row.corruption,
            row.source_sha256,
            row.derived_sha256,
            row.expected_relationship,
            "yes" if row.repair_requires_same_split else "no",
        )
        for row in training.summary.ground_truth
    )


def _provenance_rows(artifacts: _Artifacts) -> tuple[tuple[object, ...], ...]:
    supplied: tuple[tuple[str, BaseModel | None], ...] = (
        ("audit", artifacts.audit),
        ("repair", artifacts.repair),
        ("detection benchmark", artifacts.detection),
        ("scaling benchmark", artifacts.scaling),
        ("training results", artifacts.training),
    )
    rows: list[tuple[object, ...]] = []
    for name, artifact in supplied:
        if artifact is None:
            continue
        metadata = artifact.metadata  # type: ignore[attr-defined]
        rows.append(
            (
                name,
                artifact.schema_version,  # type: ignore[attr-defined]
                metadata.timestamp.isoformat(),
                metadata.configuration_sha256,
                metadata.dataset_manifest_sha256,
                ", ".join(str(seed) for seed in metadata.random_seeds) or "—",
                metadata.git_commit_sha or "unavailable",
                (
                    "dirty"
                    if metadata.git_dirty is True
                    else "clean"
                    if metadata.git_dirty is False
                    else "unavailable"
                ),
                metadata.python_version,
                metadata.os,
                metadata.cpu,
                (
                    metadata.gpu_model
                    or ("CUDA available; GPU unreported" if metadata.cuda_available else "CPU only")
                ),
                ", ".join(
                    f"{package.name}=={package.version}" for package in metadata.package_versions
                )
                or "none recorded",
            )
        )
    return tuple(rows)


def _representative_ids(audit: AuditArtifact, maximum: int) -> tuple[str, ...]:
    ordered: list[str] = []
    for group in audit.leakage_groups:
        ordered.extend(group.member_ids)
    for conflict in audit.label_conflicts:
        ordered.extend(conflict.member_ids)
    for family in audit.families:
        if family.edge_count > 0:
            ordered.extend(family.member_ids)
    ordered.extend(record.id for record in audit.records)
    known = {record.id for record in audit.records}
    return tuple(dict.fromkeys(item for item in ordered if item in known))[:maximum]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_thumbnails(
    audit: AuditArtifact,
    staging_directory: Path,
    dataset_root: Path | None,
    *,
    no_thumbnails: bool,
    thumbnail_size: int,
    max_thumbnails: int,
) -> tuple[tuple[_Thumbnail, ...], int, str]:
    thumbnail_directory = staging_directory / _THUMBNAIL_DIRECTORY
    thumbnail_directory.mkdir()
    if no_thumbnails:
        return (), 0, "Disabled for this report; no source images were read."
    if dataset_root is None:
        return (), 0, "Unavailable because no explicit dataset root was supplied."

    root = dataset_root.resolve(strict=True)
    records = {record.id: record for record in audit.records}
    thumbnails: list[_Thumbnail] = []
    skipped = 0
    for record_id in _representative_ids(audit, max_thumbnails):
        record = records[record_id]
        try:
            source = (root / Path(*PurePosixPath(record.path).parts)).resolve(strict=True)
            if not source.is_relative_to(root) or not source.is_file():
                raise OSError
            if _sha256_file(source) != record.byte_sha256:
                raise OSError
            with Image.open(source) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                image.thumbnail((thumbnail_size, thumbnail_size), Image.Resampling.LANCZOS)
                asset_name = f"{record.id}.png"
                image.save(thumbnail_directory / asset_name, format="PNG")
        except (OSError, ValueError, UnidentifiedImageError):
            skipped += 1
            continue
        thumbnails.append(
            _Thumbnail(
                record_id=record.id,
                path=record.path,
                split=record.split.value,
                label=record.label if record.label is not None else "(unlabeled)",
                asset_path=f"{_THUMBNAIL_DIRECTORY}/{asset_name}",
            )
        )
    status = f"Generated {len(thumbnails)} local thumbnails; skipped {skipped}."
    return tuple(thumbnails), skipped, status


def _figure(title: str, x_label: str, y_label: str) -> tuple[Figure, object]:
    figure = Figure(figsize=(8.4, 4.8), dpi=120, facecolor="#fbfaf7")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.set_title(title, color="#18232b", weight="bold")
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.grid(alpha=0.2)
    return figure, axes


def _empty_chart(axes: object, message: str) -> None:
    axes.text(  # type: ignore[attr-defined]
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=axes.transAxes,  # type: ignore[attr-defined]
        color="#59636b",
    )


def _save_figure(figure: Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(
        path,
        format="png",
        dpi=120,
        metadata={"Software": "SplitGuard Vision"},
    )
    figure.clear()


def _chart_detection(
    detection: DetectionBenchmarkArtifact | None,
    path: Path,
) -> None:
    figure, axes = _figure("Detection precision-recall", "Recall", "Precision")
    if detection is None or not detection.rows:
        _empty_chart(axes, "Detection benchmark artifact not supplied")
    else:
        groups: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
        for row in detection.rows:
            groups[(row.detector, row.corruption_type)].append(
                (row.metrics.recall, row.metrics.precision)
            )
        for (detector, corruption), points in sorted(groups.items()):
            ordered = sorted(points)
            label = _safe_text(f"{detector} · {corruption}")[:90]
            axes.plot(  # type: ignore[attr-defined]
                [point[0] for point in ordered],
                [point[1] for point in ordered],
                marker="o",
                linewidth=1.4,
                label=label,
            )
        axes.set_xlim(-0.02, 1.02)  # type: ignore[attr-defined]
        axes.set_ylim(-0.02, 1.02)  # type: ignore[attr-defined]
        axes.legend(fontsize=7, loc="best")  # type: ignore[attr-defined]
    _save_figure(figure, path)


def _chart_scaling(
    scaling: ScalingBenchmarkArtifact | None,
    path: Path,
) -> None:
    figure, axes = _figure("Runtime scaling", "Dataset size", "Duration (seconds)")
    if scaling is None or not scaling.rows:
        _empty_chart(axes, "Scaling benchmark artifact not supplied")
    else:
        groups: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
        for row in scaling.rows:
            groups[(row.stage, row.mode)].append((row.dataset_size, row.duration_seconds))
        for (stage, mode), points in sorted(groups.items()):
            ordered = sorted(points)
            axes.plot(  # type: ignore[attr-defined]
                [point[0] for point in ordered],
                [point[1] for point in ordered],
                marker="o",
                linewidth=1.5,
                label=_safe_text(f"{stage} · {mode}")[:90],
            )
        axes.legend(fontsize=8, loc="best")  # type: ignore[attr-defined]
    _save_figure(figure, path)


def _chart_splits(audit: AuditArtifact, repair: RepairArtifact | None, path: Path) -> None:
    figure, axes = _figure("Split distribution", "Split", "Images")
    splits = (Split.TRAIN, Split.VAL, Split.TEST, Split.CUSTOM)
    if repair is None:
        counts = Counter(record.split for record in audit.records)
        shown = tuple(split for split in splits if counts[split] > 0)
        axes.bar(  # type: ignore[attr-defined]
            [split.value for split in shown],
            [counts[split] for split in shown],
            color="#315d73",
            label="audited",
        )
    else:
        before = {item.split: item.image_count for item in repair.before_split_statistics}
        after = {item.split: item.image_count for item in repair.after_split_statistics}
        shown = tuple(split for split in splits if split in before or split in after)
        positions = list(range(len(shown)))
        width = 0.36
        axes.bar(  # type: ignore[attr-defined]
            [position - width / 2 for position in positions],
            [before.get(split, 0) for split in shown],
            width,
            color="#8aa1ad",
            label="before",
        )
        axes.bar(  # type: ignore[attr-defined]
            [position + width / 2 for position in positions],
            [after.get(split, 0) for split in shown],
            width,
            color="#b7663f",
            label="after",
        )
        axes.set_xticks(positions, [split.value for split in shown])  # type: ignore[attr-defined]
    axes.legend(loc="best")  # type: ignore[attr-defined]
    _save_figure(figure, path)


def _chart_evaluation(training: TrainingArtifact | None, path: Path) -> None:
    figure, axes = _figure(
        "Evaluation comparison",
        "Shared holdout is the primary paired cohort",
        "Accuracy",
    )
    if training is None or not training.runs:
        _empty_chart(axes, "Training-results artifact not supplied")
        _save_figure(figure, path)
        return

    metric_names = (
        "Shared clean\nholdout",
        "All condition\ntest",
        "Non-injected\ncondition test",
        "Injected derivatives\nin test",
    )
    conditions = sorted({run.condition.value for run in training.runs})
    colors = ("#315d73", "#b7663f", "#718355", "#725b83")
    width = 0.72 / max(len(conditions), 1)
    positions = list(range(len(metric_names)))
    for condition_index, condition in enumerate(conditions):
        condition_runs = [run for run in training.runs if run.condition.value == condition]
        series = (
            [run.shared_clean_holdout_accuracy.accuracy for run in condition_runs],
            [run.test_accuracy.accuracy for run in condition_runs],
            [run.non_injected_test_accuracy.accuracy for run in condition_runs],
            [
                run.contaminated_example_accuracy.accuracy
                for run in condition_runs
                if run.contaminated_example_accuracy is not None
            ],
        )
        x_values = [position - 0.36 + width / 2 + condition_index * width for position in positions]
        available = tuple(
            (x_value, sum(values) / len(values))
            for x_value, values in zip(x_values, series, strict=True)
            if values
        )
        if available:
            axes.bar(  # type: ignore[attr-defined]
                [item[0] for item in available],
                [item[1] for item in available],
                width,
                color=colors[condition_index % len(colors)],
                alpha=0.8,
                label=condition,
            )
        for x_value, values in zip(x_values, series, strict=True):
            if not values:
                axes.text(  # type: ignore[attr-defined]
                    x_value,
                    0.025,
                    "N/A",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=7,
                    color="#59636b",
                )
                continue
            axes.scatter(  # type: ignore[attr-defined]
                [x_value] * len(values),
                values,
                color="#172229",
                s=12,
                zorder=3,
            )
    axes.set_xticks(positions, metric_names)  # type: ignore[attr-defined]
    axes.set_ylim(0.0, 1.0)  # type: ignore[attr-defined]
    axes.legend(loc="best")  # type: ignore[attr-defined]
    _save_figure(figure, path)


def _render_markdown(
    artifacts: _Artifacts,
    thumbnails: tuple[_Thumbnail, ...],
    thumbnail_status: str,
) -> str:
    audit = artifacts.audit
    summary_rows = (
        ("Valid images", audit.summary.valid_image_count),
        ("Invalid images", audit.summary.invalid_image_count),
        ("Definite leakage groups", audit.summary.leakage_group_count),
        ("Contaminated evaluation images", audit.summary.contaminated_image_count),
        (
            "Contaminated evaluation fraction",
            _metric(audit.summary.contaminated_evaluation_fraction),
        ),
        ("Semantic-only review candidates", audit.summary.embedding_only_review_count),
        ("Cross-label conflicts", audit.summary.cross_label_conflict_count),
    )
    invalid_rows = tuple(
        (
            issue.record_id or "—",
            issue.path or "—",
            issue.split.value if issue.split is not None else "—",
            issue.label or "—",
            issue.code.value,
            issue.message,
        )
        for issue in audit.invalid_records
    )
    repair_summary: tuple[tuple[object, ...], ...] = ()
    if artifacts.repair is not None:
        summary = artifacts.repair.summary
        repair_summary = (
            ("Objective value", _metric(summary.objective_value)),
            ("Split-size error before", _metric(summary.split_size_error_before)),
            ("Split-size error after", _metric(summary.split_size_error_after)),
            ("Class divergence before", _metric(summary.class_divergence_before)),
            ("Class divergence after", _metric(summary.class_divergence_after)),
            ("Definite leakage groups before", summary.definite_leakage_groups_before),
            ("Definite leakage groups after", summary.definite_leakage_groups_after),
            ("Moved images", summary.moved_image_count),
            (
                "Hard family invariant",
                "satisfied" if summary.hard_group_invariant_satisfied else "failed",
            ),
        )

    thumbnail_lines = [f"{thumbnail_status}\n"]
    thumbnail_lines.extend(
        f"![{_markdown_text(item.record_id)}]({item.asset_path})  \n"
        f"`{_markdown_text(item.record_id)}` · {_markdown_text(item.split)} · "
        f"{_markdown_text(item.label)} · `{_markdown_text(item.path)}`"
        for item in thumbnails
    )
    return (
        "\n\n".join(
            (
                "# SplitGuard Vision integrity report\n\n"
                "Generated locally from strict JSON artifacts. Display values are rounded; "
                "source JSON "
                "retains full precision. Semantic-only neighbors remain review candidates.",
                "## Summary\n\n" + _markdown_table(("Metric", "Value"), summary_rows),
                "## Split statistics\n\n"
                + _markdown_table(("Split", "Images"), _split_rows(audit)),
                "### Class distribution\n\n"
                + _markdown_table(("Split", "Class", "Images"), _class_rows(audit)),
                "## Invalid files\n\n"
                + _markdown_table(
                    ("Record", "Relative path", "Split", "Label", "Code", "Message"),
                    invalid_rows,
                ),
                "## Exact duplicate groups\n\n"
                + _markdown_table(
                    ("Family", "Members", "Splits", "Labels", "Evidence", "Member IDs"),
                    _family_rows(audit, DuplicateClassification.EXACT),
                ),
                "## Near-duplicate groups\n\n"
                + _markdown_table(
                    ("Family", "Members", "Splits", "Labels", "Evidence", "Member IDs"),
                    _family_rows(audit, DuplicateClassification.TRANSFORMED_DUPLICATE),
                ),
                "## Cross-split leakage\n\n"
                + _markdown_table(
                    (
                        "Family",
                        "Boundaries",
                        "Strongest evidence",
                        "Label conflict",
                        "Labels",
                        "Members",
                    ),
                    _leakage_rows(audit),
                ),
                "## Label conflicts\n\n"
                + _markdown_table(("Family", "Kind", "Labels", "Members"), _conflict_rows(audit)),
                "## Similarity evidence\n\n"
                + _markdown_table(
                    (
                        "Left",
                        "Right",
                        "Classification",
                        "Decision",
                        "Exact",
                        "pHash distance",
                        "Cosine",
                        "Confidence",
                    ),
                    _evidence_rows(audit),
                ),
                "## Representative thumbnails\n\n" + "\n\n".join(thumbnail_lines),
                "## Repair summary\n\n"
                + (
                    _markdown_table(("Metric", "Value"), repair_summary)
                    if artifacts.repair is not None
                    else "Repair artifact not supplied."
                ),
                "### Repair split distribution\n\n"
                + _markdown_table(
                    ("Split", "Before", "After", "Classes before", "Classes after"),
                    _repair_split_rows(artifacts.repair),
                ),
                "## Detection benchmark\n\n"
                + _markdown_table(
                    (
                        "Detector",
                        "Corruption",
                        "Threshold",
                        "Precision",
                        "Recall",
                        "F1",
                        "TP",
                        "FP",
                        "FN",
                    ),
                    _detection_rows(artifacts.detection),
                )
                + "\n\n### Embedding provenance\n\n"
                + _markdown_table(
                    ("Field", "Value"),
                    _detection_provenance_rows(artifacts.detection),
                ),
                "## Runtime scaling\n\n"
                + _markdown_table(
                    (
                        "N",
                        "Stage",
                        "Mode",
                        "Seconds",
                        "Peak bytes",
                        "Memory scope",
                        "Recall@k",
                    ),
                    _scaling_rows(artifacts.scaling),
                ),
                "## Evaluation-integrity experiment\n\n"
                + _EVALUATION_NOTE
                + "\n\n### Experiment design and repair evidence\n\n"
                + _markdown_table(
                    ("Field", "Value"),
                    _training_summary_rows(artifacts.training),
                )
                + "\n\n### Condition composition\n\n"
                + _markdown_table(
                    (
                        "Condition",
                        "Split manifest SHA-256",
                        "Train images",
                        "Validation images",
                        "Test images",
                        "Non-injected test images",
                        "Injected derivatives in test",
                    ),
                    _training_condition_rows(artifacts.training),
                )
                + "\n\n### Detector-independent injected-family ground truth\n\n"
                + _markdown_table(
                    (
                        "Source ID",
                        "Derived ID",
                        "Source train position",
                        "Contaminated test position",
                        "Label",
                        "Transformation",
                        "Source SHA-256",
                        "Derived SHA-256",
                        "Expected relationship",
                        "Must share repaired split",
                    ),
                    _training_ground_truth_rows(artifacts.training),
                )
                + "\n\n### Seed-level evaluation metrics\n\n"
                "Chart bars are means; dots are raw seed-level runs.\n\n"
                + _markdown_table(
                    (
                        "Condition",
                        "Seed",
                        "Resolved device",
                        "Train accuracy",
                        "Validation accuracy",
                        "Shared clean holdout accuracy (paired)",
                        "All condition-test accuracy",
                        "Non-injected condition-test accuracy",
                        "Injected derivatives in actual test",
                        "Seconds",
                        "Split manifest SHA-256",
                    ),
                    _training_rows(artifacts.training),
                ),
                "## Charts\n\n"
                "![Detection precision-recall](detection_pr_curve.png)\n\n"
                "![Runtime scaling](runtime_scaling.png)\n\n"
                "![Split distribution](split_distribution.png)\n\n"
                "![Evaluation comparison](evaluation_comparison.png)",
                "## Provenance\n\n"
                + _markdown_table(
                    (
                        "Artifact",
                        "Schema",
                        "Timestamp",
                        "Configuration SHA-256",
                        "Dataset SHA-256",
                        "Seeds",
                        "Git commit",
                        "Git state",
                        "Python",
                        "OS",
                        "CPU",
                        "Accelerator availability",
                        "Packages",
                    ),
                    _provenance_rows(artifacts),
                ),
                "## Privacy\n\n"
                "This static report was produced locally. It contains stable IDs and relative "
                "paths only; "
                "it makes no network requests and does not modify source images.",
            )
        )
        + "\n"
    )


def _render_html(
    artifacts: _Artifacts,
    thumbnails: tuple[_Thumbnail, ...],
    thumbnail_status: str,
) -> str:
    audit = artifacts.audit
    cards = (
        ("Valid images", audit.summary.valid_image_count),
        ("Invalid images", audit.summary.invalid_image_count),
        ("Leakage groups", audit.summary.leakage_group_count),
        ("Contaminated evaluation", f"{audit.summary.contaminated_evaluation_fraction:.2%}"),
        ("Semantic review", audit.summary.embedding_only_review_count),
        ("Label conflicts", audit.summary.cross_label_conflict_count),
    )
    card_html = "".join(
        '<article class="card">'
        f"<span>{_html_text(label)}</span><strong>{_html_text(value)}</strong>"
        "</article>"
        for label, value in cards
    )
    invalid_rows = tuple(
        (
            issue.record_id or "—",
            issue.path or "—",
            issue.split.value if issue.split is not None else "—",
            issue.label or "—",
            issue.code.value,
            issue.message,
        )
        for issue in audit.invalid_records
    )
    gallery = "".join(
        '<figure class="thumb">'
        f'<img src="{html.escape(item.asset_path, quote=True)}" '
        f'alt="Representative image {_html_text(item.record_id)}" loading="lazy">'
        f"<figcaption><code>{_html_text(item.record_id)}</code><br>"
        f"{_html_text(item.split)} · {_html_text(item.label)}<br>"
        f"<code>{_html_text(item.path)}</code></figcaption></figure>"
        for item in thumbnails
    )
    repair_cards = '<p class="empty">Repair artifact not supplied.</p>'
    if artifacts.repair is not None:
        summary = artifacts.repair.summary
        repair_cards = _html_table(
            ("Metric", "Before", "After"),
            (
                (
                    "Split-size error",
                    _metric(summary.split_size_error_before),
                    _metric(summary.split_size_error_after),
                ),
                (
                    "Class divergence",
                    _metric(summary.class_divergence_before),
                    _metric(summary.class_divergence_after),
                ),
                (
                    "Definite leakage groups",
                    summary.definite_leakage_groups_before,
                    summary.definite_leakage_groups_after,
                ),
            ),
        )
        repair_cards += _html_table(
            ("Objective", "Moved images", "Hard family invariant"),
            (
                (
                    _metric(summary.objective_value),
                    summary.moved_image_count,
                    "satisfied" if summary.hard_group_invariant_satisfied else "failed",
                ),
            ),
        )

    style = """
    :root { color-scheme: light; --ink:#18232b; --muted:#59636b; --paper:#fbfaf7;
      --panel:#ffffff; --line:#d7d1c7; --accent:#315d73; --warn:#b7663f; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--paper); color:var(--ink); font:15px/1.55 system-ui,
      -apple-system,"Segoe UI",sans-serif; }
    main { width:min(1180px,calc(100% - 32px)); margin:0 auto; padding:42px 0 72px; }
    header { border-bottom:3px solid var(--ink); margin-bottom:28px; }
    h1 { font:700 clamp(2rem,6vw,4.6rem)/.95 Georgia,serif; margin:0 0 16px; }
    h2 { font:700 1.65rem/1.2 Georgia,serif; margin:42px 0 14px; }
    h3 { margin:28px 0 10px; }
    .lede,.note,.empty { color:var(--muted); }
    .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(155px,1fr));
      gap:12px; }
    .card { background:var(--panel); border:1px solid var(--line);
      border-top:4px solid var(--accent);
      padding:14px; }
    .card span { display:block; color:var(--muted); font-size:.8rem; text-transform:uppercase;
      letter-spacing:.05em; }
    .card strong { display:block; font:700 1.8rem/1.1 Georgia,serif; margin-top:6px; }
    .table-wrap { overflow-x:auto; border:1px solid var(--line); background:var(--panel); }
    table { border-collapse:collapse; width:100%; min-width:620px; }
    th,td { border-bottom:1px solid var(--line); padding:9px 11px; text-align:left;
      vertical-align:top; overflow-wrap:anywhere; }
    th { background:#eeece6; font-size:.78rem; letter-spacing:.04em; text-transform:uppercase; }
    code { font-family:"Cascadia Mono",Consolas,monospace; font-size:.84em; }
    .gallery { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:14px; }
    .thumb { margin:0; padding:10px; border:1px solid var(--line); background:var(--panel); }
    .thumb img { width:100%; height:150px; object-fit:contain; background:#e9e6df; display:block; }
    .thumb figcaption { margin-top:8px; color:var(--muted); overflow-wrap:anywhere; }
    .charts { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:16px; }
    .chart { margin:0; background:var(--panel); border:1px solid var(--line); padding:8px; }
    .chart img { width:100%; height:auto; display:block; }
    footer { margin-top:48px; padding-top:18px; border-top:1px solid var(--line);
      color:var(--muted); }
    @media (max-width:640px) { main { width:min(100% - 20px,1180px); padding-top:24px; }
      .charts { grid-template-columns:1fr; } }
    """
    exact_rows = _family_rows(audit, DuplicateClassification.EXACT)
    near_rows = _family_rows(audit, DuplicateClassification.TRANSFORMED_DUPLICATE)
    split_table = _html_table(("Split", "Images"), _split_rows(audit))
    class_table = _html_table(("Split", "Class", "Images"), _class_rows(audit))
    invalid_table = _html_table(
        ("Record", "Relative path", "Split", "Label", "Code", "Message"),
        invalid_rows,
    )
    family_headers = ("Family", "Members", "Splits", "Labels", "Evidence", "Member IDs")
    exact_table = _html_table(family_headers, exact_rows)
    near_table = _html_table(family_headers, near_rows)
    leakage_table = _html_table(
        (
            "Family",
            "Boundaries",
            "Strongest evidence",
            "Label conflict",
            "Labels",
            "Members",
        ),
        _leakage_rows(audit),
    )
    conflict_table = _html_table(
        ("Family", "Kind", "Labels", "Members"),
        _conflict_rows(audit),
    )
    evidence_table = _html_table(
        (
            "Left",
            "Right",
            "Classification",
            "Decision",
            "Exact",
            "pHash distance",
            "Cosine",
            "Confidence",
        ),
        _evidence_rows(audit),
    )
    repair_table = _html_table(
        ("Split", "Before", "After", "Classes before", "Classes after"),
        _repair_split_rows(artifacts.repair),
    )
    detection_table = _html_table(
        (
            "Detector",
            "Corruption",
            "Threshold",
            "Precision",
            "Recall",
            "F1",
            "TP",
            "FP",
            "FN",
        ),
        _detection_rows(artifacts.detection),
    )
    detection_provenance_table = _html_table(
        ("Field", "Value"),
        _detection_provenance_rows(artifacts.detection),
    )
    scaling_table = _html_table(
        (
            "N",
            "Stage",
            "Mode",
            "Seconds",
            "Peak bytes",
            "Memory scope",
            "Recall@k",
        ),
        _scaling_rows(artifacts.scaling),
    )
    training_summary_table = _html_table(
        ("Field", "Value"),
        _training_summary_rows(artifacts.training),
    )
    training_condition_table = _html_table(
        (
            "Condition",
            "Split manifest SHA-256",
            "Train images",
            "Validation images",
            "Test images",
            "Non-injected test images",
            "Injected derivatives in test",
        ),
        _training_condition_rows(artifacts.training),
    )
    training_ground_truth_table = _html_table(
        (
            "Source ID",
            "Derived ID",
            "Source train position",
            "Contaminated test position",
            "Label",
            "Transformation",
            "Source SHA-256",
            "Derived SHA-256",
            "Expected relationship",
            "Must share repaired split",
        ),
        _training_ground_truth_rows(artifacts.training),
    )
    training_table = _html_table(
        (
            "Condition",
            "Seed",
            "Resolved device",
            "Train accuracy",
            "Validation accuracy",
            "Shared clean holdout accuracy (paired)",
            "All condition-test accuracy",
            "Non-injected condition-test accuracy",
            "Injected derivatives in actual test",
            "Seconds",
            "Split manifest SHA-256",
        ),
        _training_rows(artifacts.training),
    )
    provenance_table = _html_table(
        (
            "Artifact",
            "Schema",
            "Timestamp",
            "Configuration SHA-256",
            "Dataset SHA-256",
            "Seeds",
            "Git commit",
            "Git state",
            "Python",
            "OS",
            "CPU",
            "Accelerator availability",
            "Packages",
        ),
        _provenance_rows(artifacts),
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SplitGuard Vision integrity report</title>
<style>{style}</style>
</head>
<body><main>
<header><p>LOCAL DATASET INTEGRITY ARTIFACT</p><h1>SplitGuard Vision</h1>
<p class="lede">Generated from strict JSON artifacts. Display values are rounded;
source JSON retains full precision. Semantic-only neighbors remain review candidates.</p>
</header>
<section aria-labelledby="summary"><h2 id="summary">Summary</h2>
<div class="cards">{card_html}</div></section>
<section><h2>Split statistics</h2>{split_table}
<h3>Class distribution</h3>{class_table}</section>
<section><h2>Invalid files</h2>{invalid_table}</section>
<section><h2>Exact duplicate groups</h2>{exact_table}</section>
<section><h2>Near-duplicate groups</h2>{near_table}</section>
<section><h2>Cross-split leakage</h2>{leakage_table}</section>
<section><h2>Label conflicts</h2>{conflict_table}</section>
<section><h2>Similarity evidence</h2>{evidence_table}</section>
<section><h2>Representative thumbnails</h2>
<p class="note">{_html_text(thumbnail_status)}</p>
<div class="gallery">{gallery}</div></section>
<section><h2>Repair summary</h2>{repair_cards}
<h3>Split distribution</h3>{repair_table}</section>
<section><h2>Detection benchmark</h2>{detection_table}
<h3>Embedding provenance</h3>{detection_provenance_table}</section>
<section><h2>Runtime scaling</h2>{scaling_table}</section>
<section><h2>Evaluation-integrity experiment</h2>
<p class="note">{_html_text(_EVALUATION_NOTE)}</p>
<h3>Experiment design and repair evidence</h3>{training_summary_table}
<h3>Condition composition</h3>{training_condition_table}
<h3>Detector-independent injected-family ground truth</h3>{training_ground_truth_table}
<h3>Seed-level evaluation metrics</h3>
<p class="note">Chart bars are means; dots are raw seed-level runs.</p>{training_table}</section>
<section><h2>Charts</h2><div class="charts">
<figure class="chart"><img src="detection_pr_curve.png"
alt="Detection precision-recall chart"></figure>
<figure class="chart"><img src="runtime_scaling.png" alt="Runtime scaling chart"></figure>
<figure class="chart"><img src="split_distribution.png" alt="Split distribution chart"></figure>
<figure class="chart"><img src="evaluation_comparison.png"
alt="Evaluation comparison chart"></figure>
</div></section>
<section><h2>Provenance</h2>{provenance_table}</section>
<section><h2>Privacy</h2><p>This report was produced locally. It contains stable IDs and
relative paths only, makes no network requests, and does not modify source images.</p></section>
<footer>SplitGuard Vision local integrity report.</footer>
</main></body></html>
"""


def _remove_owned_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _publish(staging: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if not output.is_dir():
        raise OSError
    owned_names = (*_REPORT_FILE_NAMES, _THUMBNAIL_DIRECTORY)
    for name in _REPORT_FILE_NAMES:
        if not (staging / name).is_file():
            raise OSError
        destination = output / name
        if destination.is_dir() and not destination.is_symlink():
            raise OSError
    if not (staging / _THUMBNAIL_DIRECTORY).is_dir():
        raise OSError

    backup = Path(tempfile.mkdtemp(prefix=".splitguard-report-backup-", dir=output.parent))
    backed_up: list[str] = []
    published: list[str] = []
    cleanup_backup = True
    try:
        for name in owned_names:
            destination = output / name
            if destination.exists() or destination.is_symlink():
                os.replace(destination, backup / name)
                backed_up.append(name)
        for name in owned_names:
            os.replace(staging / name, output / name)
            published.append(name)
    except OSError:
        rollback_failed = False
        for name in reversed(published):
            try:
                _remove_owned_path(output / name)
            except OSError:
                rollback_failed = True
        for name in reversed(backed_up):
            try:
                os.replace(backup / name, output / name)
            except OSError:
                rollback_failed = True
        cleanup_backup = not rollback_failed
        raise
    finally:
        if cleanup_backup and backup.exists():
            shutil.rmtree(backup)


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def generate_report(
    audit_path: str | Path,
    output_dir: str | Path,
    *,
    repair_path: str | Path | None = None,
    detection_benchmark_path: str | Path | None = None,
    scaling_benchmark_path: str | Path | None = None,
    training_results_path: str | Path | None = None,
    dataset_root: str | Path | None = None,
    no_thumbnails: bool = False,
    thumbnail_size: int = 160,
    max_thumbnails: int = 48,
) -> ReportResult:
    """Validate raw artifacts and publish a fully local HTML/Markdown report.

    The destination directory itself is never replaced. Only report-owned files
    and ``report_thumbnails`` are changed, preserving JSON artifacts and any
    unrelated files already stored beside them.
    """

    if not isinstance(no_thumbnails, bool):
        raise ReportInputError("no_thumbnails must be a boolean")
    if isinstance(thumbnail_size, bool) or not 32 <= thumbnail_size <= 1024:
        raise ReportInputError("thumbnail_size must be between 32 and 1024")
    if isinstance(max_thumbnails, bool) or not 1 <= max_thumbnails <= 1000:
        raise ReportInputError("max_thumbnails must be between 1 and 1000")

    artifacts = _Artifacts(
        audit=_load_artifact(audit_path, AuditArtifact, "audit"),
        repair=_load_optional_artifact(repair_path, RepairArtifact, "repair"),
        detection=_load_optional_artifact(
            detection_benchmark_path,
            DetectionBenchmarkArtifact,
            "detection benchmark",
        ),
        scaling=_load_optional_artifact(
            scaling_benchmark_path,
            ScalingBenchmarkArtifact,
            "scaling benchmark",
        ),
        training=_load_optional_artifact(
            training_results_path,
            TrainingArtifact,
            "training results",
        ),
    )
    if (
        artifacts.repair is not None
        and artifacts.repair.metadata.dataset_manifest_sha256
        != artifacts.audit.metadata.dataset_manifest_sha256
    ):
        raise ReportInputError("repair and audit artifacts describe different datasets")

    output = Path(output_dir).resolve(strict=False)
    if output.exists() and not output.is_dir():
        raise ReportInputError("report output must be a directory")
    root: Path | None = None
    if dataset_root is not None:
        root = Path(dataset_root).resolve(strict=False)
        if _paths_overlap(output, root):
            raise ReportInputError("report output and dataset root must not overlap")
        if not no_thumbnails and not root.is_dir():
            raise ReportInputError("dataset root must be a directory for thumbnails")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".splitguard-report-", dir=output.parent))
    try:
        thumbnails, thumbnails_skipped, thumbnail_status = _build_thumbnails(
            artifacts.audit,
            staging,
            root,
            no_thumbnails=no_thumbnails,
            thumbnail_size=thumbnail_size,
            max_thumbnails=max_thumbnails,
        )
        _chart_detection(artifacts.detection, staging / "detection_pr_curve.png")
        _chart_scaling(artifacts.scaling, staging / "runtime_scaling.png")
        _chart_splits(artifacts.audit, artifacts.repair, staging / "split_distribution.png")
        _chart_evaluation(artifacts.training, staging / "evaluation_comparison.png")
        (staging / "report.md").write_text(
            _render_markdown(artifacts, thumbnails, thumbnail_status),
            encoding="utf-8",
            newline="\n",
        )
        (staging / "report.html").write_text(
            _render_html(artifacts, thumbnails, thumbnail_status),
            encoding="utf-8",
            newline="\n",
        )
        _publish(staging, output)
    except ReportInputError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise ReportInputError("report generation failed without exposing source paths") from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    thumbnail_paths = tuple(output / item.asset_path for item in thumbnails)
    return ReportResult(
        output_dir=output,
        html_path=output / "report.html",
        markdown_path=output / "report.md",
        chart_paths=tuple(output / name for name in _CHART_NAMES),
        thumbnail_paths=thumbnail_paths,
        thumbnails_skipped=thumbnails_skipped,
    )


__all__ = ["ReportInputError", "ReportResult", "generate_report"]
