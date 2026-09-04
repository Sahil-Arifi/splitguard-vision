"""Command-line entry point for SplitGuard Vision."""

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Annotated, Literal

import typer
from pydantic import BaseModel, ValidationError

from splitguard import __version__
from splitguard.benchmark import (
    BenchmarkConfig,
    BenchmarkInputError,
    build_detection_artifact,
    build_scaling_artifact,
    load_benchmark_config,
    run_detection_benchmark,
    run_scaling_benchmarks,
)
from splitguard.config import ConfigLoadError, load_config
from splitguard.conflicts import analyze_conflicts
from splitguard.demo import DemoInputError, DemoRunError, run_offline_demo
from splitguard.embeddings import EmbeddingError, embed_records
from splitguard.graph import GraphInputError, build_evidence_graph
from splitguard.hashing import (
    PhashInputError,
    fingerprint_records,
    group_exact_duplicates,
    indexed_phash_pairs,
)
from splitguard.leakage import analyze_leakage
from splitguard.manifest import ManifestError, load_manifest
from splitguard.metrics import collect_run_metadata, manifest_snapshot_hash
from splitguard.models.embedder import DinoV2Embedder
from splitguard.neighbors import NeighborCandidate, build_neighbor_index
from splitguard.repair import (
    MaterializationError,
    RepairInputError,
    materialize_repaired_manifest,
    repair_splits,
    write_repaired_manifest,
)
from splitguard.reporting import ReportInputError, generate_report
from splitguard.schemas import (
    AuditArtifact,
    AuditSummary,
    RepairArtifact,
    canonical_json,
    canonical_sha256,
)
from splitguard.training import (
    CifarExperimentConfig,
    ExperimentConfigError,
    run_cifar_experiment,
)
from splitguard.validation import scan_images

app = typer.Typer(
    name="splitguard",
    help="Audit and repair image dataset split leakage without uploading source data.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Run SplitGuard Vision commands."""


@app.command()
def version(
    short: Annotated[
        bool,
        typer.Option("--short", help="Print only the version number."),
    ] = False,
) -> None:
    """Print the installed SplitGuard Vision version."""

    typer.echo(__version__ if short else f"SplitGuard Vision {__version__}")


@app.command()
def scan(
    dataset: Annotated[
        Path,
        typer.Argument(help="ImageFolder directory or path,split,label CSV manifest."),
    ],
    dataset_root: Annotated[
        Path | None,
        typer.Option(
            "--dataset-root",
            help="Root for relative CSV paths; defaults to the CSV's parent directory.",
        ),
    ] = None,
    max_image_pixels: Annotated[
        int,
        typer.Option(
            "--max-image-pixels",
            min=1,
            help="Maximum decoded width multiplied by height.",
        ),
    ] = 50_000_000,
) -> None:
    """Discover and validate every image, reporting bad inputs explicitly."""

    try:
        manifest = load_manifest(dataset, dataset_root=dataset_root)
        result = scan_images(
            manifest.root,
            manifest.entries,
            max_image_pixels=max_image_pixels,
        )
    except (ManifestError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="DATASET") from exc

    exact_groups = group_exact_duplicates(result.records)
    payload = {
        "mode": manifest.mode.value,
        "manifest_entries": result.entry_count,
        "valid_images": result.valid_count,
        "invalid_images": result.invalid_count,
        "exact_duplicate_groups": len(exact_groups),
        "exact_cross_split_groups": sum(
            group.crosses_evaluation_boundary for group in exact_groups
        ),
        "duration_seconds": result.duration_seconds,
        "invalid_records": [issue.model_dump(mode="json") for issue in result.issues],
    }
    typer.echo(json.dumps(payload, sort_keys=True, allow_nan=False))


def _write_json_artifact(path: Path, artifact: BaseModel) -> None:
    """Atomically persist canonical JSON in the requested local destination."""

    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(canonical_json(artifact))
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _load_audit_artifact(path: Path) -> AuditArtifact:
    """Load a strict audit contract without echoing host-specific paths."""

    try:
        document = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"could not read audit artifact {path.name!r}") from exc
    try:
        return AuditArtifact.model_validate_json(document)
    except (ValidationError, ValueError) as exc:
        raise ValueError("audit artifact is not valid SplitGuard audit JSON") from exc


def _parse_ratio_option(value: str) -> tuple[float, float, float]:
    """Parse the CLI's train,validation,test ratio triplet strictly."""

    parts = tuple(part.strip() for part in value.split(","))
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError("ratios must contain exactly three comma-separated numbers")
    try:
        ratios = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError("ratios must contain exactly three comma-separated numbers") from exc
    if any(not math.isfinite(ratio) for ratio in ratios):
        raise ValueError("ratios must be finite")
    if any(ratio < 0.0 for ratio in ratios):
        raise ValueError("ratios cannot be negative")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("ratios must sum to one")
    return ratios[0], ratios[1], ratios[2]


def _same_path(left: Path, right: Path) -> bool:
    """Compare destinations lexically after absolute normalization."""

    return left.resolve(strict=False) == right.resolve(strict=False)


def _benchmark_configuration_hash(config: BenchmarkConfig) -> str:
    """Hash the complete validated benchmark configuration canonically."""

    return canonical_sha256(config.model_dump(mode="json"))


def _scaling_fixture_hash(config: BenchmarkConfig) -> str:
    """Identify deterministic generated image/vector inputs, not runtime results."""

    return canonical_sha256(
        {
            "dataset_sizes": list(config.scaling.dataset_sizes),
            "embedding_dimension": config.scaling.embedding_dimension,
            "embedding_source": config.scaling.embedding_source,
            "fixture_schema": "splitguard-scaling-fixtures-v1",
            "local_image_generator": "deterministic_generated_png_files-v1",
            "seed": config.seed,
        }
    )


@app.command()
def benchmark_detection(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Validated detector benchmark YAML configuration."),
    ] = Path("configs/benchmark.yaml"),
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            help="Destination for canonical raw detector metrics JSON.",
        ),
    ] = Path("artifacts/detection_benchmark.json"),
) -> None:
    """Measure detector precision, recall, and F1 on controlled corruptions."""

    try:
        config = load_benchmark_config(config_path)
        configuration_sha256 = _benchmark_configuration_hash(config)
        benchmark_run = run_detection_benchmark(config)
        metadata = collect_run_metadata(
            configuration_sha256,
            benchmark_run.dataset_sha256,
            (config.seed,),
            repo_root=Path(__file__).parents[2],
        )
        artifact = build_detection_artifact(
            metadata,
            benchmark_run.rows,
            embedding_provenance=benchmark_run.embedding_provenance,
        )
        _write_json_artifact(output, artifact)
    except BenchmarkInputError as exc:
        raise typer.BadParameter(str(exc), param_hint="--config") from exc
    except (ImportError, OSError, RuntimeError, TypeError, ValidationError, ValueError) as exc:
        raise typer.BadParameter(
            "detector benchmark failed strict local validation",
            param_hint="--config",
        ) from exc

    typer.echo(
        json.dumps(
            {
                "artifact": output.name,
                "configuration_sha256": artifact.metadata.configuration_sha256,
                "corruption_types": sorted({row.corruption_type for row in artifact.rows}),
                "dataset_manifest_sha256": artifact.metadata.dataset_manifest_sha256,
                "embedding_backend": benchmark_run.embedding_provenance.backend,
                "embedding_is_synthetic": benchmark_run.embedding_provenance.is_synthetic,
                "metric_rows": len(artifact.rows),
                "observations": len(benchmark_run.observations),
                "seed": config.seed,
                "source_count": len(benchmark_run.source_records),
            },
            allow_nan=False,
            sort_keys=True,
        )
    )


@app.command()
def benchmark_scale(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Validated scaling benchmark YAML configuration."),
    ] = Path("configs/benchmark.yaml"),
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            help="Destination for canonical raw scaling metrics JSON.",
        ),
    ] = Path("artifacts/scaling_benchmark.json"),
) -> None:
    """Measure local audit stages and explicitly synthetic neighbor scaling."""

    try:
        config = load_benchmark_config(config_path)
        configuration_sha256 = _benchmark_configuration_hash(config)
        dataset_sha256 = _scaling_fixture_hash(config)
        rows = run_scaling_benchmarks(config)
        metadata = collect_run_metadata(
            configuration_sha256,
            dataset_sha256,
            (config.seed,),
            repo_root=Path(__file__).parents[2],
        )
        artifact = build_scaling_artifact(metadata, rows)
        _write_json_artifact(output, artifact)
    except BenchmarkInputError as exc:
        raise typer.BadParameter(str(exc), param_hint="--config") from exc
    except (ImportError, OSError, RuntimeError, TypeError, ValidationError, ValueError) as exc:
        raise typer.BadParameter(
            "scaling benchmark failed strict local validation",
            param_hint="--config",
        ) from exc

    typer.echo(
        json.dumps(
            {
                "artifact": output.name,
                "configuration_sha256": artifact.metadata.configuration_sha256,
                "dataset_manifest_sha256": artifact.metadata.dataset_manifest_sha256,
                "dataset_sizes": config.scaling.dataset_sizes,
                "embedding_source": config.scaling.embedding_source,
                "metric_rows": len(artifact.rows),
                "seed": config.seed,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )


@app.command()
def experiment(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Validated CIFAR-10 experiment YAML configuration."),
    ] = Path("configs/cifar10_experiment.yaml"),
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            help="Destination for canonical raw training-results JSON.",
        ),
    ] = Path("artifacts/training_results.json"),
) -> None:
    """Run matched contaminated and repaired CIFAR-10 training conditions."""

    try:
        config = CifarExperimentConfig.from_yaml(config_path)
        artifact = run_cifar_experiment(
            config,
            project_root=Path.cwd(),
            repo_root=Path(__file__).parents[2],
        )
        _write_json_artifact(output, artifact)
    except ExperimentConfigError as exc:
        raise typer.BadParameter(str(exc), param_hint="--config") from exc
    except (ImportError, OSError, RuntimeError, TypeError, ValidationError, ValueError) as exc:
        raise typer.BadParameter(
            "experiment failed; no training artifact was published",
            param_hint="--config",
        ) from exc

    typer.echo(
        json.dumps(
            {
                "artifact": output.name,
                "configuration_sha256": artifact.metadata.configuration_sha256,
                "dataset_manifest_sha256": artifact.metadata.dataset_manifest_sha256,
                "results": [
                    {
                        "all_test_accuracy": run.test_accuracy.accuracy,
                        "condition": run.condition.value,
                        "contaminated_example_accuracy": (
                            run.contaminated_example_accuracy.accuracy
                            if run.contaminated_example_accuracy is not None
                            else None
                        ),
                        "duration_seconds": run.duration_seconds,
                        "non_injected_test_accuracy": run.non_injected_test_accuracy.accuracy,
                        "resolved_device": run.resolved_device,
                        "seed": run.seed,
                        "shared_clean_holdout_accuracy": (
                            run.shared_clean_holdout_accuracy.accuracy
                        ),
                        "split_manifest_sha256": run.split_manifest_sha256,
                        "train_accuracy": run.train_accuracy.accuracy,
                        "validation_accuracy": run.validation_accuracy.accuracy,
                    }
                    for run in artifact.runs
                ],
                "dataset_source": artifact.summary.dataset_source,
                "injected_family_count": artifact.summary.injected_family_count,
                "repair_hard_group_invariant_satisfied": (
                    artifact.summary.repair_summary.hard_group_invariant_satisfied
                ),
                "resolved_device": artifact.summary.resolved_device,
                "run_count": len(artifact.runs),
                "seeds": artifact.metadata.random_seeds,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )


@app.command()
def report(
    audit_path: Annotated[
        Path,
        typer.Option("--audit", help="Validated SplitGuard audit JSON artifact."),
    ],
    repair_path: Annotated[
        Path | None,
        typer.Option("--repair", help="Optional validated repair JSON artifact."),
    ] = None,
    detection_benchmark_path: Annotated[
        Path | None,
        typer.Option(
            "--detection-benchmark",
            help="Optional detector benchmark JSON artifact.",
        ),
    ] = None,
    scaling_benchmark_path: Annotated[
        Path | None,
        typer.Option(
            "--scaling-benchmark",
            help="Optional scaling benchmark JSON artifact.",
        ),
    ] = None,
    training_results_path: Annotated[
        Path | None,
        typer.Option(
            "--training-results",
            help="Optional CIFAR training-results JSON artifact.",
        ),
    ] = None,
    dataset_root: Annotated[
        Path | None,
        typer.Option(
            "--dataset-root",
            help="Explicit source root used only to build local thumbnails.",
        ),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Directory for local HTML, Markdown, charts, and thumbnails.",
        ),
    ] = Path("artifacts"),
    no_thumbnails: Annotated[
        bool,
        typer.Option(
            "--no-thumbnails",
            help="Do not read or copy source images into report thumbnails.",
        ),
    ] = False,
) -> None:
    """Generate fully local HTML and Markdown reports from raw artifacts."""

    try:
        result = generate_report(
            audit_path,
            output_dir,
            repair_path=repair_path,
            detection_benchmark_path=detection_benchmark_path,
            scaling_benchmark_path=scaling_benchmark_path,
            training_results_path=training_results_path,
            dataset_root=dataset_root,
            no_thumbnails=no_thumbnails,
        )
    except ReportInputError as exc:
        raise typer.BadParameter(str(exc), param_hint="--audit") from exc
    except (OSError, RuntimeError, TypeError, ValidationError, ValueError) as exc:
        raise typer.BadParameter(
            "report generation failed; no report was published",
            param_hint="--audit",
        ) from exc

    typer.echo(
        json.dumps(
            {
                "charts": [path.name for path in result.chart_paths],
                "html": result.html_path.name,
                "markdown": result.markdown_path.name,
                "output": result.output_dir.name,
                "thumbnail_count": len(result.thumbnail_paths),
                "thumbnails_enabled": not no_thumbnails,
                "thumbnails_skipped": result.thumbnails_skipped,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )


@app.command()
def demo(
    workspace: Annotated[
        Path,
        typer.Option(
            "--workspace",
            help="New workspace for generated demo images and local cache.",
        ),
    ] = Path("demo-data"),
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="New destination for demo audit, repair, and report artifacts.",
        ),
    ] = Path("artifacts/demo"),
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            min=0,
            max=2**32 - 1,
            help="Unsigned 32-bit seed for the deterministic generated fixture.",
        ),
    ] = 20260903,
) -> None:
    """Generate, audit, repair, and report a deterministic offline demo."""

    try:
        result = run_offline_demo(workspace, output_dir, seed=seed)
    except (DemoInputError, DemoRunError) as exc:
        raise typer.BadParameter(
            str(exc),
            param_hint="--workspace/--output-dir",
        ) from exc
    except (OSError, RuntimeError, TypeError, ValidationError, ValueError) as exc:
        raise typer.BadParameter(
            "offline demo failed before verified publication",
            param_hint="--workspace/--output-dir",
        ) from exc

    typer.echo(
        json.dumps(
            {
                "artifacts": output_dir.name,
                "audit": result.audit_path,
                "charts": result.chart_paths,
                "cross_label_conflicts": result.cross_label_conflict_count,
                "dataset": result.dataset_path,
                "dataset_sha256": result.dataset_sha256,
                "invalid_images": result.invalid_image_count,
                "leakage_groups": result.leakage_group_count,
                "repair": result.repair_path,
                "repaired_manifest": result.repaired_manifest_path,
                "report_html": result.report_html_path,
                "report_markdown": result.report_markdown_path,
                "seed": result.seed,
                "source_bytes_unchanged": result.source_bytes_unchanged,
                "valid_images": result.valid_image_count,
                "workspace": workspace.name,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )


@app.command()
def audit(
    dataset: Annotated[
        Path,
        typer.Argument(help="ImageFolder directory or path,split,label CSV manifest."),
    ],
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Validated SplitGuard YAML configuration."),
    ] = Path("configs/default.yaml"),
    output: Annotated[
        Path,
        typer.Option("--output", help="Destination for the canonical audit JSON artifact."),
    ] = Path("artifacts/audit.json"),
    dataset_root: Annotated[
        Path | None,
        typer.Option("--dataset-root", help="Root for relative paths in a CSV manifest."),
    ] = None,
    no_phash: Annotated[
        bool,
        typer.Option("--no-phash", help="Disable pHash for this run."),
    ] = False,
    no_embeddings: Annotated[
        bool,
        typer.Option("--no-embeddings", help="Disable DINOv2 embeddings for this run."),
    ] = False,
) -> None:
    """Audit cross-split duplicate leakage and write a local JSON artifact."""

    try:
        config = load_config(config_path)
        manifest = load_manifest(dataset, dataset_root=dataset_root)
        scan_result = scan_images(
            manifest.root,
            manifest.entries,
            max_image_pixels=config.io.max_image_pixels,
        )
        records = scan_result.records
        if config.phash.enabled and not no_phash:
            records = fingerprint_records(manifest.root, records)
        exact_clusters = group_exact_duplicates(records)
        phash_candidates = (
            indexed_phash_pairs(records, config.phash.hamming_threshold)
            if config.phash.enabled and not no_phash
            else ()
        )

        neighbor_candidates: tuple[NeighborCandidate, ...] = ()
        if config.embeddings.enabled and not no_embeddings and len(records) > 1:
            embedder = DinoV2Embedder(
                model_name=config.embeddings.model,
                revision=config.embeddings.model_revision,
                preprocessing_version=config.embeddings.preprocessing_version,
                device=config.embeddings.device,
            )
            embedding_result = embed_records(
                manifest.root,
                records,
                embedder=embedder,
                cache_dir=config.io.cache_dir,
                batch_size=config.embeddings.batch_size,
            )
            index = build_neighbor_index(
                embedding_result.record_ids,
                embedding_result.vectors,
                kind=config.neighbors.index,
                hnsw_m=config.neighbors.hnsw_m,
                hnsw_ef_construction=config.neighbors.hnsw_ef_construction,
                hnsw_ef_search=config.neighbors.hnsw_ef_search,
            )
            neighbor_candidates = index.search(
                embedding_result.record_ids,
                embedding_result.vectors,
                k=config.neighbors.k,
                cosine_threshold=config.neighbors.cosine_threshold,
            )

        graph = build_evidence_graph(
            records,
            exact_clusters,
            phash_candidates,
            neighbor_candidates,
            phash_threshold=config.phash.hamming_threshold,
            cosine_threshold=config.neighbors.cosine_threshold,
            policy=config.policy,
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
        dataset_hash = manifest_snapshot_hash(records, scan_result.issues)
        metadata = collect_run_metadata(
            config.config_hash,
            dataset_hash,
            (config.repair.random_seed,),
            repo_root=Path(__file__).parents[2],
        )
        artifact = AuditArtifact(
            metadata=metadata,
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
                contaminated_evaluation_fraction=leakage.contaminated_evaluation_fraction,
                exact_leakage_group_count=leakage.exact_leakage_group_count,
                perceptual_leakage_group_count=leakage.perceptual_leakage_group_count,
                embedding_only_review_count=leakage.embedding_only_review_count,
                cross_label_conflict_count=conflicts.cross_label_conflict_count,
            ),
        )
        _write_json_artifact(output, artifact)
    except (
        ConfigLoadError,
        EmbeddingError,
        GraphInputError,
        ManifestError,
        OSError,
        PhashInputError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise typer.BadParameter(
            "audit failed before verified publication",
            param_hint="DATASET/--config/--output",
        ) from exc

    typer.echo(
        json.dumps(
            {
                "artifact": output.name,
                "valid_images": artifact.summary.valid_image_count,
                "invalid_images": artifact.summary.invalid_image_count,
                "leakage_groups": artifact.summary.leakage_group_count,
                "semantic_review_candidates": artifact.summary.embedding_only_review_count,
                "cross_label_conflicts": artifact.summary.cross_label_conflict_count,
            },
            sort_keys=True,
        )
    )


@app.command()
def repair(
    audit_json: Annotated[
        Path,
        typer.Argument(help="Canonical SplitGuard audit JSON artifact."),
    ],
    ratios: Annotated[
        str,
        typer.Option(
            "--ratios",
            help="Requested train,validation,test ratios as three comma-separated numbers.",
        ),
    ],
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Validated SplitGuard YAML supplying repair weights and optimizer settings.",
        ),
    ] = Path("configs/default.yaml"),
    output: Annotated[
        Path,
        typer.Option("--output", help="Destination for the canonical repair JSON artifact."),
    ] = Path("artifacts/repair.json"),
    manifest_output: Annotated[
        Path | None,
        typer.Option(
            "--manifest-output",
            help="Repaired CSV destination; defaults beside --output.",
        ),
    ] = None,
) -> None:
    """Repair definite duplicate families into indivisible split assignments."""

    repaired_manifest = (
        manifest_output
        if manifest_output is not None
        else output.parent / "repaired_manifest.csv"
    )
    if _same_path(audit_json, output) or _same_path(audit_json, repaired_manifest):
        raise typer.BadParameter(
            "repair outputs must not overwrite the source audit artifact",
            param_hint="--output/--manifest-output",
        )
    if _same_path(output, repaired_manifest):
        raise typer.BadParameter(
            "repair JSON and repaired manifest must use different destinations",
            param_hint="--output/--manifest-output",
        )

    try:
        requested_ratios = _parse_ratio_option(ratios)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--ratios") from exc

    try:
        config = load_config(config_path)
        source_audit = _load_audit_artifact(audit_json)
        plan = repair_splits(
            source_audit.records,
            source_audit.families,
            target_ratios=requested_ratios,
            split_size_weight=config.repair.split_size_weight,
            class_balance_weight=config.repair.class_balance_weight,
            seed=config.repair.random_seed,
            local_iterations=config.repair.local_improvement_iterations,
        )
        repaired_manifest_sha256 = write_repaired_manifest(
            repaired_manifest,
            source_audit.records,
            plan.assignments,
        )
        repair_configuration_sha256 = canonical_sha256(
            {
                "base_configuration_sha256": config.config_hash,
                "repair": {
                    "class_balance_weight": plan.class_balance_weight,
                    "local_improvement_iterations": plan.local_iterations,
                    "random_seed": plan.seed,
                    "split_size_weight": plan.split_size_weight,
                },
                "requested_ratios": [
                    {"ratio": item.ratio, "split": item.split.value}
                    for item in plan.requested_ratios
                ],
            }
        )
        metadata = collect_run_metadata(
            repair_configuration_sha256,
            source_audit.metadata.dataset_manifest_sha256,
            (plan.seed,),
            repo_root=Path(__file__).parents[2],
        )
        artifact = RepairArtifact(
            metadata=metadata,
            requested_ratios=plan.requested_ratios,
            integer_targets=plan.integer_targets,
            assignments=plan.assignments,
            excluded_invalid_ids=tuple(
                sorted(
                    {
                        issue.record_id
                        for issue in source_audit.invalid_records
                        if issue.record_id is not None
                    }
                )
            ),
            infeasibility_warnings=plan.infeasibility_warnings,
            split_size_weight=plan.split_size_weight,
            class_balance_weight=plan.class_balance_weight,
            random_seed=plan.seed,
            local_improvement_iterations=plan.local_iterations,
            repaired_manifest_sha256=repaired_manifest_sha256,
            before_split_statistics=plan.before_split_statistics,
            after_split_statistics=plan.after_split_statistics,
            summary=plan.summary,
        )
        _write_json_artifact(output, artifact)
    except (ConfigLoadError, RepairInputError) as exc:
        raise typer.BadParameter(str(exc), param_hint="AUDIT_JSON") from exc
    except (OSError, ValidationError, ValueError) as exc:
        raise typer.BadParameter(
            "repair input or output failed strict validation",
            param_hint="AUDIT_JSON",
        ) from exc

    typer.echo(
        json.dumps(
            {
                "artifact": output.name,
                "audit_configuration_sha256": source_audit.metadata.configuration_sha256,
                "class_divergence_after": plan.class_divergence_after,
                "configuration_sha256": artifact.metadata.configuration_sha256,
                "dataset_manifest_sha256": artifact.metadata.dataset_manifest_sha256,
                "definite_leakage_groups_after": plan.definite_leakage_groups_after,
                "hard_group_invariant_satisfied": plan.hard_group_invariant_satisfied,
                "integer_targets": {
                    split.value: count for split, count in plan.integer_targets
                },
                "manifest": repaired_manifest.name,
                "moved_images": plan.moved_image_count,
                "objective_value": plan.objective_value,
                "repaired_manifest_sha256": repaired_manifest_sha256,
                "source_audit_sha256": canonical_sha256(source_audit),
                "split_size_error_after": plan.split_size_error_after,
                "warnings": plan.infeasibility_warnings,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )


@app.command()
def materialize(
    repaired_manifest: Annotated[
        Path,
        typer.Argument(help="Repaired path,split,label CSV manifest."),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(help="New destination directory; it must not already exist."),
    ],
    dataset_root: Annotated[
        Path,
        typer.Option(
            "--dataset-root",
            help="Root containing the repaired manifest's relative source paths.",
        ),
    ],
    mode: Annotated[
        Literal["copy", "symlink"],
        typer.Option("--mode", help="Materialize verified copies or relative symbolic links."),
    ] = "copy",
) -> None:
    """Explicitly create a repaired dataset tree without changing source files."""

    try:
        result = materialize_repaired_manifest(
            repaired_manifest,
            dataset_root,
            output_dir,
            mode=mode,
        )
    except (ManifestError, MaterializationError, RepairInputError) as exc:
        raise typer.BadParameter(str(exc), param_hint="REPAIRED_MANIFEST") from exc
    except (OSError, ValidationError, ValueError) as exc:
        raise typer.BadParameter(
            "materialization failed strict safety validation",
            param_hint="REPAIRED_MANIFEST",
        ) from exc

    typer.echo(
        json.dumps(
            {
                "manifest_sha256": result.manifest_sha256,
                "mode": result.mode,
                "output": output_dir.name,
                "record_count": result.record_count,
                "verified_file_count": result.verified_file_count,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover - console script is the supported entry point
    app()
