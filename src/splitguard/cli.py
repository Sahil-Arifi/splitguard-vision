"""Command-line entry point for SplitGuard Vision."""

import json
import os
import tempfile
from pathlib import Path
from typing import Annotated

import typer

from splitguard import __version__
from splitguard.config import ConfigLoadError, load_config
from splitguard.conflicts import analyze_conflicts
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
from splitguard.schemas import AuditArtifact, AuditSummary, canonical_json
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


def _write_json_artifact(path: Path, artifact: AuditArtifact) -> None:
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
        raise typer.BadParameter(str(exc), param_hint="DATASET") from exc

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


if __name__ == "__main__":  # pragma: no cover - console script is the supported entry point
    app()
