"""Command-line entry point for SplitGuard Vision."""

import json
from pathlib import Path
from typing import Annotated

import typer

from splitguard import __version__
from splitguard.hashing import group_exact_duplicates
from splitguard.manifest import ManifestError, load_manifest
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


if __name__ == "__main__":  # pragma: no cover - console script is the supported entry point
    app()
