"""Command-line entry point for SplitGuard Vision."""

from typing import Annotated

import typer

from splitguard import __version__

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


if __name__ == "__main__":  # pragma: no cover - console script is the supported entry point
    app()
