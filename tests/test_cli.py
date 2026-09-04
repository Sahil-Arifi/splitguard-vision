"""CLI bootstrap tests."""

from typer.testing import CliRunner

from splitguard.cli import app

runner = CliRunner()


def test_root_help_is_useful() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Audit and repair image dataset split leakage" in result.stdout
    assert "version" in result.stdout


def test_version_supports_short_output() -> None:
    result = runner.invoke(app, ["version", "--short"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"
