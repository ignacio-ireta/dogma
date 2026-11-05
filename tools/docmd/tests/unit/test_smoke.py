"""Phase 0 smoke tests: package imports and the version command works."""

from typer.testing import CliRunner

import docmd
from docmd.cli import app

runner = CliRunner()


def test_package_has_version() -> None:
    assert docmd.__version__
    assert docmd.__version__ != "0.0.0+unknown"


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert f"docmd {docmd.__version__}" in result.output


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output
