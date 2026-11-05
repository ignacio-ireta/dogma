"""Timed re-solves end to end: the real sandboxed runner, the real example, a scratch repo.

Run with: ``pytest -m e2e tests/e2e/cli/test_cli_resolve_e2e.py``. Nothing here touches the real
var/ or workspace/: every command runs against a copy under tmp_path.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dogma.cli import app
from dogma.core.config import Settings

pytestmark = pytest.mark.e2e

REPO_ROOT = Path(__file__).resolve().parents[3]
TOPIC = "cml/vector-operations"
PROBLEM = "cml/vector-operations/p01-vector-projection"
SOLUTION_FILE = (
    REPO_ROOT
    / "content"
    / "cml"
    / "vector-operations"
    / "problems"
    / "p01-vector-projection"
    / "solution.py"
)


def _invoke(settings: Settings, *args: str) -> object:
    return CliRunner().invoke(app, list(args), obj=settings, env={"COLUMNS": "200"})


def _text(output: str) -> str:
    return " ".join(output.split())


@pytest.fixture
def scratch(tmp_path: Path) -> Settings:
    root = tmp_path / "repo"
    root.mkdir()
    shutil.copy(REPO_ROOT / "pyproject.toml", root / "pyproject.toml")
    shutil.copytree(REPO_ROOT / "curriculum", root / "curriculum")
    shutil.copytree(REPO_ROOT / "templates", root / "templates")
    shutil.copytree(
        REPO_ROOT / "content" / "cml" / "vector-operations",
        root / "content" / "cml" / "vector-operations",
    )
    (root / "workspace").mkdir()
    return Settings(repo_root=root)


@pytest.fixture
def workspace_file(scratch: Settings) -> Path:
    return scratch.workspace_dir / "cml" / "vector-operations" / "p01-vector-projection.py"


def test_start_then_submit_the_solution_passes_the_timed_re_solve(
    scratch: Settings, workspace_file: Path
) -> None:
    _invoke(scratch, "open", PROBLEM)
    workspace_file.write_text(SOLUTION_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    assert _invoke(scratch, "submit", PROBLEM).exit_code == 0

    started = _invoke(scratch, "resolve", PROBLEM)
    workspace_file.write_text(SOLUTION_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    submitted = _invoke(scratch, "submit", PROBLEM)

    assert started.exit_code == 0, started.output
    assert f"Timed re-solve open: {PROBLEM}" in _text(started.output)
    assert "budget 20 min" in _text(started.output)
    assert "no hints" in _text(started.output)
    assert submitted.exit_code == 0, submitted.output
    assert "Timed re-solve passed" in _text(submitted.output)
    assert "Next timed re-solve:" in _text(submitted.output)


def test_start_then_submit_the_starter_fails_the_timed_re_solve(
    scratch: Settings, workspace_file: Path
) -> None:
    _invoke(scratch, "open", PROBLEM)
    workspace_file.write_text(SOLUTION_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    assert _invoke(scratch, "submit", PROBLEM).exit_code == 0
    assert _invoke(scratch, "resolve", PROBLEM).exit_code == 0

    submitted = _invoke(scratch, "submit", PROBLEM)

    assert submitted.exit_code == 1
    assert "Timed re-solve failed: not every test passed." in _text(submitted.output)
    assert "stage 0 of 3" in _text(submitted.output)
