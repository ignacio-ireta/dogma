"""End to end through the CLI: the real sandboxed runner and the real example content.

Anything that writes (workspace/, var/runs/, var/catalog.json, new content) happens in a scratch
copy under tmp_path. The real content is only read, except by ``content verify``, which grades in
its own scratch directory. Run with: ``pytest -m e2e tests/e2e/cli``.
"""

from __future__ import annotations

import re
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
SOLUTION = (
    REPO_ROOT / "content" / "cml" / "vector-operations" / "problems" / "p01-vector-projection"
)
SOLUTION_FILE = SOLUTION / "solution.py"


def _invoke(settings: Settings, *args: str) -> object:
    return CliRunner().invoke(app, list(args), obj=settings, env={"COLUMNS": "200"})


@pytest.fixture
def scratch(tmp_path: Path) -> Settings:
    """A scratch repo: real pyproject (for ruff), curriculum, templates and the example topic."""
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


def test_content_verify_passes_on_the_real_example():
    outcome = _invoke(Settings(repo_root=REPO_ROOT), "content", "verify", TOPIC)

    assert outcome.exit_code == 0, outcome.output
    assert "All 1 topic(s) verified." in outcome.output


def test_submit_the_real_solution_passes_all_tests(scratch: Settings):
    outcome = _invoke(scratch, "submit", PROBLEM, "--submission", str(SOLUTION_FILE))

    assert outcome.exit_code == 0, outcome.output
    assert re.search(
        r"public (\d+)/\1 · hidden (\d+)/\2 · passed", " ".join(outcome.output.split())
    )


def test_run_the_real_solution_passes_the_public_tests(scratch: Settings):
    outcome = _invoke(scratch, "run", PROBLEM, "--submission", str(SOLUTION_FILE))

    assert outcome.exit_code == 0, outcome.output
    assert "passed" in outcome.output


def test_open_then_submit_the_starter_fails_and_says_why(scratch: Settings):
    opened = _invoke(scratch, "open", PROBLEM)
    assert opened.exit_code == 0, opened.output
    starter_copy = scratch.workspace_dir / "cml" / "vector-operations" / "p01-vector-projection.py"
    assert starter_copy.read_text() == (SOLUTION / "starter.py").read_text()

    outcome = _invoke(scratch, "submit", PROBLEM)

    assert outcome.exit_code == 1
    assert "✗" in outcome.output
    assert "not passed yet" in outcome.output


def test_a_scaffolded_topic_loads_but_fails_verification_until_filled_in(scratch: Settings):
    created = _invoke(scratch, "content", "new", "cml/cosine-similarity")
    assert created.exit_code == 0, created.output

    verified = _invoke(scratch, "content", "verify", "cml/cosine-similarity")

    assert verified.exit_code == 1
    assert "NotImplementedError" in verified.output or "does not pass" in verified.output


def test_catalog_build_writes_var_catalog_in_the_scratch_repo(scratch: Settings):
    outcome = _invoke(scratch, "catalog", "build")

    assert outcome.exit_code == 0, outcome.output
    assert (scratch.var_dir / "catalog.json").is_file()


def test_path_and_coverage_read_the_real_curriculum():
    settings = Settings(repo_root=REPO_ROOT)

    path = _invoke(settings, "path", "--week", "1")
    coverage = _invoke(settings, "content", "coverage", "--tier", "A")

    assert path.exit_code == 0, path.output
    assert "Week 1" in path.output
    assert coverage.exit_code == 0, coverage.output
    assert "By tier" not in coverage.output
