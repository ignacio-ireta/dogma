"""Shared CLI fixtures: a temporary repo copied from the real curriculum, templates and example.

Nothing here writes into the real content/, workspace/ or var/: the repo is a copy under tmp_path.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from tests.unit.cli.support import make_report
from typer.testing import CliRunner

from dogma.cli import app
from dogma.core.config import Settings
from dogma.grading.models import GradeReport
from dogma.runner import attempt

REPO_ROOT = Path(__file__).resolve().parents[3]
TOPIC = "cml/vector-operations"
PROBLEM = "cml/vector-operations/p01-vector-projection"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A temporary repo: real curriculum and templates, a copy of the example topic."""
    root = tmp_path / "repo"
    root.mkdir()
    # The ruff configuration: verify lints the copy with the same rules as the real repository.
    shutil.copy(REPO_ROOT / "pyproject.toml", root / "pyproject.toml")
    shutil.copytree(REPO_ROOT / "curriculum", root / "curriculum")
    shutil.copytree(REPO_ROOT / "templates", root / "templates")
    shutil.copytree(
        REPO_ROOT / "content" / "cml" / "vector-operations",
        root / "content" / "cml" / "vector-operations",
    )
    (root / "workspace").mkdir()
    return root


@pytest.fixture
def settings(repo: Path) -> Settings:
    return Settings(repo_root=repo)


@pytest.fixture
def cli() -> CliRunner:
    """A runner whose output is plain text wherever the tests run.

    Typer forces Rich's terminal mode when ``GITHUB_ACTIONS`` (or ``FORCE_COLOR``) is set, so
    help text arrives with ANSI styling between the words the tests look for. A dumb terminal
    has no colour system, which turns the styling off without changing the wording.
    """
    return CliRunner(env={"TERM": "dumb", "NO_COLOR": "1"})


@pytest.fixture
def problem_dir(repo: Path) -> Path:
    return repo / "content" / "cml" / "vector-operations" / "problems" / "p01-vector-projection"


class FakeGrader:
    """Stands in for ``runner.attempt.grade_attempt``: scripted reports, recorded calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, bool]] = []
        self.script: dict[str, GradeReport] = {}
        # Replies handed out in order for one submission name, before ``script`` applies.
        self.queue: dict[str, list[GradeReport]] = {}

    def __call__(
        self,
        problem_dir: Path,
        submission: Path,
        *,
        include_hidden: bool,
        settings: Settings,
    ) -> GradeReport:
        self.calls.append((submission.name, problem_dir.name, include_hidden))
        queued = self.queue.get(submission.name)
        if queued:
            return queued.pop(0)
        return self.script.get(submission.name, make_report())


@pytest.fixture
def fake_grader(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeGrader]:
    grader = FakeGrader()
    monkeypatch.setattr(attempt, "grade_attempt", grader)
    yield grader


@pytest.fixture
def invoke(cli: CliRunner, settings: Settings) -> Callable[..., object]:
    """Run the dogma app with the temporary settings injected."""

    def _invoke(*args: str) -> object:
        return cli.invoke(app, list(args), obj=settings, env={"COLUMNS": "200"})

    return _invoke
