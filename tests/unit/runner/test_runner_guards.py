"""Runner paths that must never start a subprocess: static rejects and infrastructure errors."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dogma.core.config import Settings
from dogma.core.errors import ErrorCode, RunnerError
from dogma.grading.models import RunStatus
from dogma.runner import new_attempt_id, run
from dogma.runner import sandbox as sandbox_module

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "problems"
PY_ADD = FIXTURES / "py-add"
SQL_ORDERS = FIXTURES / "sql-orders"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(repo_root=tmp_path / "repo")


@pytest.fixture
def no_spawn(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Fail loudly if anything reaches the sandbox; the guard paths must not get there."""
    calls: list[object] = []

    def _forbidden(*args: object, **kwargs: object) -> None:
        calls.append(args)
        raise AssertionError("the sandbox must not run for this case")

    monkeypatch.setattr(sandbox_module, "execute", _forbidden)
    return calls


def _submission(tmp_path: Path, source: str, name: str = "submission.py") -> Path:
    path = tmp_path / "work" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def test_static_finding_rejects_without_running_or_creating_a_run_dir(tmp_path, settings, no_spawn):
    sub = _submission(tmp_path, "def add(a, b) return a + b\n")
    report = run(PY_ADD, sub, include_hidden=True, settings=settings)
    assert report.status is RunStatus.REJECTED
    assert [f.rule for f in report.static] == ["syntax"]
    assert report.tests == ()
    assert not settings.runs_dir.exists(), "a rejected attempt must not leave a run directory"
    assert no_spawn == []


def test_rejected_report_carries_the_problem_identity(tmp_path, settings, no_spawn):
    sub = _submission(tmp_path, "def plus(a, b):\n    return a + b\n")
    report = run(PY_ADD, sub, include_hidden=False, settings=settings, attempt_id="fixed-1")
    assert report.attempt_id == "fixed-1"
    assert report.problem_id == "fixtures/py-add/p01-add"
    assert report.problem_version == 1
    assert report.runtime == "py-numpy"
    assert report.message


def test_empty_sql_is_rejected_before_running(tmp_path, settings, no_spawn):
    sub = _submission(tmp_path, "\n", name="submission.sql")
    report = run(SQL_ORDERS, sub, include_hidden=True, settings=settings)
    assert report.status is RunStatus.REJECTED
    assert [f.rule for f in report.static] == ["empty"]
    assert no_spawn == []


def test_problem_dir_without_meta_is_an_infrastructure_error(tmp_path, settings, no_spawn):
    sub = _submission(tmp_path, "def add(a, b):\n    return a + b\n")
    with pytest.raises(RunnerError) as info:
        run(tmp_path, sub, include_hidden=True, settings=settings)
    assert info.value.code == ErrorCode.CONTENT_NOT_FOUND


def test_missing_hidden_tests_are_an_error_only_when_they_are_needed(tmp_path, settings, no_spawn):
    problem = tmp_path / "problem"
    shutil.copytree(PY_ADD, problem)
    (problem / "test_hidden.py").unlink()
    sub = _submission(tmp_path, "def add(a, b):\n    return a + b\n")
    with pytest.raises(RunnerError) as info:
        run(problem, sub, include_hidden=True, settings=settings)
    assert info.value.code == ErrorCode.CONTENT_NOT_FOUND
    assert "test_hidden.py" in info.value.message


def test_attempt_id_that_is_not_path_safe_is_refused(tmp_path, settings, no_spawn):
    sub = _submission(tmp_path, "def add(a, b):\n    return a + b\n")
    for bad in ["../escape", "a/b", "", ".hidden", "x" * 80]:
        with pytest.raises(ValueError, match="attempt id"):
            run(PY_ADD, sub, include_hidden=False, settings=settings, attempt_id=bad)


def test_existing_attempt_directory_is_an_infrastructure_error(tmp_path, settings, no_spawn):
    sub = _submission(tmp_path, "def add(a, b):\n    return a + b\n")
    (settings.runs_dir / "taken").mkdir(parents=True)
    with pytest.raises(RunnerError) as info:
        run(PY_ADD, sub, include_hidden=False, settings=settings, attempt_id="taken")
    assert info.value.code == ErrorCode.RUNNER_CRASHED
    assert no_spawn == []


def test_new_attempt_ids_are_time_ordered_and_unique():
    first, second = new_attempt_id(), new_attempt_id()
    assert first != second
    stamp, _, suffix = first.partition("-")
    assert len(stamp) == 16 and stamp.endswith("Z") and "T" in stamp
    assert len(suffix) == 8
    assert first.split("-")[0] <= second.split("-")[0]


def test_a_run_directory_that_cannot_be_created_is_an_infrastructure_error(tmp_path, no_spawn):
    # The runs root cannot be a directory's parent when a plain file sits where ``var`` should be.
    settings = Settings(repo_root=tmp_path / "repo")
    settings.var_dir.parent.mkdir(parents=True)
    settings.var_dir.write_text("not a directory", encoding="utf-8")
    sub = _submission(tmp_path, "def add(a, b):\n    return a + b\n")
    with pytest.raises(RunnerError) as info:
        run(PY_ADD, sub, include_hidden=False, settings=settings, attempt_id="x")
    assert info.value.code == ErrorCode.RUNNER_CRASHED
    assert no_spawn == []


def test_a_harness_that_cannot_be_started_is_an_infrastructure_error(
    tmp_path, settings, monkeypatch
):
    def _refuse(*args: object, **kwargs: object) -> None:
        raise OSError("exec format error")

    monkeypatch.setattr(sandbox_module.subprocess, "Popen", _refuse)
    sub = _submission(tmp_path, "def add(a, b):\n    return a + b\n")
    with pytest.raises(RunnerError) as info:
        run(PY_ADD, sub, include_hidden=False, settings=settings, attempt_id="spawn-fail")
    assert info.value.code == ErrorCode.RUNNER_CRASHED
    assert "exec format error" in info.value.message
