"""Protected test bundle: contents, read-only modes, symlink rejection and cleanup on every path."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dogma.core.errors import ErrorCode, RunnerError
from dogma.runner import staging
from dogma.runner.meta import load_meta

REPO_ROOT = Path(__file__).resolve().parents[3]
PY_ADD = REPO_ROOT / "tests" / "fixtures" / "problems" / "py-add"
SQL_ORDERS = REPO_ROOT / "tests" / "fixtures" / "problems" / "sql-orders"
ALL_TESTS = ("test_public.py", "test_hidden.py")


def _bundles(runs_dir: Path) -> list[Path]:
    return [p for p in runs_dir.iterdir() if ".tests-" in p.name]


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    path = tmp_path / "var" / "runs"
    path.mkdir(parents=True)
    return path


@pytest.fixture
def run_dir(runs_dir: Path) -> Path:
    path = runs_dir / "attempt-1"
    path.mkdir()
    return path


def _submission(tmp_path: Path, text: str, name: str = "submission.py") -> Path:
    path = tmp_path / "work" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_bundle_holds_only_selected_tests_submission_and_fixtures(tmp_path, run_dir, runs_dir):
    sub = _submission(tmp_path, "def add(a, b):\n    return a + b\n")
    meta = load_meta(PY_ADD)
    with staging.staged_tests(run_dir, sub, PY_ADD, meta, ALL_TESTS) as bundle:
        assert sorted(p.name for p in bundle.iterdir()) == [
            "submission.py",
            "test_hidden.py",
            "test_public.py",
        ], "py-add has no fixtures/, so none are staged"
        assert (bundle / "submission.py").read_text() == sub.read_text()
        assert bundle.parent == runs_dir, "a sibling of the attempt, outside its write subtree"
        assert not bundle.is_relative_to(run_dir)


def test_run_dir_keeps_submission_and_fixtures_but_never_the_tests(tmp_path, run_dir, runs_dir):
    sub = _submission(tmp_path, "SELECT 1;\n", name="submission.sql")
    meta = load_meta(SQL_ORDERS)
    with staging.staged_tests(run_dir, sub, SQL_ORDERS, meta, ALL_TESTS):
        assert (run_dir / "submission.sql").read_text() == "SELECT 1;\n"
        assert (run_dir / "fixtures" / "orders.csv").is_file()
        assert not (run_dir / "test_hidden.py").exists()
        assert not (run_dir / "test_public.py").exists()


def test_sql_bundle_has_the_sibling_layout_the_tests_read(tmp_path, run_dir, runs_dir):
    sub = _submission(tmp_path, "SELECT 1;\n", name="submission.sql")
    meta = load_meta(SQL_ORDERS)
    with staging.staged_tests(run_dir, sub, SQL_ORDERS, meta, ALL_TESTS) as bundle:
        assert (bundle / "submission.sql").read_text() == "SELECT 1;\n"
        assert (bundle / "fixtures" / "orders.csv").is_file()


def test_bundle_files_are_read_only_during_the_run(tmp_path, run_dir, runs_dir):
    sub = _submission(tmp_path, "def add(a, b):\n    return a + b\n")
    with staging.staged_tests(run_dir, sub, PY_ADD, load_meta(PY_ADD), ALL_TESTS) as bundle:
        assert (bundle / "test_hidden.py").stat().st_mode & 0o222 == 0
        assert not bundle.stat().st_mode & 0o200


def test_bundle_is_removed_after_a_normal_exit(tmp_path, run_dir, runs_dir):
    sub = _submission(tmp_path, "def add(a, b):\n    return a + b\n")
    with staging.staged_tests(run_dir, sub, PY_ADD, load_meta(PY_ADD), ALL_TESTS) as bundle:
        assert bundle.is_dir()
    assert not bundle.exists()
    assert _bundles(runs_dir) == []
    assert run_dir.is_dir(), "the retained attempt directory is never removed"


def test_bundle_is_removed_and_the_original_error_survives_a_failing_body(
    tmp_path, run_dir, runs_dir
):
    sub = _submission(tmp_path, "def add(a, b):\n    return a + b\n")
    with (
        pytest.raises(KeyError, match="original"),
        staging.staged_tests(run_dir, sub, PY_ADD, load_meta(PY_ADD), ALL_TESTS),
    ):
        raise KeyError("original")
    assert _bundles(runs_dir) == []
    assert run_dir.is_dir()


def test_a_copy_failure_mid_staging_leaves_no_bundle_and_raises_an_infra_error(
    tmp_path, run_dir, runs_dir, monkeypatch
):
    sub = _submission(tmp_path, "def add(a, b):\n    return a + b\n")
    real_copy = staging.shutil.copyfile
    calls = {"n": 0}

    def flaky(src, dst, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 3:  # run-dir submission, then bundle submission, then the first test
            raise OSError("disk full (simulated)")
        return real_copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(staging.shutil, "copyfile", flaky)
    with (
        pytest.raises(RunnerError) as info,
        staging.staged_tests(run_dir, sub, PY_ADD, load_meta(PY_ADD), ALL_TESTS),
    ):
        pytest.fail("the body must not run after a staging failure")
    assert info.value.code is ErrorCode.RUNNER_CRASHED
    assert _bundles(runs_dir) == []
    assert run_dir.is_dir()


def test_fixture_symlink_is_rejected_before_it_is_followed(tmp_path, run_dir, runs_dir):
    problem = tmp_path / "problem"
    shutil.copytree(PY_ADD, problem)
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("not part of the problem", encoding="utf-8")
    fixtures = problem / "fixtures"
    fixtures.mkdir()
    (fixtures / "leak.txt").symlink_to(outside)
    sub = _submission(tmp_path, "def add(a, b):\n    return a + b\n")
    with (
        pytest.raises(RunnerError) as info,
        staging.staged_tests(run_dir, sub, problem, load_meta(problem), ALL_TESTS),
    ):
        pytest.fail("the body must not run")
    assert info.value.code is ErrorCode.CONTENT_SCHEMA_INVALID
    assert "symlink" in info.value.message
    assert not (run_dir / "fixtures" / "leak.txt").exists()
    assert _bundles(runs_dir) == []


def test_fixture_directory_symlink_is_rejected_too(tmp_path, run_dir, runs_dir):
    problem = tmp_path / "problem"
    shutil.copytree(PY_ADD, problem)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (problem / "fixtures").symlink_to(elsewhere, target_is_directory=True)
    sub = _submission(tmp_path, "def add(a, b):\n    return a + b\n")
    with (
        pytest.raises(RunnerError) as info,
        staging.staged_tests(run_dir, sub, problem, load_meta(problem), ALL_TESTS),
    ):
        pytest.fail("the body must not run")
    assert info.value.code is ErrorCode.CONTENT_SCHEMA_INVALID
