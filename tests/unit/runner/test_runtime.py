"""Optional runtimes fail before learner code or artifact creation."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

from dogma.core.config import Settings
from dogma.core.errors import RunnerError
from dogma.runner import run, runtime

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "problems" / "py-add"


def test_missing_torch_has_an_actionable_stable_error(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RunnerError) as error:
        runtime.ensure_available("py-torch")
    assert error.value.code.value == "RUNNER_RUNTIME_UNAVAILABLE"
    assert "uv run --frozen --group torch dogma" in error.value.message


def test_discoverable_torch_needs_no_eager_import(monkeypatch):
    seen = []

    def find(name):
        seen.append(name)
        return object()

    monkeypatch.setattr(importlib.util, "find_spec", find)
    runtime.ensure_available("py-torch")
    assert seen == ["torch"]


def test_other_runtimes_do_not_probe_torch(monkeypatch):
    def unexpected(name):
        pytest.fail(f"unnecessary optional package lookup: {name}")

    monkeypatch.setattr(importlib.util, "find_spec", unexpected)
    for name in ("py-stdlib", "py-numpy", "py-data", "sql-duckdb"):
        runtime.ensure_available(name)


def test_missing_runtime_does_not_create_artifacts_or_spawn(tmp_path, monkeypatch):
    problem = tmp_path / "problem"
    shutil.copytree(FIXTURE, problem)
    meta = problem / "meta.yaml"
    meta.write_text(meta.read_text().replace("py-numpy", "py-torch"))
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    def unexpected(*args, **kwargs):
        pytest.fail("an unavailable runtime must not execute learner code")

    monkeypatch.setattr("dogma.runner.sandbox.execute", unexpected)
    settings = Settings(repo_root=tmp_path / "scratch", _env_file=None)
    submission = tmp_path / "answer.py"
    submission.write_text("def add(a, b):\n    return a + b\n")
    with pytest.raises(RunnerError, match="RUNNER_RUNTIME_UNAVAILABLE"):
        run(problem, submission, include_hidden=True, settings=settings)
    assert not settings.runs_dir.exists()
