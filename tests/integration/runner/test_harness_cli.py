"""``python -P -s -m dogma_harness`` on its own, as the runner calls it, with no runner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _harness(run_dir: Path, *tests: str) -> dict[str, object]:
    report = run_dir / "report.json"
    subprocess.run(
        [
            sys.executable,
            "-P",
            "-s",
            "-m",
            "dogma_harness",
            "--run-dir",
            str(run_dir),
            "--tests",
            *tests,
            "--report",
            str(report),
        ],
        cwd=run_dir,
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        check=False,
        timeout=60,
    )
    return json.loads(report.read_text(encoding="utf-8"))


def test_harness_writes_a_report_for_a_passing_run(tmp_path):
    (tmp_path / "submission.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (tmp_path / "test_public.py").write_text(
        "from submission import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    payload = _harness(tmp_path, "test_public.py")
    assert payload["status"] == "passed"
    assert payload["collection_error"] is None
    (test,) = payload["tests"]  # type: ignore[misc]
    assert (test["name"], test["file"], test["outcome"]) == ("test_add", "public", "pass")
    assert (tmp_path / "pytest.ini").read_text(encoding="utf-8") == "[pytest]\n"


def test_missing_test_file_is_reported_as_an_error_not_a_crash(tmp_path):
    payload = _harness(tmp_path, "test_public.py")
    assert payload["status"] == "error"
    assert "test file not found: test_public.py" in str(payload["collection_error"])


def _harness_with_tests_dir(run_dir: Path, tests_dir: Path, *tests: str) -> dict[str, object]:
    report = run_dir / "report.json"
    subprocess.run(
        [
            sys.executable,
            "-P",
            "-s",
            "-m",
            "dogma_harness",
            "--run-dir",
            str(run_dir),
            "--tests-dir",
            str(tests_dir),
            "--tests",
            *tests,
            "--report",
            str(report),
        ],
        cwd=run_dir,
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        check=False,
        timeout=60,
    )
    return json.loads(report.read_text(encoding="utf-8"))


def test_tests_dir_collects_tests_from_outside_the_run_dir(tmp_path):
    run_dir = tmp_path / "run"
    bundle = tmp_path / "bundle"
    run_dir.mkdir()
    bundle.mkdir()
    (run_dir / "submission.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (bundle / "submission.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (bundle / "test_public.py").write_text(
        "from submission import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    payload = _harness_with_tests_dir(run_dir, bundle, "test_public.py")
    assert payload["status"] == "passed"
    (test,) = payload["tests"]  # type: ignore[misc]
    assert (test["name"], test["file"], test["outcome"]) == ("test_add", "public", "pass")
    assert not (run_dir / "test_public.py").exists()


def test_missing_test_in_tests_dir_is_reported_even_if_the_run_dir_has_one(tmp_path):
    run_dir = tmp_path / "run"
    bundle = tmp_path / "bundle"
    run_dir.mkdir()
    bundle.mkdir()
    (run_dir / "test_public.py").write_text("def test_fake():\n    pass\n", encoding="utf-8")
    payload = _harness_with_tests_dir(run_dir, bundle, "test_public.py")
    assert payload["status"] == "error"
    assert "test file not found: test_public.py" in str(payload["collection_error"])


def test_harness_never_imports_the_app_package(tmp_path):
    probe = (
        "import sys, dogma_harness.cli, dogma_harness.collector, dogma_harness.report\n"
        "leaked = sorted(m for m in sys.modules if m == 'dogma' or m.startswith('dogma.'))\n"
        "raise SystemExit(1 if leaked else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr.decode()
