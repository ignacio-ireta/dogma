"""Check actual scratch files for the local commit hook."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "check_commit_files.py"


def run_check(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(path.parent), path.name],
        capture_output=True,
        text=True,
        check=False,
    )


def load_checker() -> ModuleType:
    assert SCRIPT.is_file(), "commit file checker has not been implemented"
    spec = importlib.util.spec_from_file_location("check_commit_files", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_clean_text_and_empty_files_pass(tmp_path: Path) -> None:
    path = tmp_path / "clean.txt"
    path.write_text("clean\n", encoding="utf-8")
    result = run_check(path)
    assert result.returncode == 0, result.stdout + result.stderr
    path.write_text("", encoding="utf-8")
    assert run_check(path).returncode == 0


def test_reports_whitespace_and_missing_newline_without_modification(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    source = "first \nlast\t"
    path.write_text(source, encoding="utf-8")
    result = run_check(path)
    assert result.returncode == 1
    assert "bad.txt:1: trailing whitespace" in result.stdout
    assert "bad.txt:2: trailing whitespace" in result.stdout
    assert "missing final newline" in result.stdout
    assert path.read_text(encoding="utf-8") == source


def test_large_binary_is_rejected_before_text_filtering(tmp_path: Path) -> None:
    path = tmp_path / "large.bin"
    path.write_bytes(b"\0" * 1_000_001)
    result = run_check(path)
    assert result.returncode == 1
    assert "exceeds 1000000 bytes" in result.stdout


def test_small_binary_is_not_treated_as_whitespace(tmp_path: Path) -> None:
    path = tmp_path / "image.bin"
    path.write_bytes(b"\0 \t")
    result = run_check(path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_fixture_exception_requires_an_explicit_reason(tmp_path: Path) -> None:
    path = tmp_path / "tests" / "fixtures" / "large.bin"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\0" * 1_000_001)
    checker = load_checker()
    rel = "tests/fixtures/large.bin"
    assert checker.check_files(tmp_path, [path], {rel: "Exercises oversized parser input"}) == []
    assert checker.check_files(tmp_path, [path], {rel: ""})


def test_reason_does_not_exempt_files_outside_fixtures(tmp_path: Path) -> None:
    path = tmp_path / "large.bin"
    path.write_bytes(b"\0" * 1_000_001)
    checker = load_checker()
    assert checker.check_files(tmp_path, [path], {"large.bin": "Not a fixture"})
