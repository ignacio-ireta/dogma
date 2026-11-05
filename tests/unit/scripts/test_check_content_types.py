"""Exercise the checker against scratch content, never learner workspaces."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "check_content_types.py"


def problem(root: Path, slug: str, solution: str | None = None) -> Path:
    path = root / "content" / "test" / slug / "problems" / "p01-example"
    path.mkdir(parents=True)
    if solution is not None:
        (path / "solution.py").write_text(solution, encoding="utf-8")
    for name in ("test_public.py", "test_hidden.py"):
        (path / name).write_text("", encoding="utf-8")
    return path


def run_check(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_independent_submissions_resolve_real_types_without_executing_them(tmp_path: Path) -> None:
    for slug, kind, value in (("one", "int", "1"), ("two", "str", "'two'")):
        path = problem(
            tmp_path, slug, f"def value() -> {kind}:\n    return {value}\nraise RuntimeError()\n"
        )
        (path / "test_public.py").write_text(
            f"from submission import value\ndef test_value() -> None:\n"
            f"    result: {kind} = value()\n    assert result == {value}\n",
            encoding="utf-8",
        )
        (path / "starter.py").write_text("broken: int = 'starter'\n", encoding="utf-8")
    result = run_check(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 problem" in result.stdout


def test_annotated_solution_error_names_original_file(tmp_path: Path) -> None:
    problem(tmp_path, "broken", "def value() -> int:\n    return 'wrong'\n")
    result = run_check(tmp_path)
    assert result.returncode == 1
    assert "content/test/broken/problems/p01-example/solution.py:2:" in result.stdout
    assert "return-value" in result.stdout


def test_test_body_resolves_submission_signature(tmp_path: Path) -> None:
    path = problem(tmp_path, "wrong-call", "def value(number: int) -> int:\n    return number\n")
    (path / "test_hidden.py").write_text(
        "from submission import value\ndef test_value() -> None:\n    value('wrong')\n",
        encoding="utf-8",
    )
    result = run_check(tmp_path)
    assert result.returncode == 1
    assert "test_hidden.py:3:" in result.stdout
    assert "arg-type" in result.stdout


def test_sql_problem_python_tests_are_checked(tmp_path: Path) -> None:
    path = problem(tmp_path, "sql")
    (path / "solution.sql").write_text("SELECT 1;\n", encoding="utf-8")
    (path / "test_public.py").write_text("wrong: int = 'value'\n", encoding="utf-8")
    result = run_check(tmp_path)
    assert result.returncode == 1
    assert "test_public.py:1:" in result.stdout


def test_plain_configuration_accepts_unannotated_functions(tmp_path: Path) -> None:
    problem(tmp_path, "plain", "def value(number):\n    return number\n")
    result = run_check(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_missing_content_fails_instead_of_checking_nothing(tmp_path: Path) -> None:
    result = run_check(tmp_path)
    assert result.returncode == 2
    assert "no problem packages" in result.stderr
    assert "Traceback" not in result.stderr


def test_missing_required_test_is_an_input_error(tmp_path: Path) -> None:
    path = problem(tmp_path, "incomplete", "value = 1\n")
    (path / "test_hidden.py").unlink()
    result = run_check(tmp_path)
    assert result.returncode == 2
    assert "test_hidden.py" in result.stderr
    assert "Traceback" not in result.stderr
