"""The exercise gate's writers against the real grader, on the fixture problem.

Each test copies the fixture into a scratch repo and runs the gate's commands on it. Nothing here
is faked: a candidate is graded by the same sandboxed runner that learners use.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "exercise_gate.py"
FIXTURE = ROOT / "tests" / "fixtures" / "problems" / "py-variants"
FIXTURE_REL = "fixtures/py-variants/p01-add"
SOLUTION = (FIXTURE / "solution.py").read_text(encoding="utf-8")
NO_GUARD = "def add(a, b):\n    return a + b\n"


@pytest.fixture(scope="module")
def gate() -> ModuleType:
    spec = importlib.util.spec_from_file_location("exercise_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def problem_dir(gate: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    target = root / "content" / "fixtures" / "py-variants" / "problems" / "p01-add"
    shutil.copytree(FIXTURE, target, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(gate.audit, "REPO", root)
    return target


def _candidate(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / "candidates" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_add_accept_of_a_correct_rewrite_succeeds(gate, problem_dir, tmp_path, capsys):
    rewrite = SOLUTION.replace("return a + b", "return b + a")
    src = _candidate(tmp_path, "accept_swapped.py", rewrite)
    argv = [
        "add-accept",
        "--out",
        str(tmp_path / "out"),
        FIXTURE_REL,
        str(src),
        "--origin",
        "freedom",
        "--freedom",
        "adds in the other order",
    ]
    assert gate.main(argv) == 0
    assert "added accept accept_swapped.py (freedom)" in capsys.readouterr().out
    assert (problem_dir / "variants" / "accept_swapped.py").is_file()


def test_add_reject_of_a_missing_guard_is_killed_by_the_text_test(
    gate, problem_dir, tmp_path, capsys
):
    src = _candidate(tmp_path, "reject_no_guard.py", NO_GUARD)
    argv = [
        "add-reject",
        "--out",
        str(tmp_path / "out"),
        FIXTURE_REL,
        str(src),
        "--mistake",
        "accepts text and joins it",
        "--clause",
        "C3",
    ]
    assert gate.main(argv) == 0
    text = capsys.readouterr().out
    assert "killed_by test_text_raises_type_error (hidden)" in text
    entry = gate.read_manifest(gate.find_problem(FIXTURE_REL)).reject[0]
    assert (entry.killed_by, entry.visible, entry.clause) == (
        "test_text_raises_type_error",
        "hidden",
        gate.clause_text((problem_dir / "problem.md").read_text(encoding="utf-8"), "C3"),
    )


def test_add_reject_without_the_bool_check_survives_and_is_refused(
    gate, problem_dir, tmp_path, capsys
):
    no_bool = SOLUTION.replace("isinstance(value, bool) or ", "")
    src = _candidate(tmp_path, "reject_bool.py", no_bool)
    argv = [
        "add-reject",
        "--out",
        str(tmp_path / "out"),
        FIXTURE_REL,
        str(src),
        "--mistake",
        "counts a bool as a number",
        "--clause",
        "C1",
    ]
    assert gate.main(argv) == 1
    assert "SURVIVED" in capsys.readouterr().err
    assert not (problem_dir / "variants" / "reject_bool.py").exists()
    assert not (problem_dir / "variants" / "manifest.yaml").exists()


def test_variants_passes_then_fails_when_a_killer_is_wrong(gate, problem_dir, tmp_path, capsys):
    out = tmp_path / "out"
    accept = _candidate(tmp_path, "accept_swapped.py", SOLUTION.replace("a + b", "b + a"))
    add_accept = [
        "add-accept",
        "--out",
        str(out),
        FIXTURE_REL,
        str(accept),
        "--origin",
        "freedom",
        "--freedom",
        "adds in the other order",
    ]
    assert gate.main(add_accept) == 0
    reject = _candidate(tmp_path, "reject_no_guard.py", NO_GUARD)
    add_reject = [
        "add-reject",
        "--out",
        str(out),
        FIXTURE_REL,
        str(reject),
        "--mistake",
        "accepts text and joins it",
        "--clause",
        "C3",
    ]
    assert gate.main(add_reject) == 0
    capsys.readouterr()

    assert gate.main(["variants", "--out", str(out)]) == 0
    assert "variants: 2 ok, 0 mismatches in 1 problems" in capsys.readouterr().out

    manifest = problem_dir / "variants" / "manifest.yaml"
    text = manifest.read_text(encoding="utf-8")
    manifest.write_text(
        text.replace(
            "killed_by: test_text_raises_type_error", "killed_by: test_sum_of_negative_ints"
        ),
        encoding="utf-8",
    )
    assert gate.main(["variants", "--out", str(out)]) == 1
    assert "MISMATCH: test_sum_of_negative_ints does not fail" in capsys.readouterr().out
