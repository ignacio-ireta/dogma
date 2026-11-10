"""Tests for scripts/exercise_gate.py: clause labels, probe verdicts, notes, lint and usage errors.

The gate reuses the audit tool's discovery and grader. Both are replaced here with fakes, so no
test grades a real submission; the runbook's Accept commands do that.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from dogma.grading.models import GradeReport, RunStatus, StaticFinding
from dogma.grading.models import TestResult as GradedTest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "exercise_gate.py"
FIXTURE = ROOT / "tests" / "fixtures" / "problems" / "py-variants"
PROBLEM_ID = "t/x/p01-demo"

PROBLEM_MD = """# Demo

## Contract

Text before the list.

- A bullet that wraps
  onto a second line.
  - nested detail stays with its parent
- Second bullet.

## Notes

- not a clause

## Constraints

- Allowed: stdlib.
- Not allowed: none.
"""


@pytest.fixture(scope="module")
def gate() -> ModuleType:
    spec = importlib.util.spec_from_file_location("exercise_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _problem(gate: ModuleType, path: Path, problem_id: str = PROBLEM_ID):
    return gate.audit.Problem(problem_id, path, "python", "f", 30, ())


def _report(status, tests=(), static=()) -> GradeReport:
    return GradeReport(
        attempt_id="a",
        problem_id=PROBLEM_ID,
        problem_version=1,
        runtime="py-stdlib",
        status=status,
        tests=tuple(tests),
        static=tuple(static),
    )


@pytest.fixture
def repo_problem(gate: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """One problem, at tmp_path/p, that discover returns."""
    path = tmp_path / "p"
    path.mkdir()
    problem = _problem(gate, path)
    monkeypatch.setattr(gate.audit, "discover", lambda repo, only: [problem])
    return problem


def _grade_with(gate: ModuleType, monkeypatch, tmp_path: Path, report: GradeReport) -> Path:
    """Make the grader return ``report`` for any file; return a candidate file to probe."""
    problem = _problem(gate, tmp_path)
    monkeypatch.setattr(gate.audit, "discover", lambda repo, only: [problem])
    monkeypatch.setattr(
        gate.audit, "grade_source", lambda _problem, _source, _settings, **_kw: report
    )
    candidate = tmp_path / "candidate.py"
    candidate.write_text("def f(x):\n    return x\n", encoding="utf-8")
    return candidate


# --- text helpers -------------------------------------------------------------------------------


def test_contract_clauses_keep_wrapped_and_nested_text(gate: ModuleType):
    labels = gate.contract_clauses(PROBLEM_MD)
    assert labels == [
        ("C0", "Text before the list."),
        ("C1", "A bullet that wraps onto a second line. - nested detail stays with its parent"),
        ("C2", "Second bullet."),
        ("K1", "Allowed: stdlib."),
        ("K2", "Not allowed: none."),
    ]


def test_statement_without_contract_gives_only_constraint_labels(gate: ModuleType):
    md = "# T\n\n## Constraints\n\n- one\n- two\n"
    assert gate.contract_clauses(md) == [("K1", "one"), ("K2", "two")]


def test_clause_text_returns_label_and_refuses_unknown(gate: ModuleType):
    assert gate.clause_text(PROBLEM_MD, "C2") == "Second bullet."
    with pytest.raises(gate.GateError, match="no clause C9: run clauses"):
        gate.clause_text(PROBLEM_MD, "C9")


def test_fixture_clauses_are_substrings_of_the_statement(gate: ModuleType):
    md = (FIXTURE / "problem.md").read_text(encoding="utf-8")
    labels = [label for label, _ in gate.contract_clauses(md)]
    assert labels == ["C1", "C2", "C3", "K1", "K2"]
    for label in labels:
        assert gate.squash(gate.clause_text(md, label)) in gate.squash(md)


def test_plain_words_accepts_plain_and_refuses_markup(gate: ModuleType):
    assert gate.plain_words("divides by n, not n - 1") == "divides by n, not n - 1"
    for bad in ("uses `x`", 'say "hi"', "costs $5", "back\\slash"):
        with pytest.raises(gate.GateError, match="use plain words"):
            gate.plain_words(bad)


def test_squash_joins_whitespace(gate: ModuleType):
    assert gate.squash("  a\n  b\t c ") == "a b c"


# --- probe --------------------------------------------------------------------------------------


def test_probe_verdict_rows(gate: ModuleType):
    rejected = _report(
        RunStatus.REJECTED, static=[StaticFinding(rule="banned-import", message="uses numpy")]
    )
    assert gate.probe_verdict(rejected) == "broken (rejected: uses numpy)"
    assert gate.probe_verdict(_report(RunStatus.TIMEOUT)) == "infra (timeout)"
    assert gate.probe_verdict(_report(RunStatus.OOM)) == "infra (oom)"
    assert gate.probe_verdict(_report(RunStatus.ERROR)) == "infra (error)"
    assert gate.probe_verdict(_report(RunStatus.PASSED)) == "survived"
    failed_all = (GradedTest(name="test_a", file="public", outcome="fail"),)
    assert gate.probe_verdict(_report(RunStatus.FAILED, failed_all)) == "broken (no test passed)"
    mixed = (
        GradedTest(name="test_a", file="public", outcome="pass"),
        GradedTest(name="test_b", file="hidden", outcome="fail"),
    )
    assert gate.probe_verdict(_report(RunStatus.FAILED, mixed)) == "killed"


def test_probe_names_the_failing_public_test(gate, monkeypatch, tmp_path, capsys):
    tests = (
        GradedTest(name="test_a[1]", file="public", outcome="fail", message="wrong sum"),
        GradedTest(name="test_h", file="hidden", outcome="fail", message="AssertionError: x"),
        GradedTest(name="test_ok", file="public", outcome="pass"),
    )
    candidate = _grade_with(gate, monkeypatch, tmp_path, _report(RunStatus.FAILED, tests))
    out = tmp_path / "out"
    assert gate.main(["probe", "--out", str(out), PROBLEM_ID, str(candidate)]) == 0
    text = capsys.readouterr().out
    assert "VERDICT: killed" in text
    assert "killed_by: test_a (public)" in text
    assert "FAILED public test_a[1]" in text


def test_probe_names_the_hidden_test_when_no_public_test_fails(gate, monkeypatch, tmp_path, capsys):
    tests = (
        GradedTest(name="test_ok", file="public", outcome="pass"),
        GradedTest(name="test_h[big]", file="hidden", outcome="fail", message="AssertionError: x"),
    )
    candidate = _grade_with(gate, monkeypatch, tmp_path, _report(RunStatus.FAILED, tests))
    assert gate.main(["probe", "--out", str(tmp_path / "out"), PROBLEM_ID, str(candidate)]) == 0
    assert "killed_by: test_h (hidden)" in capsys.readouterr().out


def test_probe_missing_file_is_usage_error(gate, repo_problem, tmp_path):
    missing = tmp_path / "nope.py"
    assert gate.main(["probe", "--out", str(tmp_path / "out"), PROBLEM_ID, str(missing)]) == 2


# --- note and ledger ----------------------------------------------------------------------------


def test_note_then_ledger_round_trip(gate, repo_problem, tmp_path, capsys):
    out = tmp_path / "out"
    args = ["note", "--out", str(out), PROBLEM_ID]
    assert gate.main([*args, "--kind", "blocked", "--text", "no solution yet"]) == 0
    assert gate.main([*args, "--kind", "done", "--text", "checked twice"]) == 0
    capsys.readouterr()
    assert gate.main(["ledger", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "### blocked" in text
    assert f"- `{PROBLEM_ID}`: no solution yet" in text
    assert "ledger: 2 rows" in text
    rows = [json.loads(line) for line in (out / "ledger.jsonl").read_text().splitlines()]
    assert [row["kind"] for row in rows] == ["blocked", "done"]


def test_ledger_kind_filter(gate, repo_problem, tmp_path, capsys):
    out = tmp_path / "out"
    args = ["note", "--out", str(out), PROBLEM_ID]
    gate.main([*args, "--kind", "blocked", "--text", "no solution yet"])
    gate.main([*args, "--kind", "done", "--text", "checked twice"])
    capsys.readouterr()
    assert gate.main(["ledger", "--out", str(out), "--kind", "done"]) == 0
    text = capsys.readouterr().out
    assert "checked twice" in text
    assert "no solution yet" not in text
    assert "ledger: 1 rows" in text


def test_note_unknown_kind_is_usage_error(gate, repo_problem, tmp_path, capsys):
    out = tmp_path / "out"
    code = gate.main(["note", "--out", str(out), PROBLEM_ID, "--kind", "bogus", "--text", "x"])
    assert code == 2
    assert "unknown kind bogus" in capsys.readouterr().err
    assert not (out / "ledger.jsonl").exists()


def test_note_refuses_markup_text(gate, repo_problem, tmp_path, capsys):
    out = tmp_path / "out"
    code = gate.main(["note", "--out", str(out), PROBLEM_ID, "--kind", "done", "--text", "`x`"])
    assert code == 1
    assert "use plain words" in capsys.readouterr().err
    assert not (out / "ledger.jsonl").exists()


# --- lint ---------------------------------------------------------------------------------------


def _write_hidden(directory: Path, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "test_hidden.py").write_text(body, encoding="utf-8")
    return directory


def test_lint_reports_the_unclean_file_only(gate, tmp_path, monkeypatch, capsys):
    clean = _write_hidden(tmp_path / "clean", "def test_ok():\n    assert 1 == 1, 'a wrong sum'\n")
    unclean = _write_hidden(tmp_path / "unclean", "def test_bad():\n    assert 1 == 2\n")
    problems = [
        _problem(gate, clean, "t/x/p01-clean"),
        _problem(gate, unclean, "t/x/p02-unclean"),
    ]
    monkeypatch.setattr(gate.audit, "discover", lambda repo, only: problems)
    out = tmp_path / "out"
    assert gate.main(["lint", "--out", str(out)]) == 1
    text = capsys.readouterr().out
    assert "t/x/p02-unclean\tHIDDEN_ASSERT_MESSAGE\t2\t" in text
    assert "t/x/p01-clean" not in text
    assert "lint: 1 findings in 1 problems" in text
    rows = [json.loads(line) for line in (out / "lint.jsonl").read_text().splitlines()]
    assert [row["problem"] for row in rows] == ["t/x/p02-unclean"]


def test_lint_exits_zero_when_clean(gate, tmp_path, monkeypatch, capsys):
    clean = _write_hidden(tmp_path / "clean", "def test_ok():\n    assert 1 == 1, 'a wrong sum'\n")
    monkeypatch.setattr(gate.audit, "discover", lambda repo, only: [_problem(gate, clean)])
    out = tmp_path / "out"
    assert gate.main(["lint", "--out", str(out)]) == 0
    assert "lint: 0 findings in 0 problems" in capsys.readouterr().out
    assert (out / "lint.jsonl").read_text() == ""


# --- clauses and usage --------------------------------------------------------------------------


def test_clauses_command_prints_labels(gate, monkeypatch, tmp_path, capsys):
    problem = _problem(gate, FIXTURE, "fixtures/py-variants/p01-add")
    monkeypatch.setattr(gate.audit, "discover", lambda repo, only: [problem])
    assert gate.main(["clauses", "--out", str(tmp_path / "out"), problem.id]) == 0
    text = capsys.readouterr().out
    assert text.startswith("CLAUSES fixtures/py-variants/p01-add\n")
    assert "C1  " in text and "C3  " in text and "K2  Not allowed: none." in text
    assert "Give a label (C2, K1) when a command asks for --clause." in text


def test_unknown_problem_is_usage_error_with_close_match(gate, repo_problem, tmp_path, capsys):
    code = gate.main(["clauses", "--out", str(tmp_path / "out"), "t/x/p01-dem"])
    assert code == 2
    err = capsys.readouterr().err
    assert "no problem t/x/p01-dem" in err
    assert PROBLEM_ID in err
