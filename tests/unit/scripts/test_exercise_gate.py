"""Tests for scripts/exercise_gate.py: clause labels, probe verdicts, notes, lint and usage errors.

The gate reuses the audit tool's discovery and grader. Both are replaced here with fakes, so no
test grades a real submission; the runbook's Accept commands do that.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from datetime import date, timedelta
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


# --- manifest writers ---------------------------------------------------------------------------

FIXTURE_REL = "fixtures/py-variants/p01-add"
PLAIN = "def add(a, b):\n    return a + b\n"
BUG = "def add(a, b):\n    return a + b  # BUG\n"
PUBLIC_BUG = "def add(a, b):\n    return a + b  # PUBLICBUG\n"
BOOM = "def add(a, b):\n    return a + b  # BOOM\n"
BROKEN = "def add(a, b):\n    return a  # BROKEN\n"
SLOW = "def add(a, b):\n    return a  # SLOW\n"


def _fake_report(source: str) -> GradeReport:
    """What the fake grader says, from the marker in the submitted source."""
    if "SLOW" in source:
        return _report(RunStatus.TIMEOUT)
    if "PUBLICBUG" in source:
        return _report(
            RunStatus.FAILED,
            (
                GradedTest(
                    name="test_example_adds_two_ints",
                    file="public",
                    outcome="fail",
                    message="add(2, 3) should be 5",
                ),
                GradedTest(name="test_large_ints_stay_exact", file="hidden", outcome="pass"),
            ),
        )
    if "BUG" in source or "BOOM" in source:
        message = "ValueError: boom" if "BOOM" in source else "large ints lost precision"
        return _report(
            RunStatus.FAILED,
            (
                GradedTest(name="test_example_adds_two_ints", file="public", outcome="pass"),
                GradedTest(
                    name="test_large_ints_stay_exact",
                    file="hidden",
                    outcome="fail",
                    message=message,
                ),
            ),
        )
    if "BROKEN" in source:
        return _report(
            RunStatus.FAILED,
            (
                GradedTest(name="test_example_adds_two_ints", file="public", outcome="fail"),
                GradedTest(name="test_large_ints_stay_exact", file="hidden", outcome="fail"),
            ),
        )
    return _report(
        RunStatus.PASSED,
        (
            GradedTest(name="test_example_adds_two_ints", file="public", outcome="pass"),
            GradedTest(name="test_large_ints_stay_exact", file="hidden", outcome="pass"),
        ),
    )


@pytest.fixture
def problem_dir(gate: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The fixture problem copied into a scratch repo, which the gate's audit tool reads."""
    root = tmp_path / "repo"
    target = root / "content" / "fixtures" / "py-variants" / "problems" / "p01-add"
    shutil.copytree(FIXTURE, target, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(gate.audit, "REPO", root)
    return target


@pytest.fixture
def grader(gate: ModuleType, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the real grader with the fake. Returns the sources it was asked to grade."""
    calls: list[str] = []

    def grade(_problem, source, _settings, **_kw):
        calls.append(source)
        return _fake_report(source)

    monkeypatch.setattr(gate.audit, "grade_source", grade)
    return calls


def _candidate(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / "candidates" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _reject_argv(out: Path, source: Path, *extra: str) -> list[str]:
    base = [
        "add-reject",
        "--out",
        str(out),
        FIXTURE_REL,
        str(source),
        "--mistake",
        "drops the bool check",
        "--clause",
        "C3",
    ]
    return [*base, *extra]


def _manifest_of(gate: ModuleType, problem_dir: Path):
    return gate.read_manifest(gate.find_problem(FIXTURE_REL))


def test_add_reject_records_the_hidden_test_that_kills(gate, problem_dir, grader, tmp_path, capsys):
    src = _candidate(tmp_path, "reject_no_overflow.py", BUG)
    out = tmp_path / "out"
    assert gate.main(_reject_argv(out, src)) == 0
    text = capsys.readouterr().out
    assert "added reject reject_no_overflow.py: killed_by test_large_ints_stay_exact" in text
    assert "(hidden), message kind named" in text
    manifest = _manifest_of(gate, problem_dir)
    assert manifest.problem_version == 1
    entry = manifest.reject[0]
    assert (entry.file, entry.killed_by, entry.visible) == (
        "reject_no_overflow.py",
        "test_large_ints_stay_exact",
        "hidden",
    )
    md = (problem_dir / "problem.md").read_text(encoding="utf-8")
    assert entry.clause == gate.clause_text(md, "C3")
    assert (problem_dir / "variants" / "reject_no_overflow.py").is_file()
    assert not (out / "ledger.jsonl").exists()


def test_add_reject_prefers_a_public_test(gate, problem_dir, grader, tmp_path):
    src = _candidate(tmp_path, "reject_public.py", PUBLIC_BUG)
    assert gate.main(_reject_argv(tmp_path / "out", src)) == 0
    entry = _manifest_of(gate, problem_dir).reject[0]
    assert (entry.killed_by, entry.visible) == ("test_example_adds_two_ints", "public")


REJECT_REFUSALS = [
    # (case, file name, source, extra arguments, exit code, text the output must hold)
    ("survives", "reject_plain.py", PLAIN, [], 1, "SURVIVED"),
    ("broken", "reject_broken.py", BROKEN, [], 1, "NOT KILLED"),
    ("timeout", "reject_slow.py", SLOW, [], 1, "NOT KILLED"),
    ("unknown clause", "reject_bug.py", BUG, ["--clause", "C9"], 1, "no clause C9"),
    ("markup", "reject_bug.py", BUG, ["--mistake", "uses `x`"], 1, "use plain words"),
    ("bad name", "bad.py", BUG, [], 2, "bad variant file name"),
    ("wrong suffix", "reject_bug.sql", BUG, [], 2, "bad variant file name"),
    ("not failing", "reject_bug.py", BUG, ["--killed-by", "test_sum_x"], 1, "does not fail"),
    ("message absent", "reject_bug.py", BUG, ["--message-has", "overflow"], 1, "shows"),
]


@pytest.mark.usefixtures("grader")
@pytest.mark.parametrize("row", REJECT_REFUSALS, ids=[row[0] for row in REJECT_REFUSALS])
def test_add_reject_refusals_leave_no_trace(gate, problem_dir, tmp_path, capsys, row):
    case, name, body, extra, code, text = row
    src = _candidate(tmp_path, name, body)
    out = tmp_path / "out"
    assert gate.main(_reject_argv(out, src, *extra)) == code, case
    captured = capsys.readouterr()
    assert text in captured.err + captured.out, case
    assert not (problem_dir / "variants").exists(), case
    assert src.is_file()


def test_add_reject_refuses_a_file_inside_variants(gate, problem_dir, grader, tmp_path):
    inside = problem_dir / "variants" / "reject_inside.py"
    inside.parent.mkdir()
    inside.write_text(BUG, encoding="utf-8")
    assert gate.main(_reject_argv(tmp_path / "out", inside)) == 2
    assert not (problem_dir / "variants" / "manifest.yaml").exists()
    assert inside.is_file()


def test_refused_add_keeps_the_existing_manifest(gate, problem_dir, grader, tmp_path):
    out = tmp_path / "out"
    first = _candidate(tmp_path, "reject_first.py", BUG)
    assert gate.main(_reject_argv(out, first)) == 0
    before = (problem_dir / "variants" / "manifest.yaml").read_bytes()
    second = _candidate(tmp_path, "reject_second.py", BROKEN)
    assert gate.main(_reject_argv(out, second)) == 1
    assert (problem_dir / "variants" / "manifest.yaml").read_bytes() == before
    assert not (problem_dir / "variants" / "reject_second.py").exists()


def test_message_defect_is_noted_in_the_ledger(gate, problem_dir, grader, tmp_path):
    src = _candidate(tmp_path, "reject_boom.py", BOOM)
    out = tmp_path / "out"
    assert gate.main(_reject_argv(out, src)) == 0
    rows = [json.loads(line) for line in (out / "ledger.jsonl").read_text().splitlines()]
    assert [row["kind"] for row in rows] == ["message-defect"]
    assert "reject_boom.py: test_large_ints_stay_exact shows kind=bare_exception" in rows[0]["text"]


def test_add_accept_records_a_passing_solution(gate, problem_dir, grader, tmp_path, capsys):
    src = _candidate(tmp_path, "accept_loop.py", PLAIN)
    argv = [
        "add-accept",
        "--out",
        str(tmp_path / "out"),
        FIXTURE_REL,
        str(src),
        "--origin",
        "freedom",
        "--freedom",
        "uses a loop",
    ]
    assert gate.main(argv) == 0
    assert "added accept accept_loop.py (freedom)" in capsys.readouterr().out
    entry = _manifest_of(gate, problem_dir).accept[0]
    assert (entry.origin, entry.freedom) == ("freedom", "uses a loop")


def test_add_accept_refuses_a_failing_solution(gate, problem_dir, grader, tmp_path, capsys):
    src = _candidate(tmp_path, "accept_bug.py", BUG)
    argv = ["add-accept", "--out", str(tmp_path / "out"), FIXTURE_REL, str(src)]
    assert gate.main([*argv, "--origin", "freedom", "--freedom", "x"]) == 1
    assert "NOT ACCEPTED" in capsys.readouterr().err
    assert not (problem_dir / "variants").exists()


@pytest.mark.parametrize(
    "extra",
    [
        ["--origin", "blind"],  # a blind accept has a fixed name
        ["--origin", "freedom"],  # freedom needs its text
        ["--origin", "blind", "--freedom", "x"],  # blind takes no freedom text
    ],
)
def test_add_accept_argument_mistakes_are_usage_errors(gate, problem_dir, grader, tmp_path, extra):
    src = _candidate(tmp_path, "accept_other.py", PLAIN)
    argv = ["add-accept", "--out", str(tmp_path / "out"), FIXTURE_REL, str(src), *extra]
    assert gate.main(argv) == 2
    assert not (problem_dir / "variants").exists()


def test_add_uncovered_and_remove_variant(gate, problem_dir, grader, tmp_path, capsys):
    out = tmp_path / "out"
    uncovered = [
        "add-uncovered",
        "--out",
        str(out),
        FIXTURE_REL,
        "--clause",
        "C2",
        "--reason",
        "sums are exact by the contract",
    ]
    assert gate.main(uncovered) == 0
    assert "added uncovered C2" in capsys.readouterr().out
    md = (problem_dir / "problem.md").read_text(encoding="utf-8")
    assert _manifest_of(gate, problem_dir).uncovered[0].clause == gate.clause_text(md, "C2")

    src = _candidate(tmp_path, "accept_loop.py", PLAIN)
    add = ["add-accept", "--out", str(out), FIXTURE_REL, str(src)]
    assert gate.main([*add, "--origin", "freedom", "--freedom", "a loop"]) == 0
    remove = ["remove-variant", "--out", str(out), FIXTURE_REL, "accept_loop.py"]
    assert gate.main(remove) == 0
    assert capsys.readouterr().out.endswith("removed accept_loop.py\n")
    assert _manifest_of(gate, problem_dir).accept == ()
    assert not (problem_dir / "variants" / "accept_loop.py").exists()

    assert gate.main(["remove-variant", "--out", str(out), FIXTURE_REL, "accept_gone.py"]) == 2


def test_prune_uncovered_drops_only_entries_a_reject_now_covers(
    gate, problem_dir, grader, tmp_path, capsys
):
    out = tmp_path / "out"
    for label in ("C2", "C3"):
        argv = ["add-uncovered", "--out", str(out), FIXTURE_REL, "--clause", label]
        assert gate.main([*argv, "--reason", "left for a later sweep"]) == 0
    src = _candidate(tmp_path, "reject_bug.py", BUG)
    assert gate.main(_reject_argv(out, src)) == 0  # a reject for C3
    md = (problem_dir / "problem.md").read_text(encoding="utf-8")
    capsys.readouterr()

    assert gate.main(["prune-uncovered", "--out", str(out)]) == 0

    assert "prune-uncovered: 1 entries dropped" in capsys.readouterr().out
    left = _manifest_of(gate, problem_dir).uncovered
    assert [entry.clause for entry in left] == [gate.clause_text(md, "C2")]
    assert gate.main(["prune-uncovered", "--out", str(out)]) == 0
    assert "prune-uncovered: 0 entries dropped" in capsys.readouterr().out


def test_remove_of_the_last_entry_deletes_manifest_and_directory(
    gate, problem_dir, grader, tmp_path
):
    out = tmp_path / "out"
    src = _candidate(tmp_path, "reject_bug.py", BUG)
    assert gate.main(_reject_argv(out, src)) == 0
    assert gate.main(["remove-variant", "--out", str(out), FIXTURE_REL, "reject_bug.py"]) == 0
    assert not (problem_dir / "variants").exists()


def _blind_submit(out: Path, problem_id: str, text: str, verdict: str = "survived", sha=None):
    solution = out / "blind" / problem_id / "solution.py"
    solution.parent.mkdir(parents=True, exist_ok=True)
    solution.write_text(text, encoding="utf-8")
    row = {
        "problem": problem_id,
        "verdict": verdict,
        "source_sha": sha if sha is not None else _text_sha(text),
    }
    with (out / "blind.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def test_adopt_blind_adopts_a_passing_submission(gate, problem_dir, grader, tmp_path, capsys):
    out = tmp_path / "out"
    _blind_submit(out, FIXTURE_REL, PLAIN)
    assert gate.main(["adopt-blind", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert f"{FIXTURE_REL}  adopted  similarity=" in text
    assert "adopt-blind: 1 adopted, 0 skipped" in text
    entry = _manifest_of(gate, problem_dir).accept[0]
    assert (entry.file, entry.origin) == ("accept_blind.py", "blind")


@pytest.mark.usefixtures("grader")
@pytest.mark.parametrize(
    "kwargs",
    [{"sha": "0" * 16}, {"verdict": "killed_hidden"}],
    ids=["sha differs", "not survived"],
)
def test_adopt_blind_skips_a_submission_that_does_not_match(
    gate, problem_dir, tmp_path, capsys, kwargs
):
    out = tmp_path / "out"
    _blind_submit(out, FIXTURE_REL, PLAIN, **kwargs)
    assert gate.main(["adopt-blind", "--out", str(out)]) == 0
    reason = "no passing submit for this file"
    assert f"{FIXTURE_REL}  skipped: {reason}" in capsys.readouterr().out
    assert not (problem_dir / "variants").exists()


def test_adopt_blind_skips_a_problem_without_a_file(gate, problem_dir, grader, tmp_path, capsys):
    out = tmp_path / "out"
    out.mkdir()
    assert gate.main(["adopt-blind", "--out", str(out)]) == 0
    assert f"{FIXTURE_REL}  skipped: no file" in capsys.readouterr().out


def test_variants_grades_once_then_reuses_its_rows(gate, problem_dir, grader, tmp_path, capsys):
    out = tmp_path / "out"
    accept = _candidate(tmp_path, "accept_loop.py", PLAIN)
    add = ["add-accept", "--out", str(out), FIXTURE_REL, str(accept)]
    assert gate.main([*add, "--origin", "freedom", "--freedom", "a loop"]) == 0
    reject = _candidate(tmp_path, "reject_bug.py", BUG)
    assert gate.main(_reject_argv(out, reject)) == 0
    grader.clear()
    assert gate.main(["variants", "--out", str(out)]) == 0
    assert len(grader) == 2
    assert "variants: 2 ok, 0 mismatches in 1 problems" in capsys.readouterr().out
    grader.clear()
    assert gate.main(["variants", "--out", str(out)]) == 0
    assert grader == []


def test_variants_reports_a_reject_that_now_survives(gate, problem_dir, grader, tmp_path, capsys):
    out = tmp_path / "out"
    reject = _candidate(tmp_path, "reject_bug.py", BUG)
    assert gate.main(_reject_argv(out, reject)) == 0
    installed = problem_dir / "variants" / "reject_bug.py"
    installed.write_text(PLAIN, encoding="utf-8")  # the wrong solution no longer has the marker
    assert gate.main(["variants", "--out", str(out)]) == 1
    text = capsys.readouterr().out
    assert "reject_bug.py  expect=fail  got=survived  MISMATCH: no test fails" in text
    assert "variants: 0 ok, 1 mismatches in 1 problems" in text


def test_variants_with_no_manifest_checks_nothing(gate, problem_dir, grader, tmp_path, capsys):
    assert gate.main(["variants", "--out", str(tmp_path / "out")]) == 0
    assert "variants: 0 ok, 0 mismatches in 0 problems" in capsys.readouterr().out
    assert grader == []


@pytest.mark.usefixtures("problem_dir")
def test_blind_similarity_ignores_lines_the_starter_already_has(gate):
    problem = gate.find_problem(FIXTURE_REL)
    reference = problem.solution.read_text(encoding="utf-8")
    ratio, own_lines = gate.blind_similarity(problem, reference)
    assert ratio == 1.0
    # The signature and the docstring come from the starter, so only the body is counted.
    assert own_lines == 4
    other = (
        "from __future__ import annotations\n\n\n"
        "def add(a: int | float, b: int | float) -> int | float:\n"
        '    """Return a + b, refusing anything that is not an int or a float."""\n'
        "    if type(a) not in (int, float) or type(b) not in (int, float):\n"
        '        raise TypeError("numbers only")\n'
        "    return a + b\n"
    )
    ratio, _ = gate.blind_similarity(problem, other)
    assert ratio < 0.5


# --- the gate: normalising tests ----------------------------------------------------------------

BASE_TEST = (
    "import pytest\n"
    "from hypothesis import given, settings\n"
    "\n"
    "@settings(deadline=None, report_multiple_bugs=False, max_examples=50)\n"
    "def test_close(x):\n"
    '    """Docstring."""\n'
    '    assert x == 1, "msg"  # comment\n'
    '    np.testing.assert_allclose(x, 1.0, rtol=1e-6, err_msg="old")\n'
    '    pytest.fail("old wording")\n'
    '@pytest.mark.parametrize("v", [1, 2], ids=["one", "two"])\n'
    "def test_other(v):\n"
    '    raise AssertionError("old")\n'
)


@pytest.mark.parametrize(
    "edit",
    [
        lambda s: s.replace('"msg"', '"reworded"'),
        lambda s: s.replace('err_msg="old"', 'err_msg="new text"'),
        lambda s: s.replace('ids=["one", "two"]', 'ids=["a", "b"]'),
        lambda s: s.replace("max_examples=50", "max_examples=50, deadline=None"),
        lambda s: s.replace('pytest.fail("old wording")', 'pytest.fail("other")'),
        lambda s: s.replace('AssertionError("old")', 'AssertionError("new")'),
        lambda s: s.replace('    """Docstring."""', "    '''Other.'''"),
        lambda s: s.replace("# comment", "# changed"),
        lambda s: s.replace(
            "from hypothesis import given, settings", "from hypothesis import given"
        ),
    ],
)
def test_normalise_ignores_messages_ids_docstrings_and_settings(gate, edit):
    assert gate.normalise_tests(edit(BASE_TEST)) == gate.normalise_tests(BASE_TEST)


def test_normalise_ignores_a_settings_decorator_added_whole(gate):
    bare = "def test_a(x):\n    assert x\n"
    added = (
        "from hypothesis import settings\n"
        "@settings(deadline=None, report_multiple_bugs=False)\n"
        "def test_a(x):\n"
        "    assert x\n"
    )
    assert gate.normalise_tests(added) == gate.normalise_tests(bare)


@pytest.mark.parametrize(
    "edit",
    [
        lambda s: s.replace("rtol=1e-6", "rtol=1e-3"),
        lambda s: s.replace("max_examples=50", "max_examples=5"),
        lambda s: s.replace('    assert x == 1, "msg"', "    pass"),
        lambda s: s.replace("def test_close", "def test_renamed"),
        lambda s: s.replace("[1, 2]", "[1, 3]"),
        lambda s: s.replace("def test_other", "@pytest.mark.skip\ndef test_other"),
        lambda s: s.replace("x == 1", "x == 2"),
    ],
)
def test_normalise_sees_checks_values_names_and_skips(gate, edit):
    assert gate.normalise_tests(edit(BASE_TEST)) != gate.normalise_tests(BASE_TEST)


def test_normalise_sees_a_helper_call_replacing_pytest_raises(gate):
    raising = "import pytest\n\ndef test_x():\n    with pytest.raises(TypeError):\n        f()\n"
    helper = "import pytest\n\ndef test_x():\n    _expect_type_error(f)\n"
    assert gate.normalise_tests(raising) != gate.normalise_tests(helper)


def test_duplicate_top_level_names_and_test_function_names(gate):
    source = (
        "X = 1\nX = 2\n\ndef test_a():\n    pass\n\nclass Helper:\n    pass\n\n"
        "def test_a():\n    pass\n\ndef helper():\n    pass\n"
    )
    assert gate.duplicate_top_level_names(source) == ["X", "test_a"]
    assert gate.duplicate_top_level_names("def f():\n    pass\n") == []
    assert gate.test_function_names(source) == {"test_a"}
    assert gate.test_function_names("async def test_b():\n    pass\n") == {"test_b"}


# --- the gate: a real git repository holding the fixture problem --------------------------------

FIXTURE_ID = "fixtures/py-variants/p01-add"
PROBLEM_REL = Path("content/fixtures/py-variants/problems/p01-add")
HIDDEN_OLD = (FIXTURE / "test_hidden.py").read_text(encoding="utf-8")


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


def _replace(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{old!r} not in {path.name}"
    path.write_text(text.replace(old, new), encoding="utf-8")


def _append(path: Path, text: str) -> None:
    path.write_text(path.read_text(encoding="utf-8") + text, encoding="utf-8")


CATCH_TEST = (
    "\n\ndef test_catches_bug():  # CATCH\n"
    '    assert add(True, 1) == 2, "the sum is wrong for bools"\n'
)


@pytest.fixture
def gate_repo(gate: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The fixture problem, committed as the base. Returns the problem directory."""
    root = tmp_path / "repo"
    target = root / PROBLEM_REL
    shutil.copytree(FIXTURE, target, ignore=shutil.ignore_patterns("__pycache__"))
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    problem = gate.audit.Problem(
        id=FIXTURE_ID,
        path=target,
        language="python",
        entry="add",
        timeout_s=30,
        banned=(),
        runtime="py-stdlib",
    )
    monkeypatch.setattr(gate.audit, "REPO", root)
    monkeypatch.setattr(gate.audit, "discover", lambda repo, only: [problem])
    monkeypatch.setattr(gate, "_run_seed", lambda *_args: (True, []))
    return target


def _test_names(text: str | None) -> list[str]:
    """Test names as the fake grader reports them: one name per parametrize case id."""
    if text is None:
        return []
    names: list[str] = []
    for node in ast.parse(text).body:
        if not (isinstance(node, ast.FunctionDef) and node.name.startswith("test_")):
            continue
        cases = [
            "-".join(str(v) for v in value) if isinstance(value, tuple) else str(value)
            for decorator in node.decorator_list
            if isinstance(decorator, ast.Call)
            and getattr(decorator.func, "attr", "") == "parametrize"
            for value in ast.literal_eval(decorator.args[1])
        ]
        if cases:
            names.extend(f"{node.name}[{case}]" for case in cases)
        else:
            names.append(node.name)
    return names


def _read(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.is_file() else None


def _fake_gate_grade(problem, source: str, _settings, **_kw) -> GradeReport:
    """Results depend on the tests given: BREAKREF breaks the reference, BUG + CATCH is caught."""
    public_names = _test_names(_read(problem.path / "test_public.py"))
    hidden_text = _read(problem.path / "test_hidden.py") or ""
    tests = [GradedTest(name=name, file="public", outcome="pass") for name in public_names]
    for index, name in enumerate(_test_names(hidden_text)):
        if "BREAKREF" in hidden_text and index == 0:
            tests.append(GradedTest(name=name, file="hidden", outcome="fail", message="broken"))
        elif (
            "BUG" in source
            and "CATCH" in hidden_text
            and name.partition("[")[0] == "test_catches_bug"
        ):
            tests.append(
                GradedTest(
                    name=name,
                    file="hidden",
                    outcome="fail",
                    message="the sum is wrong for bools",
                )
            )
        else:
            tests.append(GradedTest(name=name, file="hidden", outcome="pass"))
    failed = any(test.outcome != "pass" for test in tests)
    return _report(RunStatus.FAILED if failed else RunStatus.PASSED, tests)


@pytest.fixture
def gate_grader(gate: ModuleType, gate_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate.audit, "grade_source", _fake_gate_grade)


def _checks(capsys) -> dict[str, str]:
    lines = capsys.readouterr().out.splitlines()
    found = (re.match(r"^(\w+)\s+(PASS|FAIL|SKIP)\b", line) for line in lines)
    return {match.group(1): match.group(2) for match in found if match}


ALL_PASS = {
    "frozen": "PASS",
    "manifest": "PASS",
    "reference": "PASS",
    "accepts": "PASS",
    "rejects": "PASS",
    "change": "PASS",
    "wrong": "PASS",
    "version": "PASS",
    "lint": "PASS",
    "names": "PASS",
    "seeds": "PASS",
    "time": "PASS",
}


def _run(gate: ModuleType, out: Path, *extra: str) -> int:
    return gate.main(["gate", FIXTURE_ID, "--out", str(out), *extra])


def test_gate_mode_none_passes_an_unchanged_problem(gate, gate_grader, tmp_path, capsys):
    assert _run(gate, tmp_path / "out") == 0
    checks = _checks(capsys)
    assert checks == {**ALL_PASS, "wrong": "SKIP"}


def test_gate_frozen_edit_fails_and_skips_the_rest(gate, gate_grader, gate_repo, tmp_path, capsys):
    _replace(gate_repo / "solution.py", "return a + b", "return b + a")
    assert _run(gate, tmp_path / "out") == 1
    checks = _checks(capsys)
    assert checks["frozen"] == "FAIL"
    assert set(list(checks.values())[1:]) == {"SKIP"}


def test_gate_stale_manifest_fails_the_manifest_check(
    gate, gate_grader, gate_repo, tmp_path, capsys
):
    gate.write_manifest(gate.find_problem(FIXTURE_ID), gate.VariantsManifest(problem_version=7))
    assert _run(gate, tmp_path / "out") == 1
    checks = _checks(capsys)
    assert checks["manifest"] == "FAIL"
    assert checks["reference"] == "SKIP"


def test_gate_reference_failure_skips_the_later_checks(
    gate, gate_grader, gate_repo, tmp_path, capsys
):
    _append(gate_repo / "test_hidden.py", "\n# BREAKREF\n")
    assert _run(gate, tmp_path / "out") == 1
    checks = _checks(capsys)
    assert checks["reference"] == "FAIL"
    assert set(list(checks.values())[3:]) == {"SKIP"}


def test_gate_accept_that_a_test_fails_is_refused(gate, gate_grader, gate_repo, tmp_path, capsys):
    _append(gate_repo / "test_hidden.py", CATCH_TEST)
    variants = gate_repo / "variants"
    variants.mkdir()
    (variants / "accept_bug.py").write_text(
        (FIXTURE / "solution.py").read_text(encoding="utf-8").replace("a + b", "a + b  # BUG"),
        encoding="utf-8",
    )
    manifest = gate.VariantsManifest(
        problem_version=1,
        accept=(
            gate.AcceptVariant(file="accept_bug.py", origin="freedom", freedom="adds a bug marker"),
        ),
    )
    gate.write_manifest(gate.find_problem(FIXTURE_ID), manifest)
    _run(gate, tmp_path / "out")
    checks = _checks(capsys)
    assert checks["accepts"] == "FAIL"
    assert checks["reference"] == "PASS"


def test_gate_reject_that_survives_is_refused(gate, gate_grader, gate_repo, tmp_path, capsys):
    variants = gate_repo / "variants"
    variants.mkdir()
    shutil.copyfile(FIXTURE / "solution.py", variants / "reject_none.py")
    clause = gate.clause_text((gate_repo / "problem.md").read_text(encoding="utf-8"), "C1")
    manifest = gate.VariantsManifest(
        problem_version=1,
        reject=(
            gate.RejectVariant(
                file="reject_none.py",
                mistake="no mistake",
                clause=clause,
                killed_by="test_sum_of_negative_ints",
                visible="hidden",
            ),
        ),
    )
    gate.write_manifest(gate.find_problem(FIXTURE_ID), manifest)
    _run(gate, tmp_path / "out")
    assert _checks(capsys)["rejects"] == "FAIL"


def test_gate_add_mode_fails_when_a_line_is_deleted(gate, gate_grader, gate_repo, tmp_path, capsys):
    _replace(
        gate_repo / "test_hidden.py",
        "def test_mixed_int_and_float():\n    assert add(1, 0.5) == 1.5, "
        '"an int plus a float should be their sum"\n',
        "",
    )
    _run(gate, tmp_path / "out", "--mode", "add")
    assert _checks(capsys)["change"] == "FAIL"


def test_gate_add_mode_fails_on_a_duplicate_name(gate, gate_grader, gate_repo, tmp_path, capsys):
    _append(
        gate_repo / "test_hidden.py",
        '\n\ndef test_none_raises_type_error():\n    assert add(1, 1) == 2, "again"\n',
    )
    _run(gate, tmp_path / "out", "--mode", "add")
    assert _checks(capsys)["change"] == "FAIL"


def test_gate_text_mode_fails_when_a_compared_value_changes(
    gate, gate_grader, gate_repo, tmp_path, capsys
):
    _replace(gate_repo / "test_hidden.py", "add(-4, -6) == -10", "add(-4, -6) == -11")
    _run(gate, tmp_path / "out", "--mode", "text")
    assert _checks(capsys)["change"] == "FAIL"


def test_gate_text_mode_passes_a_reworded_message(gate, gate_grader, gate_repo, tmp_path, capsys):
    _replace(gate_repo / "test_hidden.py", "check the sign", "look at the sign")
    assert _run(gate, tmp_path / "out", "--mode", "text") == 0
    checks = _checks(capsys)
    assert checks["change"] == "PASS"
    assert checks["reference"] == "PASS"


def test_gate_new_test_without_bump_fails_the_version_check(
    gate, gate_grader, gate_repo, tmp_path, capsys
):
    _append(gate_repo / "test_hidden.py", CATCH_TEST)
    _run(gate, tmp_path / "out", "--mode", "add")
    assert _checks(capsys)["version"] == "FAIL"


def test_gate_add_mode_fails_on_an_assert_without_a_message(
    gate, gate_grader, gate_repo, tmp_path, capsys
):
    _append(gate_repo / "test_hidden.py", "\n\ndef test_extra():\n    assert add(1, 1) == 2\n")
    _run(gate, tmp_path / "out", "--mode", "add")
    assert _checks(capsys)["lint"] == "FAIL"


def test_gate_add_mode_fails_on_a_new_parametrize_id_with_digits(
    gate, gate_grader, gate_repo, tmp_path, capsys
):
    _append(
        gate_repo / "test_hidden.py",
        # The decorator starts its own literal: an "n" from "\n" directly before "@name.name"
        # reads as an e-mail address to the privacy audit.
        "\n\n"
        '@pytest.mark.parametrize("a, b", [(1.5, 2)])\n'
        "def test_new_case(a, b):\n"
        '    assert add(a, b) == a + b, "adds the pair"\n',
    )
    _run(gate, tmp_path / "out", "--mode", "add")
    assert _checks(capsys)["names"] == "FAIL"


def test_gate_wrong_file_that_passes_is_refused(gate, gate_grader, gate_repo, tmp_path, capsys):
    _append(gate_repo / "test_hidden.py", CATCH_TEST)
    candidate = tmp_path / "candidates" / "wrong_none.py"
    candidate.parent.mkdir()
    shutil.copyfile(FIXTURE / "solution.py", candidate)
    assert gate.main(["bump-version", FIXTURE_ID, "--out", str(tmp_path / "out")]) == 0
    capsys.readouterr()
    assert _run(gate, tmp_path / "out", "--mode", "add", "--wrong", str(candidate)) == 1
    assert _checks(capsys)["wrong"] == "FAIL"


def test_gate_add_mode_with_bump_and_a_rejected_wrong_file_passes(
    gate, gate_grader, gate_repo, tmp_path, capsys
):
    _append(gate_repo / "test_hidden.py", CATCH_TEST)
    candidate = tmp_path / "candidates" / "wrong_bug.py"
    candidate.parent.mkdir()
    candidate.write_text("def add(a, b):\n    return a + b  # BUG\n", encoding="utf-8")
    out = tmp_path / "out"
    assert gate.main(["bump-version", FIXTURE_ID, "--out", str(out)]) == 0
    assert "version: 1 -> 2" in capsys.readouterr().out
    assert _run(gate, out, "--mode", "add", "--wrong", str(candidate)) == 0
    checks = _checks(capsys)
    assert checks == ALL_PASS
    rows = [json.loads(line) for line in (out / "gate.jsonl").read_text().splitlines()]
    assert rows[-1]["result"] == "PASS"


def test_gate_wrong_file_needs_add_mode(gate, gate_grader, gate_repo, tmp_path, capsys):
    candidate = tmp_path / "wrong.py"
    candidate.write_text("def add(a, b):\n    return a\n", encoding="utf-8")
    assert _run(gate, tmp_path / "out", "--wrong", str(candidate)) == 2
    assert "add only" in capsys.readouterr().err


def test_gate_unknown_base_is_a_usage_error(gate, gate_grader, tmp_path, capsys):
    assert _run(gate, tmp_path / "out", "--base", "no-such-revision") == 2
    assert "unknown revision" in capsys.readouterr().err


def test_gate_appends_one_row_per_run(gate, gate_grader, tmp_path, capsys):
    out = tmp_path / "out"
    _run(gate, out)
    _run(gate, out)
    rows = [json.loads(line) for line in (out / "gate.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["problem"] == FIXTURE_ID
    assert set(rows[0]["sha"]) == {"test_hidden.py", "test_public.py", "solution", "manifest"}


def test_expected_version_counts_a_new_test_and_respects_the_tag(gate, gate_repo):
    root = gate.audit.REPO
    problem = gate.find_problem(FIXTURE_ID)
    _git(root, "tag", "i32-start")  # the tag sits at version 1
    _replace(gate_repo / "meta.yaml", "version: 1\n", "version: 2\n")
    _git(root, "commit", "-q", "-am", "bump")
    assert gate.expected_version(problem, "HEAD") == 1  # from the tag, no new test
    _append(gate_repo / "test_hidden.py", CATCH_TEST)
    assert gate.expected_version(problem, "HEAD") == 2  # tag version 1, plus one new test
    _git(root, "tag", "-d", "i32-start")
    assert gate.expected_version(problem, "HEAD") == 3  # no tag: start is HEAD (version 2)


def test_expected_version_starts_from_the_newest_program_tag(gate, gate_repo):
    root = gate.audit.REPO
    problem = gate.find_problem(FIXTURE_ID)
    _git(root, "tag", "i32-start")  # the first program starts at version 1
    _append(gate_repo / "test_hidden.py", CATCH_TEST)
    _replace(gate_repo / "meta.yaml", "version: 1\n", "version: 2\n")
    _git(root, "commit", "-q", "-am", "first program adds a test")
    assert gate.expected_version(problem, "HEAD") == 2
    _git(root, "tag", "i34-start")  # the next program starts at version 2
    assert gate.start_revision(problem, "HEAD") == "i34-start"
    assert gate.expected_version(problem, "HEAD") == 2  # nothing new since the newer tag
    _append(gate_repo / "test_hidden.py", CATCH_TEST.replace("def test_", "def test_again_"))
    assert gate.expected_version(problem, "HEAD") == 3  # one more test, one more version
    _git(root, "tag", "-d", "i34-start")
    _git(root, "tag", "-d", "i32-start")


def test_bump_version_keeps_the_comment_and_updates_the_manifest(gate, gate_repo, tmp_path, capsys):
    _replace(gate_repo / "meta.yaml", "version: 1\n", "version: 1  # the tests set this\n")
    _append(gate_repo / "test_hidden.py", CATCH_TEST)
    problem = gate.find_problem(FIXTURE_ID)
    gate.write_manifest(problem, gate.VariantsManifest(problem_version=1))
    out = tmp_path / "out"
    assert gate.main(["bump-version", FIXTURE_ID, "--out", str(out)]) == 0
    assert "version: 1 -> 2" in capsys.readouterr().out
    text = (gate_repo / "meta.yaml").read_text(encoding="utf-8")
    assert "version: 2  # the tests set this\n" in text
    assert gate.read_manifest(problem).problem_version == 2
    assert gate.main(["bump-version", FIXTURE_ID, "--out", str(out)]) == 0
    assert "version: unchanged (2)" in capsys.readouterr().out


STRICT_RUFF = '[tool.ruff]\nline-length = 100\n\n[tool.ruff.lint]\nselect = ["E", "F", "B"]\n'
ZIP_NO_STRICT = (
    "def add(a, b):\n    pairs = list(zip([a], [b]))\n    return pairs[0][0] + pairs[0][1]\n"
)
UNDEFINED_NAME = "def add(a, b):\n    return a + b + missing\n"


def test_style_findings_in_a_variant_are_marked_with_noqa(gate, problem_dir, grader, tmp_path):
    (gate.audit.REPO / "pyproject.toml").write_text(STRICT_RUFF, encoding="utf-8")
    out = tmp_path / "out"
    source = _candidate(tmp_path, "accept_zip.py", ZIP_NO_STRICT)
    argv = ["add-accept", "--out", str(out), FIXTURE_REL, str(source)]
    assert gate.main([*argv, "--origin", "freedom", "--freedom", "pairs the inputs"]) == 0
    installed = (problem_dir / "variants" / "accept_zip.py").read_text(encoding="utf-8")
    assert "# noqa: B905" in installed
    assert "zip([a], [b])" in installed


def test_a_real_error_in_a_variant_still_refuses_it(gate, problem_dir, grader, tmp_path, capsys):
    (gate.audit.REPO / "pyproject.toml").write_text(STRICT_RUFF, encoding="utf-8")
    out = tmp_path / "out"
    source = _candidate(tmp_path, "accept_broken.py", UNDEFINED_NAME)
    argv = ["add-accept", "--out", str(out), FIXTURE_REL, str(source)]
    assert gate.main([*argv, "--origin", "freedom", "--freedom", "adds a name"]) == 1
    assert "F821" in capsys.readouterr().err
    assert not (problem_dir / "variants").exists()


# --- add-test, mark-swept, worklist, scorecard --------------------------------------------------


def _bytes(path: Path) -> bytes | None:
    return path.read_bytes() if path.is_file() else None


def _state(problem_dir: Path) -> dict[str, object]:
    """Every file add-test may touch, as bytes; the variant names, or None without a directory."""
    variants = problem_dir / "variants"
    return {
        "hidden": _bytes(problem_dir / "test_hidden.py"),
        "meta": _bytes(problem_dir / "meta.yaml"),
        "manifest": _bytes(variants / "manifest.yaml"),
        "variants": sorted(p.name for p in variants.iterdir()) if variants.is_dir() else None,
    }


def _blind_manifest(gate: ModuleType, problem_dir: Path, reject=()) -> None:
    """A manifest with one blind accept (the correct fixture solution), as the tool writes it."""
    variants = problem_dir / "variants"
    variants.mkdir(exist_ok=True)
    shutil.copyfile(FIXTURE / "solution.py", variants / "accept_blind.py")
    manifest = gate.VariantsManifest(
        problem_version=1,
        accept=(gate.AcceptVariant(file="accept_blind.py", origin="blind"),),
        reject=tuple(reject),
    )
    gate.write_manifest(gate.find_problem(FIXTURE_ID), manifest)


def _add_test_argv(out: Path, testfile: Path, wrong: Path, *extra: str) -> list[str]:
    base = [
        "add-test",
        FIXTURE_ID,
        str(testfile),
        "--out",
        str(out),
        "--wrong",
        str(wrong),
        "--mistake",
        "drops the bool check",
        "--clause",
        "C1",
    ]
    return [*base, *extra]


def _each_catch(problem, source, _settings, **_kw) -> GradeReport:
    """Like _fake_gate_grade, but every hidden test whose text has CATCH fails on a BUG source."""
    hidden_text = _read(problem.path / "test_hidden.py") or ""
    tests = [
        GradedTest(name=name, file="public", outcome="pass")
        for name in _test_names(_read(problem.path / "test_public.py"))
    ]
    for node in ast.parse(hidden_text).body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            text = ast.get_source_segment(hidden_text, node) or ""
            if "BUG" in source and "CATCH" in text:
                outcome = GradedTest(
                    name=node.name,
                    file="hidden",
                    outcome="fail",
                    message="the sum is wrong for bools",
                )
            else:
                outcome = GradedTest(name=node.name, file="hidden", outcome="pass")
            tests.append(outcome)
    failed = any(test.outcome != "pass" for test in tests)
    return _report(RunStatus.FAILED if failed else RunStatus.PASSED, tests)


@pytest.fixture
def ruff_repo(gate: ModuleType, gate_repo: Path) -> Path:
    """The scratch repo with the repository's own ruff configuration, as a real checkout has."""
    shutil.copyfile(ROOT / "pyproject.toml", gate.audit.REPO / "pyproject.toml")
    return gate_repo


BUG_WRONG = "def add(a, b):\n    return a + b  # BUG\n"


@pytest.mark.usefixtures("ruff_repo")
def test_add_test_appends_bumps_and_records_the_reject(
    gate, gate_repo, gate_grader, tmp_path, capsys
):
    _blind_manifest(gate, gate_repo)
    testfile = _candidate(tmp_path, "new_test.py", CATCH_TEST)
    wrong = _candidate(tmp_path, "reject_bool.py", BUG_WRONG)
    out = tmp_path / "out"
    assert gate.main(_add_test_argv(out, testfile, wrong)) == 0
    text = capsys.readouterr().out
    assert "GATE: PASS" in text
    assert "ADD-TEST: added test_catches_bug and reject_bool.py" in text
    assert "def test_catches_bug" in (gate_repo / "test_hidden.py").read_text(encoding="utf-8")
    assert "version: 2" in (gate_repo / "meta.yaml").read_text(encoding="utf-8")
    manifest = gate.read_manifest(gate.find_problem(FIXTURE_ID))
    assert manifest.problem_version == 2
    entry = manifest.reject[0]
    assert (entry.file, entry.killed_by, entry.visible) == (
        "reject_bool.py",
        "test_catches_bug",
        "hidden",
    )
    assert entry.clause == gate.clause_text(
        (gate_repo / "problem.md").read_text(encoding="utf-8"), "C1"
    )
    assert (gate_repo / "variants" / "reject_bool.py").is_file()


ADD_TEST_REFUSALS = [
    # (case, test text, blind accept present, extra arguments, exit code, text in the output)
    ("no blind accept", CATCH_TEST, False, [], 1, "no blind accept"),
    (
        "two tests",
        CATCH_TEST + "\n\ndef test_second():\n    assert add(1, 1) == 2, 'adds'\n",
        True,
        [],
        1,
        "exactly one test_ function, found 2",
    ),
    (
        "a taken name",
        "def test_sum_of_negative_ints():\n    assert add(1, 1) == 2, 'adds'\n",
        True,
        [],
        1,
        "repeats a name",
    ),
    (
        "41 lines",
        "def test_long():\n" + "    x = 1\n" * 40 + "    assert add(1, 1) == 2, 'adds'\n",
        True,
        [],
        1,
        "at most 40 lines",
    ),
    (
        "pytest.raises",
        "def test_raises():\n    with pytest.raises(TypeError):\n        add(None, 1)\n",
        True,
        [],
        1,
        "does not use pytest.raises",
    ),
    (
        "assert without a message",
        "def test_bare():\n    assert add(1, 1) == 2\n",
        True,
        [],
        1,
        "lint findings",
    ),
    ("does not parse", "def test_bad(:\n", True, [], 1, "does not parse"),
    ("bad label", CATCH_TEST, True, ["--clause", "C9"], 1, "no clause C9"),
]


@pytest.mark.parametrize("row", ADD_TEST_REFUSALS, ids=[row[0] for row in ADD_TEST_REFUSALS])
@pytest.mark.usefixtures("ruff_repo", "gate_grader")
def test_add_test_refusals_change_no_file(gate, ruff_repo, tmp_path, capsys, row):
    case, body, blind, extra, code, text = row
    if blind:
        _blind_manifest(gate, ruff_repo)
    testfile = _candidate(tmp_path, "new_test.py", body)
    wrong = _candidate(tmp_path, "reject_bug.py", BUG_WRONG)
    before = _state(ruff_repo)
    assert gate.main(_add_test_argv(tmp_path / "out", testfile, wrong, *extra)) == code, case
    captured = capsys.readouterr()
    assert text in captured.err + captured.out, case
    assert _state(ruff_repo) == before, case


@pytest.mark.usefixtures("ruff_repo")
def test_add_test_rejects_a_mistake_with_markup(gate, gate_repo, gate_grader, tmp_path, capsys):
    _blind_manifest(gate, gate_repo)
    testfile = _candidate(tmp_path, "new_test.py", CATCH_TEST)
    wrong = _candidate(tmp_path, "reject_bug.py", BUG_WRONG)
    before = _state(gate_repo)
    argv = _add_test_argv(tmp_path / "out", testfile, wrong)
    argv[argv.index("drops the bool check")] = "uses `x`"
    assert gate.main(argv) == 1
    assert "use plain words" in capsys.readouterr().err
    assert _state(gate_repo) == before


@pytest.mark.usefixtures("ruff_repo")
def test_add_test_rolls_back_when_the_wrong_file_passes(
    gate, gate_repo, gate_grader, tmp_path, capsys
):
    _blind_manifest(gate, gate_repo)
    plain = "def test_plain():\n    assert add(True, 1) == 2, 'the sum is wrong for bools'\n"
    testfile = _candidate(tmp_path, "new_test.py", plain)
    wrong = _candidate(tmp_path, "reject_bug.py", BUG_WRONG)
    before = _state(gate_repo)
    assert gate.main(_add_test_argv(tmp_path / "out", testfile, wrong)) == 1
    captured = capsys.readouterr()
    assert "wrong      FAIL" in captured.out
    assert "ADD-TEST: FAILED, nothing changed" in captured.out
    assert _state(gate_repo) == before


@pytest.mark.usefixtures("ruff_repo")
def test_add_test_rolls_back_when_the_reference_fails(
    gate, gate_repo, gate_grader, tmp_path, capsys
):
    _blind_manifest(gate, gate_repo)
    broken = "def test_broken():  # BREAKREF\n    assert add(1, 1) == 2, 'adds'\n"
    testfile = _candidate(tmp_path, "new_test.py", broken)
    wrong = _candidate(tmp_path, "reject_bug.py", BUG_WRONG)
    before = _state(gate_repo)
    assert gate.main(_add_test_argv(tmp_path / "out", testfile, wrong)) == 1
    captured = capsys.readouterr()
    assert "reference  FAIL" in captured.out
    assert "ADD-TEST: FAILED, nothing changed" in captured.out
    assert _state(gate_repo) == before


@pytest.mark.usefixtures("ruff_repo", "gate_grader")
def test_add_test_rolls_back_when_ruff_rejects_the_suite(
    gate, ruff_repo, tmp_path, monkeypatch, capsys
):
    _blind_manifest(gate, ruff_repo)
    real_ruff = gate._ruff

    def strict_check(step, target):
        if step == ("check",) and target.name == "test_hidden.py":
            return False, "F821 undefined name"
        return real_ruff(step, target)

    monkeypatch.setattr(gate, "_ruff", strict_check)
    testfile = _candidate(tmp_path, "new_test.py", CATCH_TEST)
    wrong = _candidate(tmp_path, "reject_bug.py", BUG_WRONG)
    before = _state(ruff_repo)
    assert gate.main(_add_test_argv(tmp_path / "out", testfile, wrong)) == 1
    captured = capsys.readouterr()
    assert "ruff check failed on test_hidden.py" in captured.err
    assert "ADD-TEST: FAILED, nothing changed" in captured.out
    assert _state(ruff_repo) == before


@pytest.mark.usefixtures("ruff_repo", "gate_grader")
def test_add_test_twice_bumps_the_version_once(gate, ruff_repo, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gate.audit, "grade_source", _each_catch)
    _blind_manifest(gate, ruff_repo)
    out = tmp_path / "out"
    first = _candidate(tmp_path, "first.py", CATCH_TEST)
    second_text = CATCH_TEST.replace("test_catches_bug", "test_catches_bug_again")
    second = _candidate(tmp_path, "second.py", second_text)
    wrong_one = _candidate(tmp_path, "reject_one.py", BUG_WRONG)
    wrong_two = _candidate(tmp_path, "reject_two.py", BUG_WRONG)
    assert gate.main(_add_test_argv(out, first, wrong_one)) == 0
    assert gate.main(_add_test_argv(out, second, wrong_two)) == 0
    assert "version: 2" in (ruff_repo / "meta.yaml").read_text(encoding="utf-8")
    manifest = gate.read_manifest(gate.find_problem(FIXTURE_ID))
    assert manifest.problem_version == 2
    assert [entry.killed_by for entry in manifest.reject] == [
        "test_catches_bug",
        "test_catches_bug_again",
    ]


def _weeks(count: int) -> list:
    start = date(2026, 11, 2)
    return [{"week": n, "starts": start + timedelta(weeks=n - 1)} for n in range(1, count + 1)]


def test_week_order_before_inside_and_after_the_path(gate):
    weeks = [
        gate.PathWeek(week=row["week"], starts=row["starts"], theme="t", topics=("t/one",))
        for row in _weeks(13)
    ]
    assert gate.week_order(weeks, date(2026, 10, 1)) == [*range(2, 14), 1]
    inside = weeks[3].starts + timedelta(days=2)
    assert gate.week_order(weeks, inside) == [*range(5, 14), 4, 3, 2, 1]
    assert gate.week_order(weeks, date(2027, 6, 1)) == list(range(13, 0, -1))
    assert gate.week_order([], date(2026, 11, 2)) == []


WORKLIST_WEEKS = (
    {"week": 1, "starts": date(2026, 11, 2), "topics": ("t/two", "t/one")},
    {"week": 2, "starts": date(2026, 11, 9), "topics": ("t/three",)},
)
CLEAN_HIDDEN = "def test_ok():\n    assert 1 == 1, 'a wrong sum'\n"
DIRTY_HIDDEN = "def test_bad():\n    assert 1 == 2\n"
BLIND_SWEPT = (
    "problem_version: 1\naccept:\n  - file: accept_blind.py\n    origin: blind\n"
    "sweeps:\n  - mutants\n"
)


@pytest.fixture
def worklist_repo(gate, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Five problems: a and c are unswept, b has a blind accept and mutants, e is on week 2,
    d is off the path. c has a blocked note and a dirty hidden suite."""
    root = tmp_path / "repo"
    rows = [
        ("t/one/p01-a", None, CLEAN_HIDDEN),
        ("t/one/p02-b", BLIND_SWEPT, CLEAN_HIDDEN),
        ("t/two/p01-c", None, DIRTY_HIDDEN),
        ("t/three/p01-e", None, CLEAN_HIDDEN),
        ("t/off/p01-d", None, CLEAN_HIDDEN),
    ]
    problems = []
    for problem_id, manifest, hidden in rows:
        path = root / "content" / problem_id
        path.mkdir(parents=True)
        (path / "test_hidden.py").write_text(hidden, encoding="utf-8")
        if manifest is not None:
            (path / "variants").mkdir()
            (path / "variants" / "manifest.yaml").write_text(manifest, encoding="utf-8")
        problems.append(gate.audit.Problem(problem_id, path, "python", "add", 30, ()))
    gate.append_note(root / "var" / "audit-i32" / "run", "t/two/p01-c", "blocked", "stuck")
    core = gate.CorePath(
        id="core",
        title="t",
        start=date(2026, 11, 2),
        end=date(2027, 1, 31),
        sessions_per_day=3,
        session_minutes=45,
        days_per_week=6,
        weeks=tuple(
            gate.PathWeek(
                week=row["week"],
                starts=row["starts"],
                theme="t",
                topics=row["topics"],
            )
            for row in WORKLIST_WEEKS
        ),
    )
    monkeypatch.setattr(gate.audit, "REPO", root)
    monkeypatch.setattr(gate.audit, "discover", lambda repo, only: problems)
    monkeypatch.setattr(gate, "_core_path", lambda: core)
    return root


def _worklist(gate, tmp_path: Path, kind: str, capsys) -> str:
    argv = ["worklist", "--out", str(tmp_path / "out"), "--kind", kind, "--today", "2026-11-02"]
    assert gate.main(argv) == 0
    return capsys.readouterr().out


def test_worklist_blind_lists_the_week_order_and_the_off_path_problem(
    gate, worklist_repo, tmp_path, capsys
):
    assert _worklist(gate, tmp_path, "blind", capsys) == (
        "WORKLIST blind  today=2026-11-02  current week=1\n"
        "week 2\n"
        "  t/three/p01-e\n"
        "week 1\n"
        "  t/two/p01-c  (blocked 1)\n"
        "  t/one/p01-a\n"
        "not on the path\n"
        "  t/off/p01-d\n"
        "total: 4\n"
    )


def test_worklist_reject_and_mutants_differ_by_their_sweep(gate, worklist_repo, tmp_path, capsys):
    reject = _worklist(gate, tmp_path, "reject", capsys)
    assert "  t/one/p02-b\n" in reject and reject.endswith("total: 5\n")
    mutants = _worklist(gate, tmp_path, "mutants", capsys)
    assert "t/one/p02-b" not in mutants and mutants.endswith("total: 4\n")


def test_worklist_hygiene_lists_only_the_dirty_suite(gate, worklist_repo, tmp_path, capsys):
    assert _worklist(gate, tmp_path, "hygiene", capsys) == (
        "WORKLIST hygiene  today=2026-11-02  current week=1\n"
        "week 1\n"
        "  t/two/p01-c  (blocked 1)\n"
        "total: 1\n"
    )


def test_worklist_with_nothing_left_prints_total_zero(
    gate, worklist_repo, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(gate.audit, "discover", lambda repo, only: [])
    assert _worklist(gate, tmp_path, "blind", capsys) == (
        "WORKLIST blind  today=2026-11-02  current week=1\ntotal: 0\n"
    )


def test_worklist_refuses_a_bad_date(gate, worklist_repo, tmp_path):
    argv = ["worklist", "--out", str(tmp_path / "out"), "--kind", "blind", "--today", "soon"]
    assert gate.main(argv) == 2


def test_mark_swept_marks_on_pass_and_creates_the_manifest(
    gate, gate_repo, gate_grader, tmp_path, capsys
):
    out = tmp_path / "out"
    argv = ["mark-swept", FIXTURE_ID, "--out", str(out), "--kind", "reject", "--mode", "none"]
    assert gate.main(argv) == 0
    assert "marked reject" in capsys.readouterr().out
    manifest = gate.read_manifest(gate.find_problem(FIXTURE_ID))
    assert manifest.sweeps == ("reject",)
    assert manifest.problem_version == 1


def test_mark_swept_does_not_duplicate_a_kind(gate, gate_repo, gate_grader, tmp_path, capsys):
    out = tmp_path / "out"
    base = ["mark-swept", FIXTURE_ID, "--out", str(out), "--mode", "none"]
    assert gate.main([*base, "--kind", "reject"]) == 0
    assert gate.main([*base, "--kind", "mutants"]) == 0
    assert gate.main([*base, "--kind", "reject"]) == 0
    assert "already marked reject" in capsys.readouterr().out
    assert gate.read_manifest(gate.find_problem(FIXTURE_ID)).sweeps == ("reject", "mutants")


def test_mark_swept_writes_nothing_on_a_failing_gate(
    gate, gate_repo, gate_grader, tmp_path, capsys
):
    _append(gate_repo / "problem.md", "\nan edit the gate must refuse\n")
    argv = ["mark-swept", FIXTURE_ID, "--out", str(tmp_path / "out"), "--kind", "reject"]
    argv += ["--mode", "none"]
    assert gate.main(argv) == 1
    assert "GATE: FAIL" in capsys.readouterr().out
    assert not (gate_repo / "variants" / "manifest.yaml").exists()


def test_bullet_coverage_counts_the_clauses_a_reject_or_uncovered_entry_names(gate):
    reject = gate.RejectVariant(
        file="reject_x.py",
        mistake="m",
        clause="Second bullet.",
        killed_by="test_x",
        visible="hidden",
    )
    assert gate.bullet_coverage(
        PROBLEM_MD, gate.VariantsManifest(problem_version=1, reject=(reject,))
    ) == (1, 2)
    uncovered = gate.UncoveredClause(clause="A bullet that wraps onto a second line.", reason="r")
    manifest = gate.VariantsManifest(problem_version=1, reject=(reject,), uncovered=(uncovered,))
    assert gate.bullet_coverage(PROBLEM_MD, manifest) == (2, 2)
    text_only = gate.RejectVariant(
        file="reject_y.py",
        mistake="m",
        clause="Text before the list.",
        killed_by="test_y",
        visible="hidden",
    )
    assert gate.bullet_coverage(
        PROBLEM_MD, gate.VariantsManifest(problem_version=1, reject=(text_only,))
    ) == (0, 2)
    assert gate.bullet_coverage(PROBLEM_MD, None) == (0, 2)


def test_scorecard_prints_one_row_per_problem_and_the_summary(
    gate, gate_repo, gate_grader, tmp_path, capsys
):
    gate.append_note(gate.audit.REPO / "var" / "audit-i32" / "run", FIXTURE_ID, "blocked", "x")
    assert gate.main(["scorecard", "--out", str(tmp_path / "out")]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("| Problem | Version | Lint | Blind | Accepts | Rejects |")
    rows = [line for line in lines if line.startswith(f"| {FIXTURE_ID} |")]
    assert len(rows) == 1
    assert rows[0].startswith(f"| {FIXTURE_ID} | 1 | ")
    assert rows[0].endswith("| - | blocked 1 |")
    assert "problems: 1" in lines and "rejects: 0" in lines
