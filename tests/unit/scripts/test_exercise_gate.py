"""Tests for scripts/exercise_gate.py: clause labels, probe verdicts, notes, lint and usage errors.

The gate reuses the audit tool's discovery and grader. Both are replaced here with fakes, so no
test grades a real submission; the runbook's Accept commands do that.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
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
