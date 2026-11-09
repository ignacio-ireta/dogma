"""Tests for scripts/exercise_audit.py: the mutation operators, sampling and message classes.

The tool's numbers are only worth trusting if its mutants are real, so each operator is pinned on
small sources. Nothing here grades a submission; the integration step is the tool's own baseline.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from dogma.grading.models import GradeReport, RunStatus
from dogma.grading.models import TestResult as GradedTest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "exercise_audit.py"


@pytest.fixture(scope="module")
def audit() -> ModuleType:
    spec = importlib.util.spec_from_file_location("exercise_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ops(mutants: list) -> set[str]:
    return {mutant.op for mutant in mutants}


def _call(source: str, name: str, *args: object) -> object:
    namespace: dict[str, object] = {}
    exec(compile(source, "mutant", "exec"), namespace)  # noqa: S102 - our own literal sources
    function = namespace[name]
    assert callable(function)
    return function(*args)


def test_every_python_mutant_compiles_and_differs_from_the_source(audit: ModuleType) -> None:
    source = (
        "def f(x, n=2):\n"
        "    if x < 0:\n"
        "        raise ValueError('negative')\n"
        "    total = 0\n"
        "    for i in range(n):\n"
        "        total += x * i\n"
        "    return max(total, 1) and not x\n"
    )
    mutants = audit.python_mutants(source)
    assert len(mutants) >= 8
    assert len({mutant.source for mutant in mutants}) == len(mutants)
    for mutant in mutants:
        compile(mutant.source, "mutant", "exec")
        assert mutant.change


@pytest.mark.parametrize(
    ("source", "op"),
    [
        ("def f(a, b):\n    return a < b\n", "compare"),
        ("def f(a, b):\n    return a + b\n", "arithmetic"),
        ("def f(a, b):\n    return a and b\n", "boolean"),
        ("def f(a):\n    return -a\n", "drop_negation"),
        ("def f():\n    return 3\n", "constant"),
        ("def f():\n    return True\n", "bool"),
        ("def f(a):\n    for x in a:\n        break\n", "flow"),
        ("def f(a):\n    return max(a)\n", "call"),
        ("def f(a):\n    return abs(a)\n", "drop_call"),
        ("def f(a):\n    return a.copy()\n", "drop_call"),
        ("def f(a):\n    return np.array(a)\n", "array_copy"),
        ("def f(a):\n    return a.sum(axis=0)\n", "axis"),
        ("def f(a):\n    return a.sum(dim=-1)\n", "axis"),
        ("def f(a):\n    return a.T\n", "transpose"),
        ("def f(a):\n    return a.astype(np.float64)\n", "dtype"),
        ("def f(a):\n    if a:\n        raise ValueError('x')\n    return a\n", "guard"),
        ("def f(a):\n    a.append(1)\n", "drop_statement"),
    ],
)
def test_each_python_operator_fires(audit: ModuleType, source: str, op: str) -> None:
    assert op in _ops(audit.python_mutants(source))


def test_deleting_a_guard_removes_the_error_and_keeps_the_rest(audit: ModuleType) -> None:
    source = "def f(a):\n    if a < 0:\n        raise ValueError('x')\n    return a * 2\n"
    guard = next(m for m in audit.python_mutants(source) if m.op == "guard")
    assert _call(guard.source, "f", -1) == -2


def test_compare_mutants_change_only_the_boundary(audit: ModuleType) -> None:
    source = "def f(a):\n    return a < 3\n"
    results = {_call(m.source, "f", 3) for m in audit.python_mutants(source) if m.op == "compare"}
    assert True in results


def test_messages_hints_and_docstrings_are_not_mutated(audit: ModuleType) -> None:
    source = (
        'def f(a: "int" = 1) -> "int":\n'
        '    """Doc 2."""\n'
        "    if a:\n"
        "        raise ValueError(f'bad {3 + 4}')\n"
        "    return a\n"
    )
    assert not any("3 - 4" in m.source or "3 + 5" in m.source for m in audit.python_mutants(source))


def test_a_source_with_nothing_to_mutate_gives_no_mutants(audit: ModuleType) -> None:
    assert audit.python_mutants("def f(a):\n    return a\n") == []


@pytest.mark.parametrize(
    ("before", "after", "op"),
    [
        ("SELECT a FROM t LEFT JOIN u ON t.x = u.x", "t JOIN u", "join"),
        ("SELECT a FROM t JOIN u ON t.x = u.x", "t LEFT JOIN u", "join"),
        ("SELECT a FROM t INNER JOIN u ON t.x = u.x", "t LEFT JOIN u", "join"),
        ("SELECT DISTINCT a FROM t", "SELECT a FROM t", "distinct"),
        ("SELECT ROW_NUMBER() OVER (ORDER BY a) FROM t", "RANK() OVER", "window"),
        ("SELECT LAG(a) OVER (ORDER BY a) FROM t", "LEAD(a)", "window"),
        ("SELECT a FROM t ORDER BY a DESC", "ORDER BY a ASC", "order"),
        ("SELECT a FROM t WHERE a <= 3", "a < 3", "compare"),
        ("SELECT a FROM t WHERE a = 3", "a <> 3", "compare"),
        ("SELECT a FROM t WHERE a > 3", "a > 4", "constant"),
        ("SELECT a FROM t WHERE a IS NULL", "a IS NOT NULL", "null"),
        ("SELECT COALESCE(a, 0) FROM t", "SELECT a FROM t", "null"),
        ("SELECT SUM(a) OVER (PARTITION BY b ORDER BY c) FROM t", "OVER (ORDER BY c)", "window"),
        ("SELECT a FROM t UNION ALL SELECT a FROM u", "t UNION SELECT", "set"),
    ],
)
def test_sql_rules(audit: ModuleType, before: str, after: str, op: str) -> None:
    mutants = [m for m in audit.sql_mutants(before) if m.op == op]
    assert any(after in " ".join(m.source.split()) for m in mutants), [m.source for m in mutants]


def test_sql_rules_ignore_strings_and_comments(audit: ModuleType) -> None:
    source = "SELECT 'a <= 3 AND b' -- MIN JOIN 4\nFROM t"
    assert audit.sql_mutants(source) == []


def test_sql_does_not_double_a_join_kind(audit: ModuleType) -> None:
    mutants = audit.sql_mutants("SELECT a FROM t CROSS JOIN u")
    assert not any("LEFT JOIN" in m.source for m in mutants)


def test_selection_is_repeatable_capped_and_keeps_every_guard(audit: ModuleType) -> None:
    mutants = [audit.Mutant("guard", i, f"g{i}", "") for i in range(5)]
    mutants += [audit.Mutant("compare", i, f"c{i}", "") for i in range(50)]
    mutants += [audit.Mutant("constant", i, f"k{i}", "") for i in range(50)]
    first = audit.select_mutants(mutants, "cml/x/p01", 20)
    assert len(first) == 20
    assert sum(m.op == "guard" for m in first) == 5
    assert {m.op for m in first} == {"guard", "compare", "constant"}
    assert first == audit.select_mutants(mutants, "cml/x/p01", 20)
    assert first != audit.select_mutants(mutants, "cml/y/p01", 20)


def test_a_mutant_key_depends_only_on_its_source(audit: ModuleType) -> None:
    assert audit.Mutant("a", 1, "x", "").key == audit.Mutant("b", 2, "x", "y").key


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("the shape must be (n, d)", "named"),
        ("(the test failed; its name describes what was checked)", "no_detail"),
        ("", "no_detail"),
        ("TypeError: unsupported operand", "bare_exception"),
        ("numpy.linalg.LinAlgError: Singular matrix", "bare_exception"),
        ("assert 3 == 4", "raw_assert"),
    ],
)
def test_message_classes(audit: ModuleType, text: str, kind: str) -> None:
    assert audit.classify_message(text) == kind


def _problem(audit: ModuleType, tmp_path: Path, source: str, language: str = "python") -> object:
    suffix = ".sql" if language == "sql" else ".py"
    (tmp_path / f"solution{suffix}").write_text(source, encoding="utf-8")
    return audit.Problem("t/x/p01", tmp_path, language, "f", 30, ())


def test_the_precision_wrapper_scales_floats_and_leaves_the_rest(
    audit: ModuleType, tmp_path: Path
) -> None:
    source = "def f():\n    return 2.0, 3, 'a', [1.0], {'k': 4.0}, None, True\n"
    scaled = audit.scaled_solution(_problem(audit, tmp_path, source), 0.5)
    assert _call(scaled, "f") == (3.0, 3, "a", [1.5], {"k": 6.0}, None, True)


def test_the_precision_wrapper_skips_sql_and_class_entries(
    audit: ModuleType, tmp_path: Path
) -> None:
    assert audit.scaled_solution(_problem(audit, tmp_path, "SELECT 1", "sql"), 0.1) is None
    assert audit.scaled_solution(_problem(audit, tmp_path, "class f:\n    pass\n"), 0.1) is None


def test_bundles_never_contain_the_solution_hidden_tests_or_hints(
    audit: ModuleType, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "problem"
    problem_dir.mkdir()
    for name in ("problem.md", "starter.py", "test_public.py", "solution.py"):
        (problem_dir / name).write_text("x", encoding="utf-8")
    (problem_dir / "test_hidden.py").write_text("x", encoding="utf-8")
    (problem_dir / "hints.md").write_text("x", encoding="utf-8")
    problem = audit.Problem("t/x/p01", problem_dir, "python", "f", 30, ())

    audit.cmd_bundle([problem], tmp_path / "out")

    bundle = tmp_path / "out" / "blind" / "t" / "x" / "p01"
    assert sorted(path.name for path in bundle.iterdir()) == [
        "problem.md",
        "starter.py",
        "test_public.py",
    ]


def test_triage_ranks_contract_mutants_before_the_rest(audit: ModuleType) -> None:
    rows = [
        {"op": "constant", "key": "a"},
        {"op": "compare", "key": "b"},
        {"op": "guard", "key": "c"},
    ]
    assert [row["key"] for row in sorted(rows, key=audit.triage_rank)] == ["c", "b", "a"]


def test_extra_tests_are_appended_to_a_copy_and_never_to_the_problem(
    audit: ModuleType, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "problem-src"
    problem_dir.mkdir()
    (problem_dir / "test_hidden.py").write_text("def test_a():\n    pass\n", encoding="utf-8")
    problem = audit.Problem("t/x/p01", problem_dir, "python", "f", 30, ())
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    extended = audit._with_extra_tests(problem, "def test_b():\n    pass\n", scratch)

    assert "test_b" in (extended.path / "test_hidden.py").read_text(encoding="utf-8")
    assert "test_b" not in (problem_dir / "test_hidden.py").read_text(encoding="utf-8")


def test_the_precision_label_is_the_tightest_error_caught(audit: ModuleType) -> None:
    rows = [
        {"epsilon": 1e-3, "verdict": "killed_hidden"},
        {"epsilon": 1e-6, "verdict": "killed_hidden"},
        {"epsilon": 1e-8, "verdict": "survived"},
    ]
    assert audit._precision_label(rows) == "1e-06"
    assert audit._precision_label([{"epsilon": 1e-3, "verdict": "survived"}]) == "n/a"
    assert audit._precision_label([]) == "n/a"


# --- salt and realism -----------------------------------------------------------------------------


def _report(status: RunStatus, tests: tuple[GradedTest, ...] = ()) -> GradeReport:
    return GradeReport(
        attempt_id="a",
        problem_id="t/x/p01-demo",
        problem_version=1,
        runtime="py-numpy",
        status=status,
        tests=tests,
    )


def _failure(file: str) -> GradeReport:
    failed = GradedTest(
        name=f"test_{file}_catches_bug", file=file, outcome="fail", message="shape mismatch"
    )
    return _report(RunStatus.FAILED, (failed,))


@pytest.fixture
def fake_grader(audit: ModuleType, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, bool]]:
    """A stand-in grader: a ``BUG`` source fails any suite whose file has a ``CATCH`` marker.

    Returns the list that records ``(problem id, include_hidden)`` for every call.
    """
    calls: list[tuple[str, bool]] = []

    def grade(problem: object, source: str, settings: object, *, include_hidden: bool = True):
        path = problem.path  # type: ignore[attr-defined]
        calls.append((problem.id, include_hidden))  # type: ignore[attr-defined]
        public = (path / "test_public.py").read_text(encoding="utf-8")
        hidden_file = path / "test_hidden.py"
        hidden = hidden_file.read_text(encoding="utf-8") if hidden_file.is_file() else ""
        if "BUG" in source and "CATCH" in public:
            return _failure("public")
        if include_hidden and "BUG" in source and "CATCH" in hidden:
            return _failure("hidden")
        return _report(RunStatus.PASSED)

    monkeypatch.setattr(audit, "grade_source", grade)
    return calls


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def demo(audit: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A git repo with one problem at HEAD whose tests do not catch the BUG marker."""
    root = tmp_path / "repo"
    problem_dir = root / "content" / "t" / "x" / "problems" / "p01-demo"
    problem_dir.mkdir(parents=True)
    (problem_dir / "solution.py").write_text("def f(x):\n    return x\n", encoding="utf-8")
    (problem_dir / "test_public.py").write_text("def test_example():\n    pass\n")
    (problem_dir / "test_hidden.py").write_text("def test_base_hidden():\n    pass\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    monkeypatch.setattr(audit, "REPO", root)
    problem = audit.Problem("t/x/p01-demo", problem_dir, "python", "f", 30, ())
    monkeypatch.setattr(audit, "discover", lambda repo, only: [problem])  # no meta.yaml needed
    return problem


def _add_realism_test(problem: object) -> None:
    hidden = problem.path / "test_hidden.py"  # type: ignore[attr-defined]
    hidden.write_text(hidden.read_text(encoding="utf-8") + "# CATCH realistic data\n")


def _wrong(tmp_path: Path, name: str = "wrong.py", body: str = "BUG") -> Path:
    path = tmp_path / name
    path.write_text(f"# {body}\ndef f(x):\n    return x + 1\n", encoding="utf-8")
    return path


def _rows(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "realism.jsonl").read_text().splitlines()]


@pytest.mark.parametrize(
    ("base", "new", "expected"),
    [
        ("survived", "killed_hidden", "killed"),
        ("survived", "killed_public", "killed"),
        ("survived", "survived", "survives"),
        ("killed_hidden", "killed_hidden", "already_caught"),
        ("killed_public", "survived", "already_caught"),
        ("timeout", "survived", "error"),
        ("survived", "error", "error"),
    ],
)
def test_realism_verdicts(audit: ModuleType, base: str, new: str, expected: str) -> None:
    assert audit.realism_verdict(base, new) == expected


def test_a_wrong_solution_the_base_missed_is_killed(
    audit: ModuleType, demo, tmp_path: Path, fake_grader
) -> None:
    _add_realism_test(demo)
    out = tmp_path / "out"
    wrong = _wrong(tmp_path)
    assert audit.main(["realism", "--out", str(out), demo.id, str(wrong)]) == 0
    verdicts = {row["kind"]: row["verdict"] for row in _rows(out)}
    assert verdicts == {"reference": "survived", "wrong": "killed"}


def test_a_wrong_solution_the_base_already_catches_fails_the_check(
    audit: ModuleType, demo, tmp_path: Path, fake_grader
) -> None:
    _add_realism_test(demo)
    _git(audit.REPO, "add", "-A")
    _git(audit.REPO, "commit", "-q", "-m", "base already catches")
    out = tmp_path / "out"
    assert audit.main(["realism", "--out", str(out), demo.id, str(_wrong(tmp_path))]) == 1
    (wrong_row,) = [row for row in _rows(out) if row["kind"] == "wrong"]
    assert wrong_row["verdict"] == "already_caught"


def test_a_wrong_solution_that_passes_the_new_tests_survives(
    audit: ModuleType, demo, tmp_path: Path, fake_grader, capsys
) -> None:
    _add_realism_test(demo)
    out = tmp_path / "out"
    survivor = _wrong(tmp_path, body="ok")
    assert audit.main(["realism", "--out", str(out), demo.id, str(survivor)]) == 1
    assert "wrong.py: survives" in capsys.readouterr().out


def test_every_wrong_solution_must_be_killed(
    audit: ModuleType, demo, tmp_path: Path, fake_grader
) -> None:
    _add_realism_test(demo)
    out = tmp_path / "out"
    killed = _wrong(tmp_path, "a.py")
    survivor = _wrong(tmp_path, "b.py", body="ok")
    assert audit.main(["realism", "--out", str(out), demo.id, str(killed), str(survivor)]) == 1


def test_a_failing_reference_fails_the_check(
    audit: ModuleType, demo, tmp_path: Path, fake_grader, capsys
) -> None:
    _add_realism_test(demo)
    demo.solution.write_text("# BUG\ndef f(x):\n    return x\n", encoding="utf-8")
    out = tmp_path / "out"
    assert audit.main(["realism", "--out", str(out), demo.id, str(_wrong(tmp_path))]) == 1
    assert "reference solution.py" in capsys.readouterr().out


def test_a_blind_solution_must_pass_the_new_tests(
    audit: ModuleType, demo, tmp_path: Path, fake_grader
) -> None:
    _add_realism_test(demo)
    out = tmp_path / "out"
    wrong = str(_wrong(tmp_path))
    blind_ok = tmp_path / "blind_ok.py"
    blind_ok.write_text("def f(x):\n    return x\n", encoding="utf-8")
    blind_bad = tmp_path / "blind_bad.py"
    blind_bad.write_text("# BUG\ndef f(x):\n    return x\n", encoding="utf-8")
    args = ["realism", "--out", str(out), "--blind"]
    assert audit.main([*args, str(blind_ok), demo.id, wrong]) == 0
    assert audit.main([*args, str(blind_bad), demo.id, wrong]) == 1


def test_a_base_without_hidden_tests_is_graded_on_public_only(
    audit: ModuleType, demo, tmp_path: Path, fake_grader
) -> None:
    _git(audit.REPO, "rm", "-q", "content/t/x/problems/p01-demo/test_hidden.py")
    _git(audit.REPO, "commit", "-q", "-m", "no hidden tests at base")
    (demo.path / "test_hidden.py").write_text(
        "def test_realistic():\n    pass\n# CATCH realistic data\n", encoding="utf-8"
    )
    out = tmp_path / "out"
    assert audit.main(["realism", "--out", str(out), demo.id, str(_wrong(tmp_path))]) == 0
    assert ("t/x/p01-demo", False) in fake_grader
    (wrong_row,) = [row for row in _rows(out) if row["kind"] == "wrong"]
    assert (wrong_row["base"], wrong_row["verdict"]) == ("survived", "killed")


def test_unknown_revision_and_missing_files_are_usage_errors(
    audit: ModuleType, demo, tmp_path: Path, fake_grader
) -> None:
    _add_realism_test(demo)
    wrong = str(_wrong(tmp_path))
    out = tmp_path / "out"
    assert audit.main(["realism", "--out", str(out), "--base", "no-such-rev", demo.id, wrong]) == 2
    assert audit.main(["realism", "--out", str(out), demo.id, str(tmp_path / "nope.py")]) == 2
    assert audit.main(["realism", "--out", str(out), demo.id]) == 2


def test_rows_are_reused_until_a_file_or_a_suite_changes(
    audit: ModuleType, demo, tmp_path: Path, fake_grader
) -> None:
    _add_realism_test(demo)
    out = tmp_path / "out"
    wrong = str(_wrong(tmp_path))
    argv = ["realism", "--out", str(out), demo.id, wrong]
    assert audit.main(argv) == 0
    graded = len(fake_grader)
    assert audit.main(argv) == 0
    assert len(fake_grader) == graded  # nothing changed: every row is reused
    demo.path.joinpath("test_public.py").write_text("def test_example():\n    pass\n# new\n")
    assert audit.main(argv) == 0
    assert len(fake_grader) > graded  # the public suite changed: the rows are graded again


def test_seeds_set_the_data_salt_to_each_seed(
    audit: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem_dir = tmp_path / "problem"
    problem_dir.mkdir()
    (problem_dir / "solution.py").write_text("def f(x):\n    return x\n", encoding="utf-8")
    (problem_dir / "test_public.py").write_text("def test_a():\n    pass\n", encoding="utf-8")
    (problem_dir / "test_hidden.py").write_text("def test_b():\n    pass\n", encoding="utf-8")
    problem = audit.Problem("t/x/p01", problem_dir, "python", "f", 30, ())
    monkeypatch.setattr(audit, "SEEDS", (1, 2, 3))
    envs: list[dict[str, str]] = []

    def fake_run(command, **kwargs):
        envs.append(kwargs["env"])
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(audit.subprocess, "run", fake_run)

    assert audit.cmd_seeds([problem], tmp_path / "out", workers=1) == 0
    assert sorted(env["DOGMA_DATA_SALT"] for env in envs) == ["1", "2", "3"]
    assert all(env["PYTHONHASHSEED"] == env["DOGMA_DATA_SALT"] for env in envs)
