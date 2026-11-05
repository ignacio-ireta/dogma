"""Tests for scripts/exercise_audit.py: the mutation operators, sampling and message classes.

The tool's numbers are only worth trusting if its mutants are real, so each operator is pinned on
small sources. Nothing here grades a submission; the integration step is the tool's own baseline.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

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
