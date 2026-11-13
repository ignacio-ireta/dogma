"""Tests for scripts/exercise_probe.py: each generated wrapper does what its name says.

A probe is the reference plus one wrapper. Its verdict is only worth reading if the wrapper keeps
the result right, changes exactly the thing it names, and leaves every other input alone. The
wrappers are pinned here on small sources. Only the last test runs pytest, on a fixture problem.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "exercise_probe.py"
FIXTURE = ROOT / "tests" / "fixtures" / "problems" / "py-variants"

TOTAL = "def f(x):\n    return float(sum(x))\n"
SORTED_COPY = (
    "import numpy as np\n"
    "\n"
    "\n"
    "def f(x, k=1):\n"
    "    if k < 0:\n"
    "        raise ValueError('k is negative')\n"
    "    return np.sort(np.asarray(x, dtype=float))\n"
)
NESTED = "import numpy as np\n\n\ndef f():\n    return np.array([1.0, 2.0]), [3, {'k': 4}]\n"


@pytest.fixture(scope="module")
def probe() -> ModuleType:
    spec = importlib.util.spec_from_file_location("exercise_probe", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _problem(probe: ModuleType, directory: Path, source: str, *, language: str = "python") -> Any:
    suffix = ".sql" if language == "sql" else ".py"
    (directory / f"solution{suffix}").write_text(source, encoding="utf-8")
    return probe.audit.Problem(
        id="track/topic/p01-f",
        path=directory,
        language=language,
        entry="f",
        timeout_s=30,
        banned=(),
        runtime="py-numpy",
    )


def _namespace(source: str) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    exec(compile(source, "probe", "exec"), namespace)  # noqa: S102 - our own generated sources
    return namespace


def test_entry_params_come_in_call_order_and_only_for_function_entries(probe, tmp_path):
    function = _problem(probe, tmp_path, "def f(x, y=1, *, seed):\n    return x\n")
    assert probe.entry_params(function) == ["x", "y", "seed"]

    class_dir = tmp_path / "class"
    class_dir.mkdir()
    assert probe.entry_params(_problem(probe, class_dir, "class f:\n    pass\n")) is None

    sql_dir = tmp_path / "sql"
    sql_dir.mkdir()
    assert probe.entry_params(_problem(probe, sql_dir, "SELECT 1", language="sql")) is None


def test_scribble_keeps_the_result_and_changes_only_the_named_argument(probe, tmp_path):
    problem = _problem(probe, tmp_path, TOTAL)
    f = _namespace(probe.scribble_source(problem, ["x"], "x", "sort", "ndarray"))["f"]

    values = np.array([3.0, 1.0, 2.0])
    assert f(values) == 6.0
    assert values.tolist() == [1.0, 2.0, 3.0]

    as_list = [3.0, 1.0, 2.0]
    assert f(as_list) == 6.0
    assert as_list == [3.0, 1.0, 2.0]  # the wrapper is for arrays: a list is another probe

    frozen = np.array([3.0, 1.0, 2.0])
    frozen.setflags(write=False)
    with pytest.raises(ValueError, match="read-only"):
        f(frozen)


def test_scribble_changes_nothing_where_the_change_does_not_show(probe, tmp_path):
    problem = _problem(probe, tmp_path, TOTAL)
    f = _namespace(probe.scribble_source(problem, ["x"], "x", "abs", "ndarray"))["f"]
    values = np.array([0.5, 0.0, 2.0])
    f(values)
    assert values.tolist() == [0.5, 0.0, 2.0]  # no negative entry, so nothing to see

    centre = _namespace(probe.scribble_source(problem, ["x"], "x", "centre", "ndarray"))["f"]
    whole = np.array([1, 2, 6])
    centre(whole)
    assert whole.tolist() == [1, 2, 6]  # a float-only change never touches an integer array


def test_scribbles_of_a_list_reorder_or_reach_inside_it(probe, tmp_path):
    problem = _problem(probe, tmp_path, "def f(x):\n    return len(x)\n")
    first = _namespace(probe.scribble_source(problem, ["x"], "x", "min_first", "list"))["f"]
    items = [3, 1, 2]
    assert first(items) == 3
    assert items == [1, 3, 2]

    inner = _namespace(probe.scribble_source(problem, ["x"], "x", "inner", "list"))["f"]
    grid = [[1, 2], [3, 4]]
    snapshot = list(grid)  # a shallow copy: the kind of snapshot that cannot see the write
    inner(grid)
    assert grid == [[2, 2], [3, 4]]
    assert snapshot == grid


def test_return_probes_reach_values_inside_tuples_lists_and_dicts(probe, tmp_path):
    problem = _problem(probe, tmp_path, NESTED)

    array, rest = _namespace(probe.returns_source(problem, "int_as_float"))["f"]()
    assert array.dtype == np.float64
    assert rest == [3.0, {"k": 4.0}]
    assert type(rest[0]) is float and type(rest[1]["k"]) is float

    as_list, _ = _namespace(probe.returns_source(problem, "array_as_list"))["f"]()
    assert as_list == [1.0, 2.0] and type(as_list) is list

    narrow, _ = _namespace(probe.returns_source(problem, "float64_as_float32"))["f"]()
    assert narrow.dtype == np.float32

    whole = _namespace(probe.returns_source(problem, "tuple_as_list"))["f"]()
    assert type(whole) is list and type(whole[0]) is np.ndarray  # only the outer container


def test_a_float_result_can_come_back_as_a_one_element_array(probe, tmp_path):
    problem = _problem(probe, tmp_path, TOTAL)
    f = _namespace(probe.returns_source(problem, "float_as_one_element_array"))["f"]
    result = f([1.0, 2.5])
    assert isinstance(result, np.ndarray) and result.shape == (1,) and result[0] == 3.5


def test_form_probes_fail_only_for_the_form_they_name(probe, tmp_path):
    problem = _problem(probe, tmp_path, TOTAL)
    needs_array = _namespace(probe.form_source(problem, ["x"], "x", "needs_array"))["f"]
    assert needs_array(np.array([1.0, 2.0])) == 3.0
    with pytest.raises(AttributeError):
        needs_array([1.0, 2.0])

    needs_float = _namespace(probe.form_source(problem, ["x"], "x", "needs_float"))["f"]
    assert needs_float(np.array([1.0, 2.0])) == 3.0
    assert needs_float([1.5, 2.0]) == 3.5
    with pytest.raises(TypeError):
        needs_float(np.array([1, 2]))
    with pytest.raises(TypeError):
        needs_float([[1, 2], [3, 4]])  # a rectangular list of whole numbers is an integer array


def test_recorder_notes_classes_raising_calls_and_changes_that_would_show(
    probe, tmp_path, monkeypatch
):
    problem = _problem(probe, tmp_path, SORTED_COPY)
    target = tmp_path / "record.json"
    monkeypatch.setenv("DOGMA_PROBE_OUT", str(target))
    namespace = _namespace(probe.recorder_source(problem, ["x", "k"]))
    f = namespace["f"]

    assert f(np.array([3.0, -1.0, 2.0])).tolist() == [-1.0, 2.0, 3.0]
    assert f([1, 2, 2]).tolist() == [1.0, 2.0, 2.0]
    with pytest.raises(ValueError, match="negative"):
        f(np.array([1.0]), k=-1)
    namespace["_probe_dump"]()

    record = json.loads(target.read_text(encoding="utf-8"))
    assert record["params"] == ["x", "k"]
    assert record["calls"] == [2, 1]
    given = record["ok"]["x"]
    for feature in ("type_ndarray", "type_list", "negative", "unsorted", "ties", "ints_only"):
        assert feature in given, feature
    assert "one_row" in record["raised"]["x"] and "one_row" not in given
    assert "negative" in record["raised"]["k"]
    assert "leaf_ndarray_f" in record["ok"]["return"]
    # Sorting the unsorted array would show; taking |x| of the list [1, 2, 2] would not.
    assert record["visible"]["x"]["sort"] == 1
    assert record["visible"]["x"]["abs"] == 1


def test_recorder_leaves_the_arguments_it_inspects_untouched(probe, tmp_path):
    problem = _problem(probe, tmp_path, SORTED_COPY)
    f = _namespace(probe.recorder_source(problem, ["x", "k"]))["f"]
    values = np.array([3.0, -1.0, 2.0])
    f(values)
    assert values.tolist() == [3.0, -1.0, 2.0]


def test_plans_follow_what_the_record_saw(probe):
    row = {
        "params": ["x", "labels", "cfg"],
        "ok": {
            "x": {"type_ndarray": 4, "dtype_f": 4},
            "labels": {"type_list": 4},
            "cfg": {"type_dict": 4},
            "return": {"type_tuple": 4, "leaf_ndarray_f": 4, "leaf_int": 4},
            "return.0": {"type_ndarray": 4},
            "return.1": {"type_int": 4},
        },
    }
    scribbles = set(probe.scribble_plan(row))
    assert ("x", "ndarray", "sort") in scribbles
    assert ("labels", "list", "reverse") in scribbles
    assert ("cfg", "dict", "any") in scribbles
    assert not any(kind == "Tensor" for _, kind, _ in scribbles)

    returns = probe.returns_plan(row)
    assert {"array_as_list", "float64_as_float32", "int_as_float", "tuple_as_list"} <= set(returns)
    assert "tensor_detached" not in returns and "list_reversed" not in returns


def test_a_list_handed_to_an_array_function_gets_no_list_scribble(probe):
    row = {"params": ["x"], "ok": {"x": {"type_ndarray": 2, "type_list": 2}}}
    kinds = {type_name for _, type_name, _ in probe.scribble_plan(row)}
    assert kinds == {"ndarray"}  # np.asarray copies a list, so only the array can be written to


def test_an_argument_given_in_two_containers_gets_two_named_probes(probe):
    row = {"params": ["ids"], "ok": {"ids": {"type_list": 3, "type_set": 3}}}
    names = [name for name, *_ in probe.scribble_names(row)]
    assert "ids:any" in names and "ids:any@set" in names
    assert len(names) == len(set(names))


def test_gaps_tell_never_given_from_given_only_to_calls_that_raise(probe):
    ok = {
        "type_ndarray": 9,
        "ndim1": 9,
        "dtype_f": 9,
        "negative": 5,
        "distinct": 9,
        "ascending": 9,
    }
    never, only_raised = probe._gaps(ok, {"empty": 1})
    assert only_raised == ["empty"]
    for name in ("one_row", "zero", "ties", "unsorted", "as_list", "integers"):
        assert name in never, name
    assert "negative" not in never

    assert probe._gaps({"type_str": 3}, {}) == ([], [])  # a string argument has no classes here


def test_mutation_clause_finds_the_bullet_that_promises_untouched_inputs(probe, tmp_path):
    problem = _problem(probe, tmp_path, TOTAL)
    statement = (
        "# F\n\nSum.\n\n## Contract\n\n- `x`: a sequence of numbers.\n"
        "- Returns: a float.\n- Must not mutate `x`.\n\n## Examples\n\nNone.\n"
    )
    (tmp_path / "problem.md").write_text(statement, encoding="utf-8")
    assert probe.mutation_clause(problem) == ("C3", "Must not mutate `x`.")

    (tmp_path / "problem.md").write_text(
        statement.replace("- Must not mutate `x`.\n", ""), encoding="utf-8"
    )
    assert probe.mutation_clause(problem) is None


def test_record_runs_the_tests_of_a_fixture_problem(probe):
    problem = probe.audit.Problem(
        id="fixtures/py-variants/p01-add",
        path=FIXTURE,
        language="python",
        entry="add",
        timeout_s=30,
        banned=(),
        runtime="py-stdlib",
    )
    row = probe.record_problem(problem, probe.entry_params(problem))
    assert row["status"] == "passed"
    assert row["params"] == ["a", "b"]
    assert {"type_int", "type_float", "negative"} <= set(row["ok"]["a"])
    assert "zero" in row["ok"]["b"]  # add(7, 0) in the public tests
    assert {"type_NoneType", "type_str"} <= set(row["raised"]["a"])
    assert row["calls"][1] == 2  # None and text are the two calls the fixture expects to raise


def test_visible_calls_reads_the_count_for_a_probe_of_either_container(probe):
    row = {"visible": {"ids": {"any": 7, "sort": 0}}}
    assert probe.visible_calls(row, "ids:any") == 7
    assert probe.visible_calls(row, "ids:any@set") == 7
    assert probe.visible_calls(row, "ids:sort") == 0
    assert probe.visible_calls(row, "other:abs") == 0
    assert probe.visible_calls({}, "ids:any") == 0


def test_survivors_groups_the_surviving_probes_of_the_chosen_problems(probe, tmp_path):
    rows = [
        {"problem": "a/b/p01", "probe": "x:sort", "verdict": "survived"},
        {"problem": "a/b/p01", "probe": "x:any", "verdict": "killed_hidden"},
        {"problem": "a/b/p02", "probe": "x:abs", "verdict": "survived"},
        {"problem": "c/d/p01", "probe": "y:abs", "verdict": "survived"},
    ]
    text = "".join(json.dumps(row) + "\n" for row in rows)
    (tmp_path / "scribble.jsonl").write_text(text, encoding="utf-8")
    graded, grouped = probe.survivors(tmp_path, "scribble", {"a/b/p01", "a/b/p02"})
    assert graded == 3
    assert dict(grouped) == {"a/b/p01": ["x:sort"], "a/b/p02": ["x:abs"]}
