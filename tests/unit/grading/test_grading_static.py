from __future__ import annotations

import ast
from pathlib import Path

import pytest

from dogma.grading.static import check_submission


def _check(
    tmp_path: Path,
    source: str | bytes,
    *,
    language: str = "python",
    entry: str = "add",
    banned: tuple[str, ...] = ("numpy.linalg",),
    name: str = "submission.py",
    runtime: str | None = None,
):
    path = tmp_path / name
    if isinstance(source, bytes):
        path.write_bytes(source)
    else:
        path.write_text(source, encoding="utf-8")
    return check_submission(
        path,
        language=language,
        entry=entry,
        banned_imports=banned,  # type: ignore[arg-type]
        runtime=runtime,
    )


def _rules(findings) -> list[str]:
    return [f.rule for f in findings]


def test_clean_python_submission_has_no_findings(tmp_path):
    assert _check(tmp_path, "def add(a, b):\n    return a + b\n") == []


def test_class_entry_is_accepted(tmp_path):
    assert _check(tmp_path, "class add:\n    pass\n") == []


def test_async_def_entry_is_accepted(tmp_path):
    assert _check(tmp_path, "async def add(a, b):\n    return a + b\n") == []


def test_syntax_error_is_reported_with_its_line(tmp_path):
    (finding,) = _check(tmp_path, "def add(a, b) return a + b\n")
    assert finding.rule == "syntax"
    assert finding.line == 1
    assert "line 1" in finding.message


def test_syntax_error_stops_further_checks(tmp_path):
    # Without a parse tree there is nothing to check the entry or the imports against.
    assert _rules(_check(tmp_path, "import numpy.linalg\ndef add(:\n")) == ["syntax"]


def test_null_bytes_are_a_syntax_finding(tmp_path):
    assert _rules(_check(tmp_path, b"def add(a, b):\x00\n")) == ["syntax"]


def test_invalid_utf8_is_a_syntax_finding(tmp_path):
    (finding,) = _check(tmp_path, b"\xff\xfe def add(a, b): pass\n")
    assert finding.rule == "syntax"
    assert "UTF-8" in finding.message


def test_missing_entry_is_reported(tmp_path):
    (finding,) = _check(tmp_path, "def plus(a, b):\n    return a + b\n")
    assert finding.rule == "missing-entry"
    assert "`add`" in finding.message


def test_entry_nested_in_a_function_does_not_count(tmp_path):
    source = "def outer():\n    def add(a, b):\n        return a + b\n    return add\n"
    assert _rules(_check(tmp_path, source)) == ["missing-entry"]


def test_entry_assigned_as_a_lambda_does_not_count(tmp_path):
    # The contract is a def or class; a lambda binding is reported rather than guessed at.
    assert _rules(_check(tmp_path, "add = lambda a, b: a + b\n")) == ["missing-entry"]


@pytest.mark.parametrize(
    "source",
    [
        "import numpy.linalg\n\ndef add(a, b):\n    return a + b\n",
        "from numpy.linalg import norm\n\ndef add(a, b):\n    return a + b\n",
        "from numpy import linalg\n\ndef add(a, b):\n    return a + b\n",
        "from numpy import linalg as la\n\ndef add(a, b):\n    return a + b\n",
        "import numpy.linalg as la\n\ndef add(a, b):\n    return a + b\n",
    ],
    ids=["import", "from-import", "from-package", "from-package-alias", "import-alias"],
)
def test_banned_module_imports_are_reported(tmp_path, source):
    findings = _check(tmp_path, source)
    assert _rules(findings) == ["banned-import"]
    assert "numpy.linalg" in findings[0].message


def test_banned_attribute_access_is_reported_with_its_line(tmp_path):
    source = (
        "import numpy as np\n\ndef add(a, b):\n    _ = np.linalg.norm([a, b])\n    return a + b\n"
    )
    (finding,) = _check(tmp_path, source)
    assert finding.rule == "banned-import"
    assert finding.line == 4


def test_parent_package_import_is_not_banned_by_a_child_rule(tmp_path):
    assert _check(tmp_path, "import numpy\n\ndef add(a, b):\n    return a + b\n") == []


def test_banned_top_level_package_matches_its_submodules(tmp_path):
    source = "import sklearn.linear_model\n\ndef add(a, b):\n    return a + b\n"
    assert _rules(_check(tmp_path, source, banned=("sklearn",))) == ["banned-import"]


def test_each_banned_hit_on_a_line_is_reported_once(tmp_path):
    source = "import numpy.linalg\nimport numpy.linalg\n\ndef add(a, b):\n    return a + b\n"
    lines = [f.line for f in _check(tmp_path, source)]
    assert lines == [1, 2]


def test_no_banned_list_means_no_banned_findings(tmp_path):
    source = "import numpy.linalg\n\ndef add(a, b):\n    return a + b\n"
    assert _check(tmp_path, source, banned=()) == []


def test_empty_sql_file_is_rejected(tmp_path):
    (finding,) = _check(tmp_path, "  \n\t\n", language="sql", entry="query", name="submission.sql")
    assert finding.rule == "empty"


def test_sql_file_with_content_passes_static_checks(tmp_path):
    # SQL gets only the non-empty check: whether the query is right is for the tests to say.
    findings = _check(tmp_path, "SELECT 1;\n", language="sql", entry="query", name="submission.sql")
    assert findings == []


def test_sql_is_never_parsed_as_python(tmp_path):
    findings = _check(
        tmp_path,
        "SELECT country FROM orders WHERE x = 'y' ;",
        language="sql",
        entry="query",
        name="submission.sql",
    )
    assert findings == []


def test_parser_resource_failures_are_reported_as_syntax_findings(tmp_path, monkeypatch):
    def _too_deep(source, filename):  # the parser can raise ValueError or RecursionError
        raise RecursionError("maximum recursion depth exceeded during parsing")

    monkeypatch.setattr(ast, "parse", _too_deep)
    (finding,) = _check(tmp_path, "x = 1\n")
    assert finding.rule == "syntax"
    assert "cannot parse" in finding.message


def test_bare_banned_name_does_not_match_same_named_attribute(tmp_path: Path) -> None:
    """Regression: banning ``random`` must not reject ``rng.random(n)`` or ``np.random``."""
    sub = tmp_path / "submission.py"
    sub.write_text(
        "import numpy as np\n"
        "def f(rng, n):\n"
        "    return rng.random(n) + np.random.default_rng(0).random(n)\n"
    )
    findings = check_submission(sub, language="python", entry="f", banned_imports=["random"])
    assert findings == []
    sub.write_text("import random\ndef f(rng, n):\n    return random.random()\n")
    findings = check_submission(sub, language="python", entry="f", banned_imports=["random"])
    assert [f.rule for f in findings] == ["banned-import"]


@pytest.mark.parametrize(
    "source",
    [
        "import dogma_data\n\ndef add(a, b):\n    return a + b\n",
        "from dogma_data import synth\n\ndef add(a, b):\n    return a + b\n",
        "from dogma_data.real import load\n\ndef add(a, b):\n    return a + b\n",
        "import datasets\n\ndef add(a, b):\n    return a + b\n",
        "from datasets import load_dataset\n\ndef add(a, b):\n    return a + b\n",
        "import huggingface_hub\n\ndef add(a, b):\n    return a + b\n",
        "from huggingface_hub import hf_hub_download\n\ndef add(a, b):\n    return a + b\n",
    ],
    ids=[
        "dogma-data-import",
        "dogma-data-from-import",
        "dogma-data-submodule",
        "datasets-import",
        "datasets-from-import",
        "huggingface-hub-import",
        "huggingface-hub-from-import",
    ],
)
def test_test_data_layer_is_banned_in_every_problem(tmp_path, source):
    # The ban is platform-wide: an empty meta.yaml banned list must not allow it.
    findings = _check(tmp_path, source, banned=())
    assert _rules(findings) == ["banned-import"]
    assert "test-data helpers" in findings[0].message
    assert "download datasets" in findings[0].message
    assert findings[0].line == 1


def test_test_data_ban_reports_each_offending_line_once(tmp_path):
    source = "import datasets\nfrom dogma_data import synth\n\ndef add(a, b):\n    return a + b\n"
    assert [f.line for f in _check(tmp_path, source, banned=())] == [1, 2]


def test_a_module_that_only_shares_a_prefix_is_not_banned(tmp_path):
    source = "import dogma_datasets_tools\nimport datasetsx\n\ndef add(a, b):\n    return a + b\n"
    assert _check(tmp_path, source, banned=()) == []


def test_test_data_attribute_names_are_not_banned(tmp_path):
    # Only imports count: a local variable or attribute named like the layer is fine.
    source = "class Box:\n    datasets = 1\n\ndef add(a, b):\n    return a + b + Box.datasets\n"
    assert _check(tmp_path, source, banned=()) == []


def test_test_data_ban_applies_alongside_the_problem_ban(tmp_path):
    source = "import numpy.linalg\nimport datasets\n\ndef add(a, b):\n    return a + b\n"
    findings = _check(tmp_path, source)
    assert [(f.rule, f.line) for f in findings] == [("banned-import", 1), ("banned-import", 2)]
    assert "numpy.linalg" in findings[0].message
    assert "test-data helpers" in findings[1].message


def test_platform_ban_names_the_mistake_for_a_submodule_import(tmp_path):
    findings = _check(tmp_path, "from dogma_data import synth\n\ndef add(a, b):\n    return a\n")
    assert findings[0].message.startswith("submissions may not import test-data helpers")
    assert "dogma_data" in findings[0].message


def test_platform_ban_does_not_apply_to_sql(tmp_path):
    findings = _check(
        tmp_path, "SELECT 1;\n", language="sql", entry="query", name="submission.sql", banned=()
    )
    assert findings == []


# --- runtime import rule (ADR-0021) ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("runtime", "source"),
    [
        ("py-stdlib", "import math, collections\nfrom typing import Any\nimport heapq as hq\n"),
        ("py-numpy", "import numpy as np\nfrom numpy.typing import NDArray\n"),
        (
            "py-data",
            "import pandas as pd\nfrom sklearn.linear_model import Ridge\nimport scipy.stats\n",
        ),
        ("py-torch", "import torch\nimport numpy as np\nfrom torch import nn\n"),
    ],
    ids=["stdlib", "numpy", "data", "torch"],
)
def test_imports_inside_the_runtime_are_accepted(tmp_path, runtime, source):
    body = "from __future__ import annotations\n" + source + "def add(a, b):\n    return a + b\n"
    assert _check(tmp_path, body, banned=(), runtime=runtime) == []


@pytest.mark.parametrize(
    ("runtime", "source", "module"),
    [
        ("py-stdlib", "import numpy as np\n", "numpy"),
        ("py-numpy", "import scipy.special\n", "scipy"),
        ("py-numpy", "from sklearn.preprocessing import MaxAbsScaler\n", "sklearn"),
        ("py-numpy", "import pandas\n", "pandas"),
        ("py-data", "import torch\n", "torch"),
        ("py-numpy", "import dogma.content\n", "dogma"),
    ],
    ids=[
        "numpy-in-stdlib",
        "scipy-in-numpy",
        "sklearn-in-numpy",
        "pandas-in-numpy",
        "torch-in-data",
        "app",
    ],
)
def test_imports_outside_the_runtime_are_rejected_and_name_it(tmp_path, runtime, source, module):
    findings = _check(
        tmp_path, source + "def add(a, b):\n    return a + b\n", banned=(), runtime=runtime
    )
    assert _rules(findings) == ["runtime-import"]
    assert f"`{module}`" in findings[0].message
    assert runtime in findings[0].message
    assert findings[0].line == 1


def test_no_runtime_or_an_unknown_runtime_skips_the_rule(tmp_path):
    source = "import sklearn\ndef add(a, b):\n    return a + b\n"
    assert _check(tmp_path, source, banned=()) == []
    assert _check(tmp_path, source, banned=(), runtime="py-jax") == []


def test_a_module_banned_by_the_problem_is_reported_once(tmp_path):
    source = "import torch\ndef add(a, b):\n    return a + b\n"
    findings = _check(tmp_path, source, banned=("torch",), runtime="py-numpy")
    assert _rules(findings) == ["banned-import"]


def test_a_platform_banned_module_gets_only_the_platform_message(tmp_path):
    source = "import dogma_data\ndef add(a, b):\n    return a + b\n"
    findings = _check(tmp_path, source, banned=(), runtime="py-stdlib")
    assert _rules(findings) == ["banned-import"]
    assert "test-data helpers" in findings[0].message


def test_each_outside_import_is_listed_with_its_line(tmp_path):
    source = (
        "import os\nimport scipy\nimport pandas as pd, numpy\ndef add(a, b):\n    return a + b\n"
    )
    findings = _check(tmp_path, source, banned=(), runtime="py-numpy")
    assert [(f.rule, f.line) for f in findings] == [("runtime-import", 2), ("runtime-import", 3)]
    assert "`pandas`" in findings[1].message


def test_every_runtime_in_use_has_a_rule():
    import yaml  # noqa: PLC0415

    from dogma.grading.static import RUNTIME_IMPORTS  # noqa: PLC0415

    repo = Path(__file__).resolve().parents[3]
    used = {
        yaml.safe_load(meta.read_text(encoding="utf-8"))["runtime"]
        for meta in repo.glob("content/*/*/problems/*/meta.yaml")
    }
    python_runtimes = {name for name in used if name.startswith("py-")}
    assert python_runtimes <= set(RUNTIME_IMPORTS), python_runtimes - set(RUNTIME_IMPORTS)
