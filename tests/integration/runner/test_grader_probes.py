"""Grader probes (ROADMAP I-28): what the verdict does when a submission misbehaves.

The rule these pin: a wrong submission must never read as passed by accident, and no submission
may crash the grader. The sandbox is not a security boundary (ENGINEERING_STANDARDS §11), so a
deliberate cheat is a known limit, pinned under ``test_known_limit_*`` so that the limit is written
down and a change to it gets noticed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dogma.grading.models import RunStatus
from dogma.runner import run

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
PY_ADD = REPO_ROOT / "tests" / "fixtures" / "problems" / "py-add"
SQL_ORDERS = REPO_ROOT / "tests" / "fixtures" / "problems" / "sql-orders"
GOOD = "def add(a, b):\n    return a + b\n"
CORRECT_SQL = (
    "SELECT country, SUM(amount) AS total FROM orders WHERE amount >= 10 GROUP BY country;\n"
)
_SKIP_AT_IMPORT = "import pytest\n\npytest.skip('x', allow_module_level=True)\n\n" + GOOD

WRONG_PYTHON = {
    "skip in the function": "import pytest\n\ndef add(a, b):\n    pytest.skip('x')\n",
    "xfail in the function": "import pytest\n\ndef add(a, b):\n    pytest.xfail('x')\n",
    "module-level skip": _SKIP_AT_IMPORT,
    "exit 0 at import": "import sys\n\nsys.exit(0)\n\n" + GOOD,
    "hard exit 0 at import": "import os\n\nos._exit(0)\n\n" + GOOD,
    "exit 0 in the function": "import sys\n\ndef add(a, b):\n    sys.exit(0)\n",
    "hard exit 0 in the function": "import os\n\ndef add(a, b):\n    os._exit(0)\n",
    "keyboard interrupt": "def add(a, b):\n    raise KeyboardInterrupt\n",
    "system exit raised": "def add(a, b):\n    raise SystemExit(0)\n",
    "base exception": "def add(a, b):\n    raise BaseException('x')\n",
    "returns NaN": "def add(a, b):\n    return float('nan')\n",
    "returns None": "def add(a, b):\n    return None\n",
    "returns a generator": "def add(a, b):\n    yield a + b\n",
    "returns text": "def add(a, b):\n    return str(a + b)\n",
    "prints the right answer": "def add(a, b):\n    print(a + b)\n",
}

WRONG_SQL = {
    "comment only": "-- nothing here\n",
    "drops the table": "DROP TABLE orders;\n",
    "two statements, wrong last": CORRECT_SQL + "SELECT 1 AS country, 2 AS total;\n",
    "semicolon only": ";\n",
}

# Each must be rejected or graded, never raised through the grader as an exception.
HOSTILE_SOURCE = {
    "deep unary chain": "x = " + "-" * 200_000 + "1\n",
    "deep parentheses": "x = " + "(" * 5_000 + "1" + ")" * 5_000 + "\n",
    "long attribute chain": "x = a" + ".b" * 100_000 + "\n",
    "null byte": "def add(a, b):\n    return a + b\n\0",
    "only a comment": "# nothing\n",
    "empty file": "",
    "lone surrogate escape": "def add(a, b):\n    return '\\ud800'\n",
}

BAN_BYPASSES = {
    "dunder import": "m = __import__('numpy.linalg')\n",
    "importlib": "import importlib\n\nm = importlib.import_module('numpy.linalg')\n",
    "getattr": "import numpy as np\n\nm = getattr(np, 'linalg')\n",
    "eval": "import numpy as np\n\nm = eval('np.lin' + 'alg')\n",
    "module dict": "import numpy as np\n\nm = np.__dict__['linalg']\n",
}


@pytest.mark.parametrize("label", sorted(WRONG_PYTHON))
def test_a_wrong_python_submission_never_passes(settings, submit, label):
    report = run(PY_ADD, submit(WRONG_PYTHON[label]), include_hidden=True, settings=settings)
    assert report.status is not RunStatus.PASSED, label


@pytest.mark.parametrize("label", sorted(WRONG_SQL))
def test_a_wrong_sql_submission_never_passes(settings, submit, label):
    file = submit(WRONG_SQL[label], "submission.sql")
    report = run(SQL_ORDERS, file, include_hidden=True, settings=settings)
    assert report.status is not RunStatus.PASSED, label


@pytest.mark.parametrize("label", sorted(HOSTILE_SOURCE))
def test_a_hostile_source_gets_a_verdict_instead_of_crashing_the_grader(settings, submit, label):
    report = run(PY_ADD, submit(HOSTILE_SOURCE[label]), include_hidden=True, settings=settings)
    assert report.status is not RunStatus.PASSED, label


def test_a_file_that_is_not_utf8_is_rejected_as_a_syntax_problem(settings, tmp_path):
    file = tmp_path / "submission.py"
    file.write_bytes(b"def add(a, b):\n    return '\xff'\n")
    report = run(PY_ADD, file, include_hidden=True, settings=settings)
    assert report.status is RunStatus.REJECTED
    assert [finding.rule for finding in report.static] == ["syntax"]


def test_a_correct_answer_with_noise_on_stdout_still_passes(settings, submit):
    source = "print('debug')\n\ndef add(a, b):\n    print(a, b)\n    return a + b\n"
    assert run(PY_ADD, submit(source), include_hidden=True, settings=settings).passed


def test_a_banned_name_as_a_plain_variable_is_not_rejected(settings, submit):
    source = "def add(a, b):\n    linalg = a + b\n    return linalg\n"
    assert run(PY_ADD, submit(source), include_hidden=True, settings=settings).passed


@pytest.mark.parametrize("label", sorted(BAN_BYPASSES))
def test_known_limit_a_deliberate_ban_bypass_is_not_caught(settings, submit, label):
    source = BAN_BYPASSES[label] + "\n" + GOOD
    report = run(PY_ADD, submit(source), include_hidden=True, settings=settings)
    assert report.status is RunStatus.PASSED, label


def test_known_limit_a_submission_can_disable_the_tests_failure_helper(settings, submit):
    # Tests run in the submission's process. Replacing pytest.fail neutralises any test that signals
    # failure through it. This is deliberate cheating, not something a learner does by accident.
    source = "import pytest\n\npytest.fail = lambda *a, **k: None\n\n" + GOOD
    assert run(PY_ADD, submit(source), include_hidden=True, settings=settings).passed
