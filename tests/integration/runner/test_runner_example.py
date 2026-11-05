"""The example topic's problem (content/cml/vector-operations) through ``runner.run``.

The reference solution must pass and the starter must not, so the content gate can trust the
runner (CONTENT_SPEC §7).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dogma.grading.models import RunStatus
from dogma.runner import run

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = REPO_ROOT / "content" / "cml" / "vector-operations" / "problems" / "p01-vector-projection"


def test_reference_solution_passes_every_public_and_hidden_test(settings):
    report = run(EXAMPLE, EXAMPLE / "solution.py", include_hidden=True, settings=settings)
    assert report.status is RunStatus.PASSED, report.message or report.tests
    public_passed, public_total = report.counts("public")
    hidden_passed, hidden_total = report.counts("hidden")
    assert public_passed == public_total >= 3
    assert hidden_passed == hidden_total >= 5
    assert report.problem_id == "cml/vector-operations/p01-vector-projection"


def test_starter_is_not_passed(settings):
    report = run(EXAMPLE, EXAMPLE / "starter.py", include_hidden=True, settings=settings)
    assert report.status is RunStatus.FAILED
    assert not report.passed
    assert report.counts("public")[0] == 0, "the starter raises NotImplementedError on every call"
    assert "NotImplementedError" in report.tests[0].message
