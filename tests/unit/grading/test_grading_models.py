from __future__ import annotations

import pytest
from pydantic import ValidationError

from dogma.grading.models import GradeReport, RunStatus, StaticFinding
from dogma.grading.models import TestResult as Result


def _test(name: str, file: str, outcome: str) -> Result:
    return Result(name=name, file=file, outcome=outcome)  # type: ignore[arg-type]


def _report(status: RunStatus, *tests: Result) -> GradeReport:
    return GradeReport(
        attempt_id="a1",
        problem_id="cml/x/p01",
        problem_version=2,
        runtime="py-numpy",
        status=status,
        tests=tests,
    )


def test_passed_is_true_only_for_the_passed_status():
    assert _report(RunStatus.PASSED).passed is True
    for status in (RunStatus.FAILED, RunStatus.ERROR, RunStatus.TIMEOUT, RunStatus.OOM):
        assert _report(status).passed is False, f"{status} must not count as passed"


def test_counts_split_public_and_hidden_tests():
    report = _report(
        RunStatus.FAILED,
        _test("a", "public", "pass"),
        _test("b", "public", "fail"),
        _test("c", "hidden", "pass"),
        _test("d", "hidden", "error"),
        _test("e", "hidden", "skip"),
    )
    assert report.counts("public") == (1, 2)
    assert report.counts("hidden") == (1, 3), "skips and errors are totals but not passes"


def test_counts_are_zero_when_a_file_never_ran():
    assert _report(RunStatus.REJECTED).counts("hidden") == (0, 0)


def test_reports_are_frozen():
    report = _report(RunStatus.PASSED)
    with pytest.raises(ValidationError):
        report.status = RunStatus.FAILED  # type: ignore[misc]


def test_unknown_outcome_is_rejected():
    with pytest.raises(ValidationError):
        Result(name="x", file="public", outcome="maybe")  # type: ignore[arg-type]


def test_report_round_trips_through_json_with_status_as_string():
    report = _report(RunStatus.TIMEOUT, _test("slow", "public", "error"))
    restored = GradeReport.model_validate_json(report.model_dump_json())
    assert restored == report
    assert '"status":"timeout"' in report.model_dump_json()


def test_static_finding_line_is_optional():
    finding = StaticFinding(rule="missing-entry", message="define add")
    assert finding.line is None
