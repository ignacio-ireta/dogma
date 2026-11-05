"""Builders for the DB tests: GradeReports with a chosen number of passing tests."""

from __future__ import annotations

from dogma.grading.models import GradeReport, RunStatus, TestResult

PROBLEM = "cml/vector-operations/p01-vector-projection"
ATTEMPT = "20261024T090000Z-aaaaaaaa"


def make_report(
    status: RunStatus = RunStatus.PASSED,
    *,
    public: tuple[int, int] = (4, 4),
    hidden: tuple[int, int] = (0, 0),
    problem_id: str = PROBLEM,
    attempt_id: str = ATTEMPT,
) -> GradeReport:
    """A report whose first ``passed`` tests of each file pass and the rest fail."""
    tests = [*_results("public", *public), *_results("hidden", *hidden)]
    return GradeReport(
        attempt_id=attempt_id,
        problem_id=problem_id,
        problem_version=2,
        runtime="py-numpy",
        status=status,
        tests=tuple(tests),
    )


def _results(file: str, passed: int, total: int) -> list[TestResult]:
    return [
        TestResult(
            name=f"test_{file}_{index}",
            file=file,  # type: ignore[arg-type]
            outcome="pass" if index < passed else "fail",
        )
        for index in range(total)
    ]
