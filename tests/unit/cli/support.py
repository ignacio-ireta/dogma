"""Report builders for the CLI tests. Plain helpers; fixtures live in conftest.py."""

from __future__ import annotations

from dogma.grading.models import GradeReport, RunStatus, StaticFinding
from dogma.grading.models import TestResult as Result

PROBLEM = "cml/vector-operations/p01-vector-projection"


def make_report(
    status: RunStatus = RunStatus.PASSED,
    *,
    tests: tuple[Result, ...] = (),
    static: tuple[StaticFinding, ...] = (),
    message: str = "",
) -> GradeReport:
    return GradeReport(
        attempt_id="a1",
        problem_id=PROBLEM,
        problem_version=1,
        runtime="py-numpy",
        status=status,
        tests=tests,
        static=static,
        message=message,
    )


def result(name: str, file: str, outcome: str, message: str = "") -> Result:
    return Result(name=name, file=file, outcome=outcome, message=message)  # type: ignore[arg-type]


PUBLIC_PASS = tuple(result(f"test_pub_{i}", "public", "pass") for i in range(4))
HIDDEN_PASS = tuple(result(f"test_hid_{i}", "hidden", "pass") for i in range(11))
