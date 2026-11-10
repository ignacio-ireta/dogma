"""Learner-safe wording for a GradeReport (ARCHITECTURE §4.3, ENGINEERING_STANDARDS §4.5).

Shared by the CLI and the API. A hidden test's message can name its inputs and values, so the only
text that leaves the server for a hidden test is ``hidden_summary``: the first line that names the
mistake, with no values. Public tests show their full message, because the learner was given those
inputs. Pure: no I/O, and nothing here imports the CLI or the API (import-linter).
"""

from __future__ import annotations

import re
from typing import Final

from dogma.grading.models import GradeReport, RunStatus, StaticFinding, TestResult

NO_DETAIL: Final = "(the test failed; its name describes what was checked)"
# Lines from numpy.testing, pytest and hypothesis output that carry values, not the mistake.
_VALUE_PREFIXES: Final = (
    "x:",
    "y:",
    "ACTUAL",
    "DESIRED",
    "Max absolute",
    "Max relative",
    "Mismatched",
    "Arrays are not",
    "Mismatch at",
    "+",
    "-",
    "E ",
    "Not equal to tolerance",
    "Falsifying example",
    "assert ",
    "where ",
)
_FALSIFYING: Final = "Falsifying example"
_BACKTICKED: Final = re.compile(r"`([^`]+)`")
_HEADLINES: Final[dict[RunStatus, str]] = {
    # Name the usual kinds of mistake, since a killed run has no failing test to point at.
    RunStatus.TIMEOUT: (
        "Timed out: usually a loop that never ends (a bound that never moves, a missing"
        " update) or an algorithm too slow for the input size"
    ),
    RunStatus.OOM: (
        "Out of memory: usually building something far too large (an n x n matrix,"
        " a whole stream held in a list, unbounded output)"
    ),
    RunStatus.ERROR: "Could not run the tests",
}


def hidden_summary(message: str) -> str:
    """The first line of a hidden-test failure that names the mistake, without any values."""
    lines: list[str] = []
    for raw in message.splitlines():
        line = raw.strip()
        while line.startswith(("AssertionError:", "Failed:")):
            line = line.partition(":")[2].lstrip()
        lines.append(line)
    # Hypothesis lists the falsifying inputs after this marker, so nothing from it on is shown.
    cut = next((i for i, line in enumerate(lines) if line.startswith(_FALSIFYING)), len(lines))
    keep = [line for line in lines[:cut] if line and not line.startswith(_VALUE_PREFIXES)]
    if not keep:
        return NO_DETAIL
    return keep[0]


def learner_message(test: TestResult) -> str:
    """The text a learner may see for one test.

    Public tests keep their full message. Hidden tests show ``hidden_summary`` when they failed or
    errored, and nothing otherwise (a passing hidden test has no detail to show).
    """
    if test.file == "public":
        return test.message
    if test.outcome not in ("fail", "error"):
        return ""
    return hidden_summary(test.message)


def describe_finding(finding: StaticFinding) -> str:
    """A static finding in plain words, e.g. 'line 3 uses numpy.linalg, which this problem bans'."""
    if finding.rule == "banned-import":
        match = _BACKTICKED.search(finding.message)
        if match is None:  # the platform-wide ban (test-data helpers) explains itself
            return finding.message
        where = f"line {finding.line} " if finding.line is not None else ""
        return f"{where}uses {match.group(1)}, which this problem bans"
    if finding.rule == "runtime-import":
        return finding.message.replace("`", "")
    if finding.rule == "missing-entry":
        return f"the file must {finding.message}"
    if finding.rule == "syntax":
        return f"syntax error: {finding.message}"
    return finding.message


def status_headline(status: RunStatus) -> str | None:
    """A plain sentence for a status that has no per-test detail, or None when none is needed."""
    return _HEADLINES.get(status)


def findings_in_words(report: GradeReport) -> list[str]:
    """Every static finding of a report, in plain words, in the order the checks produced them."""
    return [describe_finding(finding) for finding in report.static]
