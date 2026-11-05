"""Grading result models (ARCHITECTURE §4.3).

Frozen Pydantic models: domain data crosses module boundaries as models, never raw dicts
(ENGINEERING_STANDARDS §4.2). A learner's failing code is a ``GradeReport`` status, not an
exception.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict

FileKind = Literal["public", "hidden"]
Outcome = Literal["pass", "fail", "error", "skip"]


class RunStatus(StrEnum):
    """Overall verdict for one attempt. ``rejected``: static checks stopped it before running."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    TIMEOUT = "timeout"
    OOM = "oom"
    REJECTED = "rejected"


class TestResult(BaseModel):
    """Outcome of one test function. ``message`` names the kind of mistake when it failed."""

    model_config = ConfigDict(frozen=True)

    name: str
    file: FileKind
    outcome: Outcome
    message: str = ""
    duration_ms: int = 0


class StaticFinding(BaseModel):
    """One problem found by a static check before execution."""

    model_config = ConfigDict(frozen=True)

    rule: str
    message: str
    line: int | None = None


class GradeReport(BaseModel):
    """Everything the learner (or the CLI, tutor and progress code) needs about one attempt."""

    model_config = ConfigDict(frozen=True)

    attempt_id: str
    problem_id: str
    problem_version: int
    runtime: str
    status: RunStatus
    tests: tuple[TestResult, ...] = ()
    static: tuple[StaticFinding, ...] = ()
    stdout: str = ""
    duration_ms: int = 0
    # Learner-safe explanation for statuses without per-test detail (error, timeout, oom,
    # rejected). Empty when the tests say everything.
    message: str = ""

    @property
    def passed(self) -> bool:
        """True only when every test ran and passed."""
        return self.status is RunStatus.PASSED

    def counts(self, file: FileKind) -> tuple[int, int]:
        """Return ``(passed, total)`` for the public or the hidden test file."""
        mine = [t for t in self.tests if t.file == file]
        return sum(t.outcome == "pass" for t in mine), len(mine)
