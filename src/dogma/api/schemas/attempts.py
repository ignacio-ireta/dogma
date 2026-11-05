"""Attempts: grade the workspace file, record it, and read a recorded attempt back."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from dogma.api.schemas.resolves import TimedVerdictOut
from dogma.grading.models import FileKind, Outcome, RunStatus, StaticFinding


class AttemptRequest(BaseModel):
    """Only the problem and the mode. The API always grades the learner's workspace file."""

    model_config = ConfigDict(extra="forbid")

    problem_id: str
    mode: Literal["run", "submit"]


class LearnerTest(BaseModel):
    """One test as the learner may see it.

    ``message`` is the full text for a public test. For a hidden test it is one safe line (no
    inputs, no values) when the test failed or errored, and empty otherwise.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    file: FileKind
    outcome: Outcome
    message: str
    duration_ms: int


class LearnerReport(BaseModel):
    """A GradeReport as the browser and the CLI may show it (ENGINEERING_STANDARDS §4.5).

    ``findings`` are the static findings in plain words. ``headline`` is a sentence for a status
    with no per-test detail (timeout, out of memory, could not run), or null.
    """

    model_config = ConfigDict(frozen=True)

    attempt_id: str
    problem_id: str
    problem_version: int
    runtime: str
    status: RunStatus
    headline: str | None
    tests: list[LearnerTest]
    static: list[StaticFinding]
    findings: list[str]
    stdout: str
    duration_ms: int
    message: str


class AttemptCreated(BaseModel):
    """The graded attempt. ``timed_resolve`` is the verdict of the open timed re-solve this submit
    closed (null for a run, for a submit with no open window, and when recording is off)."""

    mode: Literal["run", "submit"]
    recorded: bool
    record_warning: str | None
    report: LearnerReport
    timed_resolve: TimedVerdictOut | None


class AttemptRecord(BaseModel):
    id: str
    problem_id: str
    problem_version: int
    mode: str
    status: str
    public_passed: int
    public_total: int
    hidden_passed: int
    hidden_total: int
    hints_used: int
    started_at: datetime
    submitted_at: datetime
    duration_ms: int
