"""Mock interviews over HTTP: the plan, the reveals, the 30 scores, the report and the debrief.

A mock is one open run at a time. Its plan shows the case brief, the coding problem and the theory
question's title; the question's body and the case's reveals and debrief are never sent until the
learner asks (reveal) or has scored (debrief).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, StrictInt

MAX_REVEAL_QUESTION = 300


class MockStartRequest(BaseModel):
    """Start a mock. Without ``case_id`` the planner's rule picks the case."""

    model_config = ConfigDict(extra="forbid")

    case_id: str | None = Field(default=None, max_length=200)


class CaseOut(BaseModel):
    """The case of the mock: its id, title and the candidate-facing brief."""

    id: str
    title: str
    brief: str | None


class CodingOut(BaseModel):
    problem_id: str
    title: str


class TheoryOut(BaseModel):
    """The theory question: its topic, the topic's title, and the question's number and title."""

    topic_id: str
    title: str
    question: int | None
    question_title: str | None


class PartTimingOut(BaseModel):
    """A part's planned minutes. Shown calmly: a plan, not a deadline."""

    title: str
    minutes: str


class FocusOut(BaseModel):
    """The part to work on: the weakest over the last three scored mocks."""

    part_id: str
    title: str
    percent: float


class MockPlanOut(BaseModel):
    """An open mock: what it will work through, and how long each part is planned for."""

    session_id: str
    started_at: datetime
    case: CaseOut | None
    coding: CodingOut | None
    theory: TheoryOut | None
    parts: list[PartTimingOut]
    focus: FocusOut | None


class RevealRequest(BaseModel):
    """What you asked the interviewer, in your words. A reveal comes only for a match."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=MAX_REVEAL_QUESTION)


class RevealOut(BaseModel):
    """The interviewer's reply. ``matched`` is false when nothing was released; ``exhibit`` is a
    repo-relative path to a chart or table that comes with the reveal, or null."""

    text: str
    matched: bool
    exhibit: str | None


class ScoreRequest(BaseModel):
    """The 30 item scores, each a whole number from 1 to 4, keyed by item id. Nothing else."""

    model_config = ConfigDict(extra="forbid")

    scores: dict[str, StrictInt]


class PartScoreOut(BaseModel):
    """One part's total against its maximum. ``percent`` is 0 to 100, so parts compare fairly."""

    part_id: str
    title: str
    minutes: str
    score: int
    maximum: int
    percent: float


class MockBandOut(BaseModel):
    label: str
    low: int
    high: int
    text: str


class MockReportOut(BaseModel):
    """A scored mock: part totals, the total and its band, and the strongest and weakest parts."""

    session_id: str
    parts: list[PartScoreOut]
    total: int
    maximum: int
    band: MockBandOut
    strongest: PartScoreOut
    weakest: PartScoreOut


class MockDebriefOut(BaseModel):
    """A scored mock's report with its case's debrief (an outline for a strong answer)."""

    report: MockReportOut
    case_id: str | None
    case_debrief: str | None


class MockStatusOut(BaseModel):
    """What is open now, and the last scored mock (for a debrief after a reload). Always 200."""

    open: MockPlanOut | None
    last_scored: MockReportOut | None


class AbandonedOut(BaseModel):
    """The mock closed without scores. Nothing was stored for it."""

    session_id: str
    status: str
