"""Timed re-solves: the open window, the verdict a submit returns, due items and readiness.

These are the wire shapes of ``/resolves`` and of the additive fields on attempts, sessions and
progress. Every time is an aware UTC datetime. ``budget_s`` is seconds; ``used_s`` too.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from dogma.api.schemas.interview import InterviewReadyOut


class StartResolveRequest(BaseModel):
    """The core problem to time. Unknown fields are a 422."""

    model_config = ConfigDict(extra="forbid")

    problem_id: str


class ResolveWindowOut(BaseModel):
    """One timed re-solve. ``status`` is ``open`` while it runs, then its verdict or ``abandoned``.

    ``deadline`` is ``started_at + budget_s``. Passing it lets the browser count down without a
    server round trip. ``hints_used`` counts hints from the tutor routes while the window was open.
    """

    id: str
    problem_id: str
    status: str
    started_at: datetime
    budget_s: int
    deadline: datetime
    hints_used: int
    closed_at: datetime | None


class ResolveStartedOut(BaseModel):
    """The window just opened. The starter is in the workspace; the old file was kept if any."""

    window: ResolveWindowOut
    previous_file_kept: bool


class ReSolveDueOut(BaseModel):
    """A core problem's next timed re-solve: its stage (0 to 3), when it is due, and why."""

    problem_id: str
    stage: int
    due_at: datetime
    due_now: bool
    reason: str


class TimedVerdictOut(BaseModel):
    """What a submit did to the open window on its problem (null when none was open).

    ``status`` is ``passed``, ``partial`` or ``failed``. ``next_due`` is the problem's schedule
    after this window closed.
    """

    problem_id: str
    status: str
    used_s: int
    budget_s: int
    hints_used: int
    next_due: ReSolveDueOut | None


class TimedResultOut(BaseModel):
    """A closed timed re-solve as progress shows it: the verdict and time used against budget."""

    problem_id: str
    status: str
    used_s: int
    budget_s: int
    closed_at: datetime


class ReadinessOut(BaseModel):
    """Readiness on core problems: the next three due re-solves, the last five results, and how
    many topics are interview-ready (I-18e).

    There are no streaks, debt or overdue counts (ADR-0013). ``due_now`` marks an item that is due.
    """

    next_due: list[ReSolveDueOut]
    recent_results: list[TimedResultOut]
    interview_ready: InterviewReadyOut


class TimedResolveOut(BaseModel):
    """The timed re-solve a Consolidate session plans, for the browser's start action.

    ``open`` is true when the window is already running, so the action is to continue it.
    """

    problem_id: str
    title: str
    minutes: int
    open: bool
