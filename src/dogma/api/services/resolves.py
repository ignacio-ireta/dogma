"""Timed re-solves over HTTP: the same use cases ``dogma resolve`` and ``dogma submit`` call.

The rules live in ``dogma.learning.resolve_service`` and ``dogma.learning.resolves``. This module
only maps them to the wire shapes, so a rule change lands in the CLI and the API at once.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from dogma.api.schemas.interview import InterviewReadyOut
from dogma.api.schemas.resolves import (
    ReadinessOut,
    ReSolveDueOut,
    ResolveStartedOut,
    ResolveWindowOut,
    TimedResultOut,
    TimedVerdictOut,
)
from dogma.core.config import Settings
from dogma.core.errors import DogmaError, ErrorCode
from dogma.db.session import database_path
from dogma.learning import resolve_service
from dogma.learning.resolve_service import Readiness
from dogma.learning.resolves import ClosedWindow, ReSolveSchedule, TimedResult, WindowRecord


def start(settings: Settings, problem_id: str, *, now: datetime) -> ResolveStartedOut:
    started = resolve_service.start(settings, problem_id, now=now)
    return ResolveStartedOut(
        window=window_out(started.window),
        previous_file_kept=started.kept is not None,
    )


def current(settings: Settings, *, now: datetime) -> ResolveWindowOut | None:
    window = resolve_service.status(settings, now=now)
    return None if window is None else window_out(window)


def abandon(settings: Settings, *, now: datetime) -> ResolveWindowOut:
    # Checked here so abandoning with no database creates none, and says what the CLI says.
    if not database_path(settings).is_file():
        raise DogmaError(ErrorCode.RESOLVE_NOT_FOUND, "No timed re-solve is open.")
    return window_out(resolve_service.abandon(settings, None, now=now))


def window_out(window: WindowRecord) -> ResolveWindowOut:
    return ResolveWindowOut(
        id=window.window_id,
        problem_id=window.problem_id,
        status=window.status.value,
        started_at=window.started_at,
        budget_s=window.budget_s,
        deadline=window.started_at + timedelta(seconds=window.budget_s),
        hints_used=window.hints_used,
        closed_at=window.closed_at,
    )


def verdict_out(closed: ClosedWindow | None, *, now: datetime) -> TimedVerdictOut | None:
    if closed is None:
        return None
    return TimedVerdictOut(
        problem_id=closed.problem_id,
        status=closed.status.value,
        used_s=closed.used_s,
        budget_s=closed.budget_s,
        hints_used=closed.hints_used,
        next_due=None if closed.next_due is None else due_out(closed.next_due, now=now),
    )


def due_out(schedule: ReSolveSchedule, *, now: datetime) -> ReSolveDueOut:
    return ReSolveDueOut(
        problem_id=schedule.problem_id,
        stage=schedule.stage,
        due_at=schedule.due_at,
        due_now=schedule.is_due(now),
        reason=schedule.reason,
    )


def readiness_out(
    view: Readiness, *, now: datetime, interview_ready: InterviewReadyOut
) -> ReadinessOut:
    return ReadinessOut(
        next_due=[due_out(item, now=now) for item in view.upcoming],
        recent_results=[timed_result_out(item) for item in view.recent],
        interview_ready=interview_ready,
    )


def timed_result_out(result: TimedResult) -> TimedResultOut:
    return TimedResultOut(
        problem_id=result.problem_id,
        status=result.status.value,
        used_s=result.used_s,
        budget_s=result.budget_s,
        closed_at=result.closed_at,
    )
