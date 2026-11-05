"""Storage for timed re-solve windows (ARCHITECTURE §4.4, §4.9).

Every function takes the caller's session and never commits, so a window change lands in the same
transaction as the attempt, rating or start that caused it. Each one takes ``now`` explicitly. A
window open for more than 12 hours is abandoned the first time any reader asks for windows.
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from dogma.core.errors import DogmaError, ErrorCode
from dogma.db.models import Attempt, AttemptMode, Mastery, TopicState
from dogma.db.resolve_models import ResolveWindow, WindowStatus
from dogma.grading.models import RunStatus
from dogma.learning.resolves import (
    ClosedWindow,
    ReSolveSchedule,
    WindowRecord,
    build_schedules,
    is_stale,
    judge_window,
    reevaluate_mastery,
    topic_of,
)


def to_record(row: ResolveWindow) -> WindowRecord:
    return WindowRecord(
        window_id=row.id,
        problem_id=row.problem_id,
        topic_id=row.topic_id,
        started_at=row.started_at,
        budget_s=row.budget_s,
        hints_used=row.hints_used,
        status=WindowStatus(row.status),
        closed_at=row.closed_at,
        attempt_id=row.attempt_id,
    )


def expire_stale(session: Session, now: datetime) -> None:
    """Abandon the open window if it has been open for more than 12 hours."""
    row = _open_row(session)
    if row is not None and is_stale(to_record(row), now):
        row.status = WindowStatus.ABANDONED.value
        row.closed_at = now


def open_window(session: Session, *, now: datetime) -> WindowRecord | None:
    expire_stale(session, now)
    row = _open_row(session)
    return None if row is None else to_record(row)


def all_windows(session: Session, *, now: datetime) -> tuple[WindowRecord, ...]:
    """Every window, oldest start first."""
    expire_stale(session, now)
    rows = session.scalars(
        select(ResolveWindow).order_by(ResolveWindow.started_at, ResolveWindow.id)
    )
    return tuple(to_record(row) for row in rows)


def windows_for(session: Session, problem_id: str, *, now: datetime) -> tuple[WindowRecord, ...]:
    """Every window on one problem, oldest start first."""
    expire_stale(session, now)
    rows = session.scalars(
        select(ResolveWindow)
        .where(ResolveWindow.problem_id == problem_id)
        .order_by(ResolveWindow.started_at, ResolveWindow.id)
    )
    return tuple(to_record(row) for row in rows)


def first_passes(session: Session) -> dict[str, datetime]:
    """The time of each problem's first passing submit. A problem never passed is absent."""
    rows = session.execute(
        select(Attempt.problem_id, Attempt.submitted_at)
        .where(
            Attempt.mode == AttemptMode.SUBMIT.value,
            Attempt.status == RunStatus.PASSED.value,
        )
        .order_by(Attempt.submitted_at, Attempt.id)
    )
    first: dict[str, datetime] = {}
    for problem_id, submitted_at in rows:
        first.setdefault(problem_id, submitted_at)
    return first


def start_window(
    session: Session, *, problem_id: str, problem_version: int, budget_s: int, now: datetime
) -> WindowRecord:
    """Open a window. Refuses while another is open, and names that one."""
    expire_stale(session, now)
    open_row = _open_row(session)
    if open_row is not None:
        raise DogmaError(
            ErrorCode.RESOLVE_ALREADY_OPEN,
            f"A timed re-solve of {open_row.problem_id} is still open. Submit it with "
            f"dogma submit {open_row.problem_id}, or abandon it with "
            f"dogma resolve {open_row.problem_id} --abandon.",
        )
    row = ResolveWindow(
        id=uuid4().hex,
        problem_id=problem_id,
        problem_version=problem_version,
        topic_id=topic_of(problem_id),
        started_at=now,
        budget_s=budget_s,
        hints_used=0,
        status=WindowStatus.OPEN.value,
    )
    session.add(row)
    session.flush()
    return to_record(row)


def abandon_window(
    session: Session, *, now: datetime, problem_id: str | None = None
) -> WindowRecord:
    """Close the open window as abandoned. Nothing else changes: abandoning carries no penalty."""
    expire_stale(session, now)
    row = _open_row(session)
    if row is None:
        raise DogmaError(ErrorCode.RESOLVE_NOT_FOUND, "No timed re-solve is open.")
    if problem_id is not None and row.problem_id != problem_id:
        raise DogmaError(
            ErrorCode.RESOLVE_NOT_FOUND,
            f"The open timed re-solve is for {row.problem_id}, not {problem_id}.",
        )
    row.status = WindowStatus.ABANDONED.value
    row.closed_at = now
    return to_record(row)


def count_hint(session: Session, *, problem_id: str, now: datetime) -> bool:
    """Count one hint against the open window on ``problem_id``. False when there is none."""
    expire_stale(session, now)
    row = _open_row(session)
    if row is None or row.problem_id != problem_id:
        return False
    row.hints_used += 1
    return True


def close_window_for_attempt(
    session: Session,
    *,
    problem_id: str,
    attempt_id: str,
    passed: bool,
    submitted_at: datetime,
) -> ClosedWindow | None:
    """Close the open window on ``problem_id`` with the verdict of this submit, if one is open.

    The caller has just added the attempt row. The window links to it, and the topic's mastery is
    re-evaluated in the same transaction. A partial verdict never changes mastery.
    """
    session.flush()  # the window's attempt_id must point at a row that is already written
    expire_stale(session, submitted_at)
    row = _open_row(session)
    if row is None or row.problem_id != problem_id:
        return None
    elapsed = (submitted_at - row.started_at).total_seconds()
    verdict = judge_window(
        passed=passed,
        elapsed_s=elapsed,
        budget_s=row.budget_s,
        hints_used=row.hints_used,
    )
    row.status = verdict.value
    row.attempt_id = attempt_id
    row.closed_at = submitted_at
    if verdict is not WindowStatus.PARTIAL:
        state = session.get(TopicState, topic_of(problem_id))
        if state is not None:
            apply_mastery(session, state)
    return ClosedWindow(
        problem_id=problem_id,
        status=verdict,
        used_s=round(elapsed),
        budget_s=row.budget_s,
        hints_used=row.hints_used,
        next_due=_next_due(session, problem_id),
    )


def apply_mastery(session: Session, state: TopicState) -> None:
    """Re-evaluate one topic from its windows. Used after a window closes and after a card lapse."""
    windows = _topic_windows(session, state.topic_id)
    change = reevaluate_mastery(
        Mastery(state.mastery),
        solved_at=state.solved_at,
        retained_at=state.retained_at,
        windows=windows,
    )
    state.mastery = change.mastery.value
    state.retained_at = change.retained_at


def _next_due(session: Session, problem_id: str) -> ReSolveSchedule | None:
    """The problem's schedule after the window that just closed, or None if it was never passed."""
    passed_at = first_passes(session).get(problem_id)
    if passed_at is None:
        return None
    windows = [
        window
        for window in _topic_windows(session, topic_of(problem_id))
        if window.problem_id == problem_id
    ]
    return build_schedules({problem_id: passed_at}, windows)[problem_id]


def _topic_windows(session: Session, topic_id: str) -> list[WindowRecord]:
    rows = session.scalars(select(ResolveWindow).where(ResolveWindow.topic_id == topic_id))
    return [to_record(row) for row in rows]


def _open_row(session: Session) -> ResolveWindow | None:
    """The one open window, if any. A partial unique index keeps this to at most one row."""
    return session.scalars(
        select(ResolveWindow).where(ResolveWindow.status == WindowStatus.OPEN.value)
    ).first()
