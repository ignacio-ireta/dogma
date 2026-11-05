"""Timed re-solves as pure rules: the verdict, the derived schedule, mastery and readiness.

Nothing here reads a clock, a file or the database. Storage (``dogma.db.resolve_store``) turns
rows into ``WindowRecord`` values, calls these functions inside its own transaction, and writes
the result back. Every rule is table-driven in ``tests/unit/learning/test_resolves.py``.

The rules, in short (ARCHITECTURE §4.4):
- The first recorded submit of a window's problem decides it: all tests passed, within the budget,
  with no hints is ``passed``; all passed but late or hinted is ``partial``; anything else is
  ``failed``.
- The schedule is derived, never stored. The stage is the number of ``passed`` windows, capped at
  3. The interval after a stage is 7, 21, 60, then 120 days (maintenance). After a ``partial`` or
  ``failed`` window, the problem is due again 3 days after that window closed, at the same stage.
  ``abandoned`` and open windows do not move the schedule.
- Mastery moves ``consolidating`` to ``retained`` on a ``passed`` window at least 7 days after the
  topic was solved, with no ``failed`` window after it. A ``failed`` window drops ``retained`` or
  ``interview-ready`` to ``consolidating``, never lower.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final

from dogma.db.models import Mastery, mastery_rank
from dogma.db.resolve_models import CLOSED_VERDICTS, WindowStatus

INTERVALS_DAYS: Final = (7, 21, 60, 120)
MAX_STAGE: Final = len(INTERVALS_DAYS) - 1
RETRY_DAYS: Final = 3
RETAIN_AFTER: Final = timedelta(days=7)
OPEN_LIMIT: Final = timedelta(hours=12)
RECENT_LIMIT: Final = 5
UPCOMING_LIMIT: Final = 3


@dataclass(frozen=True)
class WindowRecord:
    """One timed re-solve, detached from the database. ``budget_s`` is seconds."""

    window_id: str
    problem_id: str
    topic_id: str
    started_at: datetime
    budget_s: int
    hints_used: int
    status: WindowStatus
    closed_at: datetime | None
    attempt_id: str | None


_Ended = tuple[datetime, WindowRecord]  # a closed verdict and the time it closed


def _in_close_order(pair: _Ended) -> tuple[datetime, str]:
    return pair[0], pair[1].window_id


@dataclass(frozen=True)
class ReSolveSchedule:
    """When a core problem's next timed re-solve is due, and the one-line reason."""

    problem_id: str
    topic_id: str
    stage: int
    due_at: datetime
    reason: str

    def is_due(self, now: datetime) -> bool:
        return self.due_at <= now


@dataclass(frozen=True)
class ClosedWindow:
    """What closing a window reports back to the caller that submitted."""

    problem_id: str
    status: WindowStatus
    used_s: int
    budget_s: int
    hints_used: int
    next_due: ReSolveSchedule | None


@dataclass(frozen=True)
class TimedResult:
    """A closed window as progress shows it: the verdict and the time used against the budget."""

    problem_id: str
    status: WindowStatus
    used_s: int
    budget_s: int
    closed_at: datetime


@dataclass(frozen=True)
class MasteryChange:
    mastery: Mastery
    retained_at: datetime | None


def judge_window(*, passed: bool, elapsed_s: float, budget_s: int, hints_used: int) -> WindowStatus:
    """The verdict of the decisive submit. ``elapsed_s`` runs from the window's start."""
    if not passed:
        return WindowStatus.FAILED
    if elapsed_s > budget_s or hints_used > 0:
        return WindowStatus.PARTIAL
    return WindowStatus.PASSED


def is_stale(window: WindowRecord, now: datetime) -> bool:
    """An open window older than 12 hours is abandoned the next time windows are read."""
    return window.status is WindowStatus.OPEN and now - window.started_at > OPEN_LIMIT


def topic_of(problem_id: str) -> str:
    """``track/topic/pNN-slug`` → ``track/topic``."""
    return problem_id.rsplit("/", 1)[0]


def build_schedules(
    first_passes: Mapping[str, datetime], windows: Iterable[WindowRecord]
) -> dict[str, ReSolveSchedule]:
    """The schedule of every problem with a passing submit. ``first_passes`` maps id to its time."""
    by_problem: dict[str, list[_Ended]] = {}
    for window in windows:
        if window.status in CLOSED_VERDICTS and window.closed_at is not None:
            by_problem.setdefault(window.problem_id, []).append((window.closed_at, window))
    return {
        problem_id: _schedule(
            problem_id, first, sorted(by_problem.get(problem_id, []), key=_in_close_order)
        )
        for problem_id, first in first_passes.items()
    }


def _schedule(problem_id: str, first_pass: datetime, ended: list[_Ended]) -> ReSolveSchedule:
    """``ended`` holds each closed verdict with its close time, oldest first."""
    stage = min(sum(1 for _, w in ended if w.status is WindowStatus.PASSED), MAX_STAGE)
    topic_id = topic_of(problem_id)
    if ended and ended[-1][1].status is not WindowStatus.PASSED:
        closed_at, last = ended[-1]
        due = closed_at + timedelta(days=RETRY_DAYS)
        reason = (
            f"The last timed re-solve was {last.status.value}: due again "
            f"{RETRY_DAYS} days after it closed, at the same stage."
        )
        return ReSolveSchedule(problem_id, topic_id, stage, due, reason)
    anchor = ended[-1][0] if ended else first_pass
    days = INTERVALS_DAYS[stage]
    return ReSolveSchedule(
        problem_id, topic_id, stage, anchor + timedelta(days=days), _stage_reason(stage, days)
    )


_ORDINALS: Final = ("First", "Second", "Third")


def _stage_reason(stage: int, days: int) -> str:
    if stage == 0:
        return f"First timed re-solve: {days} days after your first pass."
    if stage == MAX_STAGE:
        return f"Maintenance: every {days} days after your last timed pass."
    return f"{_ORDINALS[stage]} timed re-solve: {days} days after your last timed pass."


def reevaluate_mastery(
    current: Mastery,
    *,
    solved_at: datetime | None,
    retained_at: datetime | None,
    windows: Iterable[WindowRecord],
) -> MasteryChange:
    """The topic's mastery after its windows, given its current state.

    Only ``consolidating``, ``retained`` and ``interview-ready`` can change. A topic still at
    ``solved`` waits for a card review to move it to ``consolidating``, then this runs again.
    """
    if mastery_rank(current) < mastery_rank(Mastery.CONSOLIDATING):
        return MasteryChange(current, retained_at)
    closed = [(w.closed_at, w) for w in windows if w.closed_at is not None]
    fails = [at for at, w in closed if w.status is WindowStatus.FAILED]
    last_fail = max(fails, default=None)
    mastery, kept = current, retained_at
    if (
        mastery in (Mastery.RETAINED, Mastery.INTERVIEW_READY)
        and last_fail is not None
        and (kept is None or last_fail > kept)
    ):
        mastery = Mastery.CONSOLIDATING
    if mastery is Mastery.CONSOLIDATING and solved_at is not None:
        qualifying = [
            at
            for at, w in closed
            if w.status is WindowStatus.PASSED
            and at - solved_at >= RETAIN_AFTER
            and (last_fail is None or at > last_fail)
        ]
        if qualifying:
            mastery, kept = Mastery.RETAINED, min(qualifying)
    return MasteryChange(mastery, kept)


def timed_results(
    windows: Iterable[WindowRecord], limit: int = RECENT_LIMIT
) -> tuple[TimedResult, ...]:
    """The newest closed verdicts first. Abandoned windows are not results."""
    ended = sorted(
        (
            (w.closed_at, w)
            for w in windows
            if w.status in CLOSED_VERDICTS and w.closed_at is not None
        ),
        key=_in_close_order,
        reverse=True,
    )
    return tuple(
        TimedResult(
            problem_id=w.problem_id,
            status=w.status,
            used_s=max(0, round((closed_at - w.started_at).total_seconds())),
            budget_s=w.budget_s,
            closed_at=closed_at,
        )
        for closed_at, w in ended[:limit]
    )


def upcoming(
    schedules: Mapping[str, ReSolveSchedule], limit: int = UPCOMING_LIMIT
) -> tuple[ReSolveSchedule, ...]:
    """The soonest due re-solves, soonest first. Overdue and due-now items are not counted apart."""
    ordered = sorted(schedules.values(), key=lambda s: (s.due_at, s.problem_id))
    return tuple(ordered[:limit])
