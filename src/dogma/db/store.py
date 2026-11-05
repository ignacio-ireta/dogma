"""Writes: one row per graded attempt, and the mastery a passing core submit earns (§4.4).

``solved`` is set here, and a submit closes the open timed re-solve window on its problem, in the
same transaction. Mastery never moves backwards, so a later failure cannot undo a solved topic.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from dogma.core.config import Settings
from dogma.db.errors import StorageErrorCode
from dogma.db.models import Attempt, AttemptMode, Mastery, TopicState, mastery_rank
from dogma.db.resolve_store import close_window_for_attempt
from dogma.db.session import open_session
from dogma.grading.models import GradeReport, RunStatus
from dogma.learning.resolves import ClosedWindow


def record_attempt(
    settings: Settings,
    report: GradeReport,
    *,
    mode: AttemptMode,
    core: bool,
    started_at: datetime,
    submitted_at: datetime,
) -> ClosedWindow | None:
    """Store one graded attempt. A passing submit of a core problem marks its topic solved.

    A submit (never a run) also closes the open timed re-solve on its problem. The verdict comes
    back so the caller can say what it was; None when no window was open on that problem.
    """
    public_passed, public_total = report.counts("public")
    hidden_passed, hidden_total = report.counts("hidden")
    closed: ClosedWindow | None = None
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        session.add(
            Attempt(
                id=report.attempt_id,
                problem_id=report.problem_id,
                problem_version=report.problem_version,
                mode=mode.value,
                status=report.status.value,
                public_passed=public_passed,
                public_total=public_total,
                hidden_passed=hidden_passed,
                hidden_total=hidden_total,
                hints_used=0,
                run_dir=_run_dir(settings, report),
                started_at=started_at,
                submitted_at=submitted_at,
                duration_ms=_elapsed_ms(started_at, submitted_at),
            )
        )
        if core and mode is AttemptMode.SUBMIT and report.passed:
            mark_solved(session, topic_of(report.problem_id), submitted_at)
        if mode is AttemptMode.SUBMIT:
            closed = close_window_for_attempt(
                session,
                problem_id=report.problem_id,
                attempt_id=report.attempt_id,
                passed=report.passed,
                submitted_at=submitted_at,
            )
    return closed


def mark_solved(session: Session, topic_id: str, when: datetime) -> None:
    """Move a topic to ``solved`` unless it is already there or further along."""
    state = session.get(TopicState, topic_id)
    if state is None:
        session.add(TopicState(topic_id=topic_id, mastery=Mastery.SOLVED.value, solved_at=when))
    elif mastery_rank(state.mastery) < mastery_rank(Mastery.SOLVED):
        state.mastery = Mastery.SOLVED.value
        state.solved_at = when


def topic_of(problem_id: str) -> str:
    """``track/topic/pNN-slug`` → ``track/topic``."""
    return problem_id.rsplit("/", 1)[0]


def _run_dir(settings: Settings, report: GradeReport) -> str | None:
    """Where the run's files went, relative to the repo. A rejected attempt never ran."""
    if report.status is RunStatus.REJECTED:
        return None
    path = settings.runs_dir / report.attempt_id
    try:
        return path.relative_to(settings.repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _elapsed_ms(started_at: datetime, submitted_at: datetime) -> int:
    return max(0, round((submitted_at - started_at).total_seconds() * 1000))
