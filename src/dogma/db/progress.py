"""The read side of progress: one snapshot of the learner's state, summarised per topic.

Pure functions on the snapshot, so the rules (best result, window counts) are tested without a
database. ``read_snapshot`` is the only function here that touches SQLite.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from sqlalchemy import select

from dogma.core.config import Settings
from dogma.db.errors import StorageErrorCode
from dogma.db.models import Attempt, Mastery, TopicState, mastery_rank
from dogma.db.resolve_store import all_windows
from dogma.db.session import database_path, open_session
from dogma.learning.resolves import WindowRecord


@dataclass(frozen=True)
class AttemptRow:
    """One stored attempt, reduced to what progress shows. ``passed``/``total`` count every test."""

    problem_id: str
    mode: str
    status: str
    passed: int
    total: int
    submitted_at: datetime


@dataclass(frozen=True)
class Snapshot:
    """The learner's state at one moment. An empty snapshot is a learner who has done nothing."""

    mastery: Mapping[str, str]
    attempts: tuple[AttemptRow, ...]
    windows: tuple[WindowRecord, ...] = field(default=())


@dataclass(frozen=True)
class BestResult:
    passed: int
    total: int
    mode: str


@dataclass(frozen=True)
class TopicProgress:
    topic_id: str
    mastery: Mastery
    attempts: int
    best: BestResult | None
    last_attempt_at: datetime | None

    @property
    def solved(self) -> bool:
        return mastery_rank(self.mastery) >= mastery_rank(Mastery.SOLVED)


def read_snapshot(settings: Settings, *, now: datetime | None = None) -> Snapshot:
    """Read the learner's state. With no database file yet it is empty, and nothing is created.

    Reading windows abandons any window open for more than 12 hours, so ``now`` (default: the
    current time) is the moment that rule is judged at.
    """
    if not database_path(settings).is_file():
        return Snapshot(mastery={}, attempts=())
    when = datetime.now(UTC) if now is None else now
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        mastery = {row.topic_id: row.mastery for row in session.scalars(select(TopicState))}
        attempts = tuple(
            AttemptRow(
                problem_id=row.problem_id,
                mode=row.mode,
                status=row.status,
                passed=row.public_passed + row.hidden_passed,
                total=row.public_total + row.hidden_total,
                submitted_at=row.submitted_at,
            )
            for row in session.scalars(select(Attempt).order_by(Attempt.submitted_at, Attempt.id))
        )
        windows = all_windows(session, now=when)
    return Snapshot(mastery=mastery, attempts=attempts, windows=windows)


def first_passes(snapshot: Snapshot) -> dict[str, datetime]:
    """Each problem's first passing submit time. A problem never passed is absent."""
    first: dict[str, datetime] = {}
    for row in snapshot.attempts:  # oldest first
        if row.mode == "submit" and row.status == "passed":
            first.setdefault(row.problem_id, row.submitted_at)
    return first


def topic_progress(topic_id: str, snapshot: Snapshot) -> TopicProgress:
    """A topic's mastery, attempt count, best result and last attempt, from the snapshot."""
    mine = [row for row in snapshot.attempts if row.problem_id.startswith(f"{topic_id}/")]
    scored = [row for row in mine if row.total > 0]  # a rejected attempt ran no tests
    best = max(scored, key=_quality, default=None)
    return TopicProgress(
        topic_id=topic_id,
        mastery=Mastery(snapshot.mastery.get(topic_id, Mastery.UNSEEN.value)),
        attempts=len(mine),
        best=None if best is None else BestResult(best.passed, best.total, best.mode),
        last_attempt_at=max((row.submitted_at for row in mine), default=None),
    )


def attempts_between(snapshot: Snapshot, start: date, end: date | None) -> int:
    """Attempts submitted on a local date in ``[start, end)``. ``end=None`` means no upper bound."""
    count = 0
    for row in snapshot.attempts:
        day = row.submitted_at.astimezone().date()
        if day >= start and (end is None or day < end):
            count += 1
    return count


def _quality(row: AttemptRow) -> tuple[bool, float, datetime]:
    """Higher is better: a passing submit, then the share of tests passed, then the newest."""
    passing_submit = row.mode == "submit" and row.status == "passed"
    return (passing_submit, row.passed / row.total, row.submitted_at)
