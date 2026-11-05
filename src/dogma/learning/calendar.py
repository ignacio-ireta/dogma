"""The personal course calendar: a whole-week offset on the authored core path (ADR-0019, I-30a).

``curriculum/paths/core.yaml`` is never rewritten. The learner's shift is one integer, the newest
row of ``calendar_warp``, and every date-dependent reader loads the shifted path through
``load_calendar``. Card and timed re-solve due dates do not move: they measure memory over real
time. Nothing here says anything about being behind; the words are "move the calendar" and "this
week becomes week N".

``apply`` and ``reset_course`` are the only writers. Both refuse when recording is off, before
anything is touched. ``preview`` and every reader leave the database as they found it.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Final

from sqlalchemy import select
from sqlalchemy.orm import Session

from dogma.content.loader import load_core_path
from dogma.content.schemas import CorePath, PathWeek
from dogma.core.config import Settings
from dogma.core.errors import ContentError, DogmaError, ErrorCode
from dogma.db import (  # noqa: F401  registers every table on Base.metadata before a reset
    answer_models,
    calendar_models,
    mock_models,
    models,
    resolve_models,
    review_models,
    session_models,
)
from dogma.db.backup import write_backup
from dogma.db.base import Base
from dogma.db.calendar_models import CalendarWarp
from dogma.db.errors import StorageErrorCode
from dogma.db.models import Attempt
from dogma.db.session import database_path, open_session
from dogma.db.store import topic_of
from dogma.learning.path import current_week

MOCKS_FROM_WEEK: Final = 3  # the first week with weekly mock interviews (ARCHITECTURE §4.4)
FIRST_WEEK: Final = 1


class WarpKind(StrEnum):
    SYNC = "sync"
    TO_WEEK = "to_week"
    AHEAD = "ahead"
    BACK = "back"
    ORIGINAL = "original"
    COURSE_RESET = "course_reset"  # written only by reset_course, never requested


_STEP_KINDS: Final = frozenset({WarpKind.AHEAD, WarpKind.BACK})


@dataclass(frozen=True)
class WarpRequest:
    """What the learner asked for. ``week`` goes with to_week, ``weeks`` with ahead and back."""

    kind: WarpKind
    week: int | None = None
    weeks: int | None = None

    def __post_init__(self) -> None:
        if self.kind is WarpKind.TO_WEEK:
            valid = self.week is not None and self.week >= 1 and self.weeks is None
        elif self.kind in _STEP_KINDS:
            valid = self.weeks is not None and self.weeks >= 1 and self.week is None
        elif self.kind is WarpKind.SYNC or self.kind is WarpKind.ORIGINAL:
            valid = self.week is None and self.weeks is None
        else:
            valid = False
        if not valid:
            raise DogmaError(
                ErrorCode.CALENDAR_REQUEST_INVALID,
                f"a {self.kind.value} move needs its own amount: a week number from 1, "
                "or a number of weeks from 1",
            )


@dataclass(frozen=True)
class Anchor:
    """The furthest core-path week the learner has run an exercise in, and the problem in it."""

    week: int
    problem_id: str | None  # None when nothing has been run yet


@dataclass(frozen=True)
class LastWarp:
    kind: WarpKind
    created_at: datetime
    offset_weeks: int


@dataclass(frozen=True)
class CalendarStatus:
    offset_weeks: int
    current_week: PathWeek
    anchor: Anchor
    start: date
    end: date
    last_warp: LastWarp | None


@dataclass(frozen=True)
class CalendarPoint:
    """The calendar at one offset: the current week, and the dates the learner sees."""

    offset_weeks: int
    current_week: int
    current_starts: date
    mocks_from: date  # the start of week 3, the first weekly mock
    interview_week: date  # the start of the last week
    end: date


@dataclass(frozen=True)
class WarpPreview:
    request: WarpRequest
    changed: bool
    written: bool
    before: CalendarPoint
    after: CalendarPoint
    anchor: Anchor | None  # set for a sync, to say what set the target


@dataclass(frozen=True)
class ResetResult:
    backup_path: Path
    snapshot_path: Path | None
    rows_cleared: Mapping[str, int]
    history_dir: Path | None
    files_set_aside: int
    workspace_error: str | None
    offset_weeks: int
    week_one_starts: date
    course_end: date


# --- the shift itself (pure) ------------------------------------------------------------------


def shift_path(core: CorePath, offset_weeks: int) -> CorePath:
    """Move every date of the path by ``offset_weeks`` whole weeks (7 days each).

    Weekdays never change, so the weeks stay Monday-based. Zero returns the path unchanged.
    """
    if offset_weeks == 0:
        return core
    delta = timedelta(weeks=offset_weeks)
    return core.model_copy(
        update={
            "start": core.start + delta,
            "end": core.end + delta,
            "weeks": tuple(
                item.model_copy(update={"starts": item.starts + delta}) for item in core.weeks
            ),
        }
    )


def offset_for_week(core: CorePath, week: int, today: date) -> int:
    """The offset that makes ``today`` fall in ``week``. ``week`` is clamped to the path's weeks.

    ``core`` is the authored path. With offset k, week N starts ``authored_starts(N) + 7k`` days,
    so k is the whole number of weeks from the authored start to today (floor division, so a day
    earlier than the authored start rounds toward the earlier week).
    """
    ordered = _ordered(core)
    number = min(max(week, FIRST_WEEK), len(ordered))
    return (today - ordered[number - 1].starts).days // 7


def offset_note(offset_weeks: int) -> str | None:
    """The line ``dogma path`` shows while the calendar is moved, or None at the authored dates."""
    if offset_weeks == 0:
        return None
    count = abs(offset_weeks)
    unit = "week" if count == 1 else "weeks"
    direction = "later" if offset_weeks > 0 else "earlier"
    return (
        f"Dates moved {count} {unit} {direction} by time-warp. "
        "`dogma warp --original` restores them."
    )


def furthest_anchor(problem_ids: Sequence[str], weeks_of: Mapping[str, int]) -> Anchor:
    """The furthest core-path week among ``problem_ids`` (oldest first), or week 1 with none.

    Problems whose topic is not on the path are ignored. On a tie the later attempt wins, so the
    problem shown is the one the learner most recently ran in that week.
    """
    best: Anchor | None = None
    for problem_id in problem_ids:
        week = weeks_of.get(topic_of(problem_id))
        if week is not None and (best is None or week >= best.week):
            best = Anchor(week=week, problem_id=problem_id)
    return best if best is not None else Anchor(week=FIRST_WEEK, problem_id=None)


# --- readers (never write) --------------------------------------------------------------------


def load_calendar(settings: Settings) -> CorePath:
    """The authored core path moved by the learner's current offset. Every date reader uses this."""
    offset, _ = _read_log(settings)
    return shift_path(_authored(settings), offset)


def current_offset(settings: Settings) -> int:
    """The offset in force: the newest warp's value, or 0 with no warps or no database yet."""
    offset, _ = _read_log(settings)
    return offset


def anchor(settings: Settings, core: CorePath) -> Anchor:
    """The furthest path week among every recorded attempt, run or submit. Week 1 with none."""
    if not database_path(settings).is_file():
        return Anchor(week=FIRST_WEEK, problem_id=None)
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as db:
        problem_ids = list(
            db.scalars(select(Attempt.problem_id).order_by(Attempt.submitted_at, Attempt.id))
        )
    return furthest_anchor(problem_ids, _topic_weeks(core))


def status(settings: Settings, *, now: datetime) -> CalendarStatus:
    """Where the calendar stands now: offset, current week, anchor, course dates and last move."""
    authored = _authored(settings)
    offset, last = _read_log(settings)
    shifted = shift_path(authored, offset)
    return CalendarStatus(
        offset_weeks=offset,
        current_week=current_week(shifted.weeks, _today(now)),
        anchor=anchor(settings, authored),
        start=shifted.start,
        end=shifted.end,
        last_warp=last,
    )


def preview(settings: Settings, request: WarpRequest, *, now: datetime) -> WarpPreview:
    """What a move would do, before and after. Never writes, even when recording is off."""
    authored = _authored(settings)
    offset, _ = _read_log(settings)
    today = _today(now)
    current = shift_path(authored, offset)
    found = anchor(settings, authored) if request.kind is WarpKind.SYNC else None
    target = _target_offset(authored, current, request, found, today)
    return WarpPreview(
        request=request,
        changed=target != offset,
        written=False,
        before=_point(current, offset, today),
        after=_point(shift_path(authored, target), target, today),
        anchor=found,
    )


def require_recording(settings: Settings) -> None:
    """Refuse a write when attempts are not recorded (DOGMA_RECORD_ATTEMPTS=false)."""
    if not settings.record_attempts:
        raise DogmaError(
            ErrorCode.CALENDAR_RECORDING_OFF,
            "Recording is switched off (DOGMA_RECORD_ATTEMPTS=false), so nothing was moved.",
        )


# --- writers ----------------------------------------------------------------------------------


def apply(settings: Settings, request: WarpRequest, *, now: datetime) -> WarpPreview:
    """Move the calendar. Writes one ``calendar_warp`` row only when the offset changes."""
    require_recording(settings)
    result = preview(settings, request, now=now)
    if not result.changed:
        return result
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as db:
        db.add(
            CalendarWarp(
                created_at=_checked(now),
                kind=request.kind.value,
                offset_weeks=result.after.offset_weeks,
                previous_offset_weeks=result.before.offset_weeks,
                week_before=result.before.current_week,
                week_after=result.after.current_week,
            )
        )
    return replace(result, written=True)


def reset_course(settings: Settings, *, now: datetime) -> ResetResult:
    """Back up, clear every progress table, set workspace files aside, and make this week week 1.

    The backup comes first: if it fails, the error propagates and nothing is cleared. The clear
    and the new warp row share one transaction. Moving the workspace files happens after the
    commit, and a failure there is reported in the result rather than raised.
    """
    require_recording(settings)
    when = _checked(now)
    backup = write_backup(settings, now=when)
    authored = _authored(settings)
    today = _today(when)
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as db:
        before, _ = _read_log_in(db)
        cleared = _clear_progress(db)
        after = offset_for_week(authored, FIRST_WEEK, today)
        db.add(
            CalendarWarp(
                created_at=when,
                kind=WarpKind.COURSE_RESET.value,
                offset_weeks=after,
                previous_offset_weeks=before,
                week_before=current_week(shift_path(authored, before).weeks, today).week,
                week_after=current_week(shift_path(authored, after).weeks, today).week,
            )
        )
    shifted = shift_path(authored, after)
    history, moved, problem = _set_aside_workspace(settings, when)
    return ResetResult(
        backup_path=backup.path,
        snapshot_path=backup.snapshot_path,
        rows_cleared=cleared,
        history_dir=history,
        files_set_aside=moved,
        workspace_error=problem,
        offset_weeks=after,
        week_one_starts=_week_starts(shifted, FIRST_WEEK),
        course_end=shifted.end,
    )


# --- helpers ----------------------------------------------------------------------------------


def _authored(settings: Settings) -> CorePath:
    return load_core_path(settings.curriculum_dir / "paths" / "core.yaml")


def _ordered(core: CorePath) -> tuple[PathWeek, ...]:
    if not core.weeks:
        raise ContentError(ErrorCode.CONTENT_SCHEMA_INVALID, "the core path has no weeks")
    return tuple(sorted(core.weeks, key=lambda item: item.week))


def _topic_weeks(core: CorePath) -> dict[str, int]:
    """Each topic on the path mapped to the first week that lists it."""
    weeks_of: dict[str, int] = {}
    for item in core.weeks:
        for topic_id in item.topics:
            weeks_of[topic_id] = min(item.week, weeks_of.get(topic_id, item.week))
    return weeks_of


def _read_log(settings: Settings) -> tuple[int, LastWarp | None]:
    """The offset in force and the newest move. Reads nothing and creates nothing without a DB."""
    if not database_path(settings).is_file():
        return 0, None
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as db:
        return _read_log_in(db)


def _read_log_in(db: Session) -> tuple[int, LastWarp | None]:
    row = db.scalars(select(CalendarWarp).order_by(CalendarWarp.id.desc()).limit(1)).first()
    if row is None:
        return 0, None
    return row.offset_weeks, LastWarp(
        kind=WarpKind(row.kind), created_at=row.created_at, offset_weeks=row.offset_weeks
    )


def _target_offset(
    authored: CorePath,
    current: CorePath,
    request: WarpRequest,
    found: Anchor | None,
    today: date,
) -> int:
    if request.kind is WarpKind.ORIGINAL:
        return 0
    if request.kind is WarpKind.TO_WEEK:
        return offset_for_week(authored, request.week or FIRST_WEEK, today)
    if request.kind is WarpKind.SYNC:
        return offset_for_week(authored, found.week if found is not None else FIRST_WEEK, today)
    here = current_week(current.weeks, today).week
    step = request.weeks or 0
    if request.kind is WarpKind.BACK:
        step = -step
    target = min(max(here + step, FIRST_WEEK), len(authored.weeks))
    return offset_for_week(authored, target, today)


def _point(path: CorePath, offset: int, today: date) -> CalendarPoint:
    here = current_week(path.weeks, today)
    return CalendarPoint(
        offset_weeks=offset,
        current_week=here.week,
        current_starts=here.starts,
        mocks_from=_week_starts(path, MOCKS_FROM_WEEK),
        interview_week=_ordered(path)[-1].starts,
        end=path.end,
    )


def _week_starts(path: CorePath, number: int) -> date:
    """The start of week ``number``, or the course start when the path is too short to have it."""
    for item in path.weeks:
        if item.week == number:
            return item.starts
    return path.start


def _clear_progress(db: Session) -> dict[str, int]:
    """Delete every row of every table except the calendar log, children before parents."""
    connection = db.connection()
    cleared: dict[str, int] = {}
    for table in reversed(Base.metadata.sorted_tables):
        if table.name == CalendarWarp.__tablename__:
            continue
        cleared[table.name] = connection.execute(table.delete()).rowcount
    return cleared


def _set_aside_workspace(settings: Settings, when: datetime) -> tuple[Path | None, int, str | None]:
    """Move every workspace file to ``var/workspace-history/reset-<stamp>/``, same relative paths.

    Returns the folder (None when nothing was there), the number of files moved, and a reason
    when a move failed. Files moved before the failure stay moved.
    """
    root = settings.workspace_dir
    if not root.is_dir():
        return None, 0, None
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        return None, 0, None
    stamp = when.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    history = settings.var_dir / "workspace-history" / f"reset-{stamp}"
    moved = 0
    try:
        for source in files:
            target = history / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            moved += 1
    except OSError as exc:
        return history, moved, exc.strerror or str(exc)
    _prune_empty_folders(root)
    return history, moved, None


def _prune_empty_folders(root: Path) -> None:
    """Remove folders left empty under ``root``, deepest first. ``root`` itself stays."""
    folders = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for folder in folders:
        try:
            folder.rmdir()
        except OSError:
            continue  # not empty, or in use: leave it


def _checked(now: datetime) -> datetime:
    if now.tzinfo is None or now.utcoffset() is None:
        raise DogmaError(ErrorCode.SESSION_TIME_INVALID, "Session times must be timezone-aware.")
    return now.astimezone(UTC)


def _today(now: datetime) -> date:
    """The learner's local calendar date, as ``sessions.local_day`` reads it."""
    return _checked(now).astimezone().date()
