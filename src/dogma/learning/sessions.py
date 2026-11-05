"""Plan today's session, resume it, and close it with a transition note (ARCHITECTURE §4.4).

``today`` and ``close_session`` are the only writers. A local date has at most ``sessions_per_day``
sessions, one per slot. Running ``today`` again resumes the open session, so a repeat never makes a
second plan. Closing stores where the learner stopped, the next step, an optional blocker and a
self-written answer. A closed session never changes again.

Closing a Learn session marks its topic ``introduced`` when the topic was unseen and enrolls the
topic's cards in the same transaction. Nothing here marks a topic solved: only a graded submit does
(``dogma.db.store``). The planner itself is pure (``dogma.learning.planner``).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from typing import Final
from uuid import uuid4

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from dogma.content.loader import load_topic
from dogma.content.schemas import CorePath
from dogma.core.config import Settings
from dogma.core.errors import ContentError, DogmaError, ErrorCode
from dogma.db.errors import StorageErrorCode
from dogma.db.models import AttemptMode, Mastery, TopicState
from dogma.db.progress import Snapshot, first_passes, read_snapshot
from dogma.db.resolve_models import WindowStatus
from dogma.db.session import database_path, open_session
from dogma.db.session_models import SessionBlock, StudySession
from dogma.grading.models import RunStatus
from dogma.learning.calendar import load_calendar
from dogma.learning.cards import due_cards, sync_cards, sync_cards_in_session
from dogma.learning.mocks import recent_weak_part
from dogma.learning.planner import (
    READY_STATUSES,
    Block,
    Energy,
    FamiliarProblem,
    PlanFacts,
    ProblemFacts,
    ResolveDue,
    SessionKind,
    SessionPlan,
    TopicFacts,
    plan_session,
)
from dogma.learning.resolves import build_schedules

DUE_SCAN_LIMIT: Final = 500
NOTE_LIMIT: Final = 500
ANSWER_LIMIT: Final = 2000
_OPEN: Final = "open"
_CLOSED: Final = "closed"


@dataclass(frozen=True)
class CloseNote:
    """What the learner writes when a session stops: where, what next, and any blocker."""

    stopped_where: str
    next_step: str
    blocker: str | None = None


@dataclass(frozen=True)
class TransitionNote:
    """A closed session's note, shown first the next time the learner starts a session."""

    session_id: str
    plan_date: date
    slot: int
    kind: SessionKind
    topic_id: str | None
    stopped_where: str
    next_step: str
    blocker: str | None
    closed_at: datetime | None


@dataclass(frozen=True)
class StoredSession:
    """The session the learner works through now, detached from the database."""

    session_id: str
    plan_date: date
    slot: int
    sessions_per_day: int
    plan: SessionPlan


@dataclass(frozen=True)
class TodayView:
    local_date: date
    sessions_per_day: int
    closed_today: int
    last_note: TransitionNote | None
    current: StoredSession | None  # None once every session of the day is closed
    resumed: bool
    replanned: bool
    stale_open: tuple[date, ...]  # earlier sessions that were never closed


@dataclass(frozen=True)
class CloseTarget:
    """The latest open session, or the last completed session for an idempotent retry."""

    session_id: str
    plan_date: date
    slot: int
    kind: SessionKind
    topic_id: str | None
    stored: TransitionNote | None  # set when the session is already closed


@dataclass(frozen=True)
class CloseResult:
    note: TransitionNote
    already_closed: bool
    introduced_topic: str | None
    new_cards: int


def local_day(now: datetime) -> date:
    """The learner's local calendar date for an aware instant."""
    return _aware(now).astimezone().date()


def today(settings: Settings, *, now: datetime, energy: Energy | None = None) -> TodayView:
    """Resume an open session, even across midnight, or plan today's next one.

    Without ``energy`` an open session resumes exactly as planned. With a different ``energy`` the
    open session is re-planned in place, which is safe: nothing is recorded until it is closed.
    """
    when = _aware(now)
    day = local_day(when)
    core = _load_core(settings)
    sync_cards(settings, now=when)
    facts = _gather(settings, core, when)
    slots = core.sessions_per_day
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as db:
        todays = list(
            db.scalars(
                select(StudySession)
                .where(StudySession.plan_date == day.isoformat())
                .order_by(StudySession.slot)
            )
        )
        last = _last_note(db)
        stale = _stale_dates(db, day)
        closed = sum(1 for row in todays if row.status == _CLOSED)
        opened = _latest_open(db)
        if opened is not None:
            requested = Energy(opened.energy) if energy is None else energy
            if requested.value == opened.energy:
                plan = _load_plan(db, opened)
                stored = _stored(opened, plan, slots)
                return _view(
                    day=day,
                    slots=slots,
                    closed=closed,
                    last=last,
                    stale=stale,
                    current=stored,
                    resumed=True,
                    replanned=False,
                )
            # The open session is re-planned, so it no longer counts as this week's interview.
            counted = facts.interviews_this_week
            if opened.kind == SessionKind.INTERVIEW.value:
                counted = max(0, counted - 1)
            replanned = plan_session(
                replace(facts, interviews_this_week=counted),
                slot=opened.slot,
                sessions_per_day=slots,
                energy=requested,
                exclude=_topics_of(
                    db.scalars(
                        select(StudySession).where(StudySession.plan_date == opened.plan_date)
                    ),
                    skip=opened,
                ),
            )
            _write_plan(db, opened, replanned)
            stored = _stored(opened, replanned, slots)
            return _view(
                day=day,
                slots=slots,
                closed=closed,
                last=last,
                stale=stale,
                current=stored,
                resumed=False,
                replanned=True,
            )
        if len(todays) >= slots:
            return _view(
                day=day,
                slots=slots,
                closed=closed,
                last=last,
                stale=stale,
                current=None,
                resumed=False,
                replanned=False,
            )
        slot = len(todays) + 1
        plan = plan_session(
            facts,
            slot=slot,
            sessions_per_day=slots,
            energy=energy or Energy.NORMAL,
            exclude=_topics_of(todays, skip=None),
        )
        row = _new_row(day, slot, plan, when)
        db.add(row)
        db.flush()  # the blocks reference this row; no relationship orders the inserts
        _add_blocks(db, row.id, plan)
        return _view(
            day=day,
            slots=slots,
            closed=closed,
            last=last,
            stale=stale,
            current=_stored(row, plan, slots),
            resumed=False,
            replanned=False,
        )


def close_target(settings: Settings, *, now: datetime) -> CloseTarget:
    """Find the session ``dogma close`` should act on, without writing anything."""
    day = local_day(now)
    if not database_path(settings).is_file():
        raise DogmaError(ErrorCode.SESSION_NOT_FOUND, _no_session_message())
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as db:
        todays = list(
            db.scalars(
                select(StudySession)
                .where(StudySession.plan_date == day.isoformat())
                .order_by(StudySession.slot.desc())
            )
        )
        row = _latest_open(db)
        if row is None and todays:
            row = todays[0]  # the latest closed session of the day
        if row is None:
            row = db.scalars(
                select(StudySession)
                .where(StudySession.status == _CLOSED)
                .order_by(StudySession.closed_at.desc(), StudySession.id.desc())
                .limit(1)
            ).first()
        if row is None:
            raise DogmaError(ErrorCode.SESSION_NOT_FOUND, _no_session_message())
        stored = None if row.status == _OPEN else _note(row)
        return CloseTarget(
            session_id=row.id,
            plan_date=date.fromisoformat(row.plan_date),
            slot=row.slot,
            kind=SessionKind(row.kind),
            topic_id=row.topic_id,
            stored=stored,
        )


def close_session(
    settings: Settings,
    session_id: str,
    note: CloseNote,
    *,
    answer: str | None,
    now: datetime,
) -> CloseResult:
    """Close a session once. A second close returns the stored note and changes nothing.

    The close, topic introduction and card enrollment share one transaction, so a
    failed write leaves the session open and safe to retry.
    """
    when = _aware(now)
    stopped, next_step, blocker = _note_fields(note)
    answer_text = _optional(answer, "Your answer", ANSWER_LIMIT)
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as db:
        row = db.get(StudySession, session_id)
        if row is None:
            raise DogmaError(ErrorCode.SESSION_NOT_FOUND, "That session does not exist.")
        if row.status == _CLOSED:
            return CloseResult(
                note=_note(row), already_closed=True, introduced_topic=None, new_cards=0
            )
        row.status = _CLOSED
        row.closed_at = when
        row.stopped_where = stopped
        row.next_step = next_step
        row.blocker = blocker
        row.self_answer = answer_text
        introduced: str | None = None
        if row.kind == SessionKind.LEARN and row.topic_id is not None:
            introduced = row.topic_id if _introduce(db, row.topic_id, when) else None
        new_cards = sync_cards_in_session(settings, db, now=when)
        result_note = _note(row)
    return CloseResult(
        note=result_note,
        already_closed=False,
        introduced_topic=introduced,
        new_cards=new_cards,
    )


# --- gathering facts ---------------------------------------------------------------------------


def _gather(settings: Settings, core: CorePath, when: datetime) -> PlanFacts:
    topics = tuple(_topic_facts(settings, topic_id) for topic_id in _path_topic_ids(core))
    snapshot = read_snapshot(settings, now=when)
    mastery = {topic_id: Mastery(value) for topic_id, value in snapshot.mastery.items()}
    due = len(due_cards(settings, now=when, limit=DUE_SCAN_LIMIT))
    week = _path_week(core, local_day(when))
    weak = recent_weak_part(settings)
    return PlanFacts(
        topics=topics,
        mastery=mastery,
        due_count=due,
        familiar=_familiar(snapshot, topics),
        resolves=_resolves(snapshot, topics, when),
        path_week=None if week is None else week[0],
        interviews_this_week=0 if week is None else _interviews_between(settings, week[1], week[2]),
        weak_part=None if weak is None else weak.title,
        weak_part_percent=None if weak is None else round(weak.percent * 100),
    )


def _path_week(core: CorePath, day: date) -> tuple[int, date, date] | None:
    """The core-path week holding ``day`` as (week number, first day, day after the week).

    A week runs from its ``starts`` date to the next week's start. The last week runs seven days.
    """
    ordered = sorted(core.weeks, key=lambda item: item.starts)
    for index, item in enumerate(ordered):
        if index + 1 < len(ordered):
            following = ordered[index + 1].starts
        else:
            following = item.starts + timedelta(days=7)
        if item.starts <= day < following:
            return item.week, item.starts, following
    return None


def _interviews_between(settings: Settings, start: date, end: date) -> int:
    """Interview sessions planned on a local date in ``[start, end)``, including closed ones."""
    if not database_path(settings).is_file():
        return 0
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as db:
        return len(
            db.scalars(
                select(StudySession.id).where(
                    StudySession.kind == SessionKind.INTERVIEW.value,
                    StudySession.plan_date >= start.isoformat(),
                    StudySession.plan_date < end.isoformat(),
                )
            ).all()
        )


def _resolves(
    snapshot: Snapshot, topics: tuple[TopicFacts, ...], when: datetime
) -> tuple[ResolveDue, ...]:
    """Timed re-solves that are due, or whose window is open, on ready core problems only."""
    ready: dict[str, tuple[str, ProblemFacts]] = {}
    for topic in topics:
        core = topic.ready_core()
        if core is not None:
            ready[core.problem_id] = (topic.topic_id, core)
    running = {w.problem_id for w in snapshot.windows if w.status is WindowStatus.OPEN}
    due: list[ResolveDue] = []
    for problem_id, schedule in build_schedules(first_passes(snapshot), snapshot.windows).items():
        if problem_id not in ready or not (problem_id in running or schedule.is_due(when)):
            continue
        topic_id, core = ready[problem_id]
        due.append(
            ResolveDue(
                problem_id=problem_id,
                topic_id=topic_id,
                title=core.title,
                minutes=core.minutes,
                stage=schedule.stage,
                due_at=schedule.due_at,
                open=problem_id in running,
            )
        )
    return tuple(due)


def _path_topic_ids(core: CorePath) -> tuple[str, ...]:
    """Every topic on the path, in study order, each listed once."""
    ordered: dict[str, None] = {}
    for week in core.weeks:
        for topic_id in week.topics:
            ordered.setdefault(topic_id, None)
    return tuple(ordered)


def _topic_facts(settings: Settings, topic_id: str) -> TopicFacts:
    topic_dir = settings.content_dir / topic_id
    if not (topic_dir / "topic.yaml").is_file():
        return _absent(topic_id)
    try:
        package = load_topic(topic_dir)
    except ContentError:
        return _absent(topic_id)
    problems = tuple(
        ProblemFacts(
            problem_id=f"{topic_id}/{problem.dir.name}",
            title=problem.meta.title,
            status=problem.meta.status,
            core=problem.meta.core,
            minutes=problem.meta.minutes,
        )
        for problem in package.problems
        if problem.meta is not None
    )
    meta = package.meta
    return TopicFacts(
        topic_id=topic_id,
        title=meta.title,
        status=meta.status,
        objective=meta.objectives[0],
        requires=meta.requires,
        problems=problems,
    )


def _absent(topic_id: str) -> TopicFacts:
    """A path topic with no usable package: it is listed, never planned."""
    return TopicFacts(
        topic_id=topic_id,
        title=topic_id,
        status=None,
        objective="",
        requires=(),
        problems=(),
    )


def _familiar(snapshot: Snapshot, topics: tuple[TopicFacts, ...]) -> tuple[FamiliarProblem, ...]:
    """Ready problems with a passing submit, the oldest last pass first."""
    ready: dict[str, FamiliarProblem] = {}
    for topic in topics:
        if topic.status not in READY_STATUSES:
            continue
        for problem in topic.problems:
            if problem.status in READY_STATUSES:
                ready[problem.problem_id] = FamiliarProblem(
                    problem_id=problem.problem_id, topic_id=topic.topic_id, title=problem.title
                )
    last_pass: dict[str, datetime] = {}
    for attempt in snapshot.attempts:  # oldest first, so the newest pass wins
        if (
            attempt.mode == AttemptMode.SUBMIT.value
            and attempt.status == RunStatus.PASSED.value
            and attempt.problem_id in ready
        ):
            last_pass[attempt.problem_id] = attempt.submitted_at
    ordered = sorted(last_pass, key=lambda problem_id: (last_pass[problem_id], problem_id))
    return tuple(ready[problem_id] for problem_id in ordered)


# --- persistence -------------------------------------------------------------------------------


def _latest_open(db: Session) -> StudySession | None:
    """Resume the latest unfinished work, including sessions created before midnight."""
    return db.scalars(
        select(StudySession)
        .where(StudySession.status == _OPEN)
        .order_by(StudySession.created_at.desc(), StudySession.id.desc())
        .limit(1)
    ).first()


def _new_row(day: date, slot: int, plan: SessionPlan, when: datetime) -> StudySession:
    return StudySession(
        id=uuid4().hex,
        plan_date=day.isoformat(),
        slot=slot,
        kind=plan.kind.value,
        energy=plan.energy.value,
        status=_OPEN,
        topic_id=plan.topic_id,
        must_do=plan.must_do,
        first_action=plan.first_action,
        timed_problem_id=plan.timed_problem_id,
        plan_notes=_join_notes(plan.notes),
        created_at=when,
    )


def _write_plan(db: Session, row: StudySession, plan: SessionPlan) -> None:
    row.kind = plan.kind.value
    row.energy = plan.energy.value
    row.topic_id = plan.topic_id
    row.must_do = plan.must_do
    row.first_action = plan.first_action
    row.timed_problem_id = plan.timed_problem_id
    row.plan_notes = _join_notes(plan.notes)
    db.execute(delete(SessionBlock).where(SessionBlock.session_id == row.id))
    _add_blocks(db, row.id, plan)


def _add_blocks(db: Session, session_id: str, plan: SessionPlan) -> None:
    db.add_all(
        SessionBlock(
            session_id=session_id,
            position=position,
            name=block.name,
            minutes=block.minutes,
            detail=block.detail,
        )
        for position, block in enumerate(plan.blocks, start=1)
    )


def _load_plan(db: Session, row: StudySession) -> SessionPlan:
    blocks = db.scalars(
        select(SessionBlock)
        .where(SessionBlock.session_id == row.id)
        .order_by(SessionBlock.position)
    )
    return SessionPlan(
        kind=SessionKind(row.kind),
        energy=Energy(row.energy),
        blocks=tuple(Block(item.name, item.minutes, item.detail) for item in blocks),
        topic_id=row.topic_id,
        must_do=row.must_do,
        first_action=row.first_action,
        timed_problem_id=row.timed_problem_id,
        notes=_split_notes(row.plan_notes),
    )


def _introduce(db: Session, topic_id: str, when: datetime) -> bool:
    """Mark a topic introduced if it is unseen. Any stored state is kept: mastery never drops."""
    if db.get(TopicState, topic_id) is not None:
        return False
    db.add(TopicState(topic_id=topic_id, mastery=Mastery.INTRODUCED.value, introduced_at=when))
    return True


def _last_note(db: Session) -> TransitionNote | None:
    row = db.scalars(
        select(StudySession)
        .where(StudySession.status == _CLOSED)
        .order_by(
            StudySession.closed_at.desc(),
            StudySession.plan_date.desc(),
            StudySession.slot.desc(),
        )
        .limit(1)
    ).first()
    return None if row is None else _note(row)


def _stale_dates(db: Session, day: date) -> tuple[date, ...]:
    values = db.scalars(
        select(StudySession.plan_date)
        .where(StudySession.status == _OPEN, StudySession.plan_date != day.isoformat())
        .distinct()
        .order_by(StudySession.plan_date)
    )
    return tuple(date.fromisoformat(value) for value in values)


# --- value conversions and validation ----------------------------------------------------------


def _stored(row: StudySession, plan: SessionPlan, slots: int) -> StoredSession:
    return StoredSession(
        session_id=row.id,
        plan_date=date.fromisoformat(row.plan_date),
        slot=row.slot,
        sessions_per_day=slots,
        plan=plan,
    )


def _note(row: StudySession) -> TransitionNote:
    return TransitionNote(
        session_id=row.id,
        plan_date=date.fromisoformat(row.plan_date),
        slot=row.slot,
        kind=SessionKind(row.kind),
        topic_id=row.topic_id,
        stopped_where=row.stopped_where or "",
        next_step=row.next_step or "",
        blocker=row.blocker,
        closed_at=row.closed_at,
    )


def _view(
    *,
    day: date,
    slots: int,
    closed: int,
    last: TransitionNote | None,
    stale: tuple[date, ...],
    current: StoredSession | None,
    resumed: bool,
    replanned: bool,
) -> TodayView:
    return TodayView(
        local_date=day,
        sessions_per_day=slots,
        closed_today=closed,
        last_note=last,
        current=current,
        resumed=resumed,
        replanned=replanned,
        stale_open=stale,
    )


def _topics_of(rows: Iterable[StudySession], *, skip: StudySession | None) -> frozenset[str]:
    return frozenset(row.topic_id for row in rows if row is not skip and row.topic_id is not None)


def _note_fields(note: CloseNote) -> tuple[str, str, str | None]:
    stopped = _required(note.stopped_where, "Where you stopped")
    next_step = _required(note.next_step, "The next step")
    return stopped, next_step, _optional(note.blocker, "The blocker", NOTE_LIMIT)


def _required(value: str, label: str) -> str:
    text = value.strip()
    if not text:
        raise DogmaError(ErrorCode.SESSION_NOTE_INVALID, f"{label} cannot be empty.")
    if len(text) > NOTE_LIMIT:
        raise DogmaError(
            ErrorCode.SESSION_NOTE_INVALID, f"{label} is longer than {NOTE_LIMIT} characters."
        )
    return text


def _optional(value: str | None, label: str, limit: int) -> str | None:
    if value is None or not value.strip():
        return None
    text = value.strip()
    if len(text) > limit:
        raise DogmaError(
            ErrorCode.SESSION_NOTE_INVALID, f"{label} is longer than {limit} characters."
        )
    return text


def _aware(now: datetime) -> datetime:
    if now.tzinfo is None or now.utcoffset() is None:
        raise DogmaError(ErrorCode.SESSION_TIME_INVALID, "Session times must be timezone-aware.")
    return now.astimezone(UTC)


def _load_core(settings: Settings) -> CorePath:
    """The core path moved by the learner's time-warp offset (I-30a); the authored dates when 0."""
    return load_calendar(settings)


def _split_notes(text: str | None) -> tuple[str, ...]:
    return () if not text else tuple(text.split("\n"))


def _join_notes(notes: tuple[str, ...]) -> str | None:
    return "\n".join(notes) or None


def _no_session_message() -> str:
    return "No session is planned for today. Run dogma today to plan one."
