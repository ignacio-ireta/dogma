"""Today, transition notes, the review queue and progress: the same services the CLI calls."""

from __future__ import annotations

from datetime import date, datetime

from dogma.api.ids import is_card_id, is_session_id
from dogma.api.schemas.interview import InterviewReadyOut, WeekInterviewReadyOut
from dogma.api.schemas.learning import (
    BestOut,
    BlockOut,
    CloseOut,
    CloseRequest,
    CourseOut,
    DueReviewsOut,
    NoteOut,
    ProgressOut,
    ProgressSummaryOut,
    RatingRequest,
    ReviewCardOut,
    ReviewOut,
    SessionOut,
    TodayOut,
    TopicProgressOut,
    WeekProgressOut,
)
from dogma.api.schemas.resolves import TimedResolveOut
from dogma.api.services import resolves
from dogma.api.services.content import problem_ref
from dogma.content.schemas import CorePath, PathWeek
from dogma.core.config import Settings
from dogma.core.errors import DogmaError, ErrorCode
from dogma.db.models import Mastery
from dogma.db.progress import Snapshot, attempts_between, read_snapshot, topic_progress
from dogma.db.session import database_path
from dogma.learning import resolve_service, sessions
from dogma.learning.calendar import load_calendar
from dogma.learning.cards import ReviewCard, due_cards, sync_cards
from dogma.learning.path import current_week, topic_row
from dogma.learning.planner import Energy
from dogma.learning.review import ReviewResult, rate_card
from dogma.learning.sessions import (
    CloseNote,
    CloseResult,
    StoredSession,
    TodayView,
    TransitionNote,
)


def today(settings: Settings, now: datetime, energy: Energy | None) -> TodayOut:
    view = sessions.today(settings, now=now, energy=energy)
    return _today_out(settings, view, now)


def close(settings: Settings, now: datetime, session_id: str, body: CloseRequest) -> CloseOut:
    if not is_session_id(session_id) or not database_path(settings).is_file():
        raise _no_session(session_id)
    note = CloseNote(
        stopped_where=body.stopped_where,
        next_step=body.next_step,
        blocker=body.blocker,
    )
    result = sessions.close_session(settings, session_id, note, answer=body.answer, now=now)
    return _close_out(result)


def due_reviews(settings: Settings, now: datetime, limit: int) -> DueReviewsOut:
    enrolled = sync_cards(settings, now=now)
    cards = due_cards(settings, now=now, limit=limit)
    return DueReviewsOut(enrolled=enrolled, cards=[_card_out(card) for card in cards])


def rate(settings: Settings, now: datetime, card_id: str, body: RatingRequest) -> ReviewOut:
    if not is_card_id(card_id) or not database_path(settings).is_file():
        raise DogmaError(ErrorCode.REVIEW_NOT_FOUND, "This card is no longer available.")
    result = rate_card(
        settings,
        card_id,
        body.rating,
        now=now,
        expected_due=body.expected_due,
        expected_content_hash=body.expected_content_hash,
    )
    return _review_out(result)


def progress(settings: Settings, now: datetime, week: int | None) -> ProgressOut:
    core = load_calendar(settings)
    chosen = _chosen_weeks(core, week)
    snapshot = read_snapshot(settings, now=now)
    today_date = sessions.local_day(now)
    topic_ids = [topic for item in core.weeks for topic in item.topics]
    readiness = resolve_service.readiness(settings, topic_ids, snapshot, now)
    return ProgressOut(
        course=CourseOut(id=core.id, title=core.title, start=core.start, end=core.end),
        summary=_summary(core, snapshot, today_date),
        weeks=[_week(settings, item, snapshot) for item in chosen],
        readiness=resolves.readiness_out(
            readiness, now=now, interview_ready=_interview_ready(core, snapshot)
        ),
    )


# --- mapping ---------------------------------------------------------------------------------


def _today_out(settings: Settings, view: TodayView, now: datetime) -> TodayOut:
    return TodayOut(
        local_date=view.local_date,
        sessions_per_day=view.sessions_per_day,
        closed_today=view.closed_today,
        last_note=None if view.last_note is None else _note_out(view.last_note),
        current=None if view.current is None else _session_out(settings, view.current, now),
        resumed=view.resumed,
        replanned=view.replanned,
        stale_open=list(view.stale_open),
    )


def _timed_resolve_out(
    settings: Settings, problem_id: str | None, now: datetime
) -> TimedResolveOut | None:
    if problem_id is None:
        return None
    try:
        ref = problem_ref(settings, problem_id)
    except DogmaError:
        return None
    return TimedResolveOut(
        problem_id=ref.id,
        title=ref.meta.title,
        minutes=ref.meta.minutes,
        open=resolve_service.open_window_for(settings, problem_id, now=now) is not None,
    )


def _session_out(settings: Settings, stored: StoredSession, now: datetime) -> SessionOut:
    plan = stored.plan
    return SessionOut(
        timed_resolve=_timed_resolve_out(settings, plan.timed_problem_id, now),
        session_id=stored.session_id,
        plan_date=stored.plan_date,
        slot=stored.slot,
        sessions_per_day=stored.sessions_per_day,
        kind=plan.kind,
        energy=plan.energy,
        topic_id=plan.topic_id,
        must_do=plan.must_do,
        first_action=plan.first_action,
        minutes=plan.minutes,
        blocks=[
            BlockOut(name=block.name, minutes=block.minutes, detail=block.detail)
            for block in plan.blocks
        ],
        notes=list(plan.notes),
    )


def _note_out(note: TransitionNote) -> NoteOut:
    return NoteOut(
        session_id=note.session_id,
        plan_date=note.plan_date,
        slot=note.slot,
        kind=note.kind,
        topic_id=note.topic_id,
        stopped_where=note.stopped_where,
        next_step=note.next_step,
        blocker=note.blocker,
        closed_at=note.closed_at,
    )


def _close_out(result: CloseResult) -> CloseOut:
    return CloseOut(
        note=_note_out(result.note),
        already_closed=result.already_closed,
        introduced_topic=result.introduced_topic,
        new_cards=result.new_cards,
    )


def _card_out(card: ReviewCard) -> ReviewCardOut:
    return ReviewCardOut(
        card_id=card.card_id,
        topic_id=card.topic_id,
        question=card.question,
        answer=card.answer,
        due=card.due,
        content_hash=card.content_hash,
    )


def _review_out(result: ReviewResult) -> ReviewOut:
    return ReviewOut(
        card_id=result.card_id,
        topic_id=result.topic_id,
        due=result.due,
        rating=result.rating,
        mastery=result.mastery.value,
    )


def _chosen_weeks(core: CorePath, week: int | None) -> tuple[PathWeek, ...]:
    if week is None:
        return core.weeks
    matches = tuple(item for item in core.weeks if item.week == week)
    if not matches:
        raise DogmaError(
            ErrorCode.CONTENT_NOT_FOUND,
            f"the core path has weeks 1 to {len(core.weeks)}, not {week}",
        )
    return matches


def _summary(core: CorePath, snapshot: Snapshot, today: date) -> ProgressSummaryOut:
    """Topics solved across the whole path, and attempts in the study week that holds today."""
    topics = {topic for item in core.weeks for topic in item.topics}
    solved = sum(topic_progress(topic, snapshot).solved for topic in topics)
    current = current_week(core.weeks, today)
    # Before the path starts, the week counts from today, so practice done now shows up.
    start = min(current.starts, today)
    later = [item.starts for item in core.weeks if item.starts > current.starts]
    attempts = attempts_between(snapshot, start, min(later) if later else None)
    return ProgressSummaryOut(
        solved_topics=solved,
        total_topics=len(topics),
        current_week=current.week,
        week_start=start,
        attempts_this_week=attempts,
    )


def _interview_ready(core: CorePath, snapshot: Snapshot) -> InterviewReadyOut:
    """Topics whose mastery is interview-ready, per study week and in total (`dogma progress`)."""
    weeks = [
        WeekInterviewReadyOut(
            week=item.week,
            ready=sum(
                topic_progress(topic_id, snapshot).mastery is Mastery.INTERVIEW_READY
                for topic_id in item.topics
            ),
            total=len(item.topics),
        )
        for item in core.weeks
    ]
    return InterviewReadyOut(
        ready=sum(item.ready for item in weeks),
        total=sum(item.total for item in weeks),
        weeks=weeks,
    )


def _week(settings: Settings, item: PathWeek, snapshot: Snapshot) -> WeekProgressOut:
    rows = [_topic_entry(settings, topic_id, snapshot) for topic_id in item.topics]
    return WeekProgressOut(
        week=item.week,
        theme=item.theme,
        starts=item.starts,
        solved=sum(row.solved for row in rows),
        total=len(rows),
        topics=rows,
    )


def _topic_entry(settings: Settings, topic_id: str, snapshot: Snapshot) -> TopicProgressOut:
    info = topic_row(settings, topic_id)
    progress_row = topic_progress(topic_id, snapshot)
    best = progress_row.best
    return TopicProgressOut(
        topic_id=topic_id,
        title=info.title,
        status=info.status,
        mastery=progress_row.mastery.value,
        solved=progress_row.solved,
        attempts=progress_row.attempts,
        best=(
            None if best is None else BestOut(passed=best.passed, total=best.total, mode=best.mode)
        ),
        last_attempt_at=progress_row.last_attempt_at,
    )


def _no_session(session_id: str) -> DogmaError:
    return DogmaError(ErrorCode.SESSION_NOT_FOUND, f"no session {session_id!r}")
