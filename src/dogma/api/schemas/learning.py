"""Today, transition notes, review cards and progress: the learner's plan and memory."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field

from dogma.api.schemas.resolves import ReadinessOut, TimedResolveOut
from dogma.learning.planner import Energy, SessionKind


class BlockOut(BaseModel):
    name: str
    minutes: int
    detail: str


class SessionOut(BaseModel):
    session_id: str
    plan_date: date
    slot: int
    sessions_per_day: int
    kind: SessionKind
    energy: Energy
    topic_id: str | None
    must_do: str | None
    first_action: str
    minutes: int
    blocks: list[BlockOut]
    notes: list[str]
    # The timed re-solve the Re-solve block plans, or null (untimed practice, or no such block).
    timed_resolve: TimedResolveOut | None


class NoteOut(BaseModel):
    session_id: str
    plan_date: date
    slot: int
    kind: SessionKind
    topic_id: str | None
    stopped_where: str
    next_step: str
    blocker: str | None
    closed_at: datetime | None


class TodayOut(BaseModel):
    local_date: date
    sessions_per_day: int
    closed_today: int
    last_note: NoteOut | None
    current: SessionOut | None
    resumed: bool
    replanned: bool
    stale_open: list[date]


class CloseRequest(BaseModel):
    """The note a learner writes when stopping. Empty required fields are a 422 from the service."""

    stopped_where: str
    next_step: str
    blocker: str | None = None
    answer: str | None = None


class CloseOut(BaseModel):
    note: NoteOut
    already_closed: bool
    introduced_topic: str | None
    new_cards: int


class ReviewCardOut(BaseModel):
    card_id: str
    topic_id: str
    question: str
    answer: str
    due: datetime
    content_hash: str


class DueReviewsOut(BaseModel):
    enrolled: int
    cards: list[ReviewCardOut]


class RatingRequest(BaseModel):
    """The rating plus the due time and content hash the learner saw (stale ratings are refused)."""

    model_config = ConfigDict(extra="forbid")

    rating: int = Field(ge=1, le=4)
    expected_due: datetime
    expected_content_hash: str


class ReviewOut(BaseModel):
    card_id: str
    topic_id: str
    due: datetime
    rating: int
    mastery: str


class BestOut(BaseModel):
    passed: int
    total: int
    mode: str


class TopicProgressOut(BaseModel):
    topic_id: str
    title: str
    status: str
    mastery: str
    solved: bool
    attempts: int
    best: BestOut | None
    last_attempt_at: datetime | None


class WeekProgressOut(BaseModel):
    week: int
    theme: str
    starts: date
    solved: int
    total: int
    topics: list[TopicProgressOut]


class CourseOut(BaseModel):
    id: str
    title: str
    start: date
    end: date


class ProgressSummaryOut(BaseModel):
    solved_topics: int
    total_topics: int
    current_week: int
    week_start: date
    attempts_this_week: int


class ProgressOut(BaseModel):
    course: CourseOut
    summary: ProgressSummaryOut
    weeks: list[WeekProgressOut]
    readiness: ReadinessOut
