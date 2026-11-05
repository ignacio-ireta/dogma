from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from dogma.api.deps import ClockDep, SettingsDep
from dogma.api.openapi import error_responses
from dogma.api.schemas.learning import (
    CloseOut,
    CloseRequest,
    DueReviewsOut,
    ProgressOut,
    RatingRequest,
    ReviewOut,
    TodayOut,
)
from dogma.api.services import learning
from dogma.learning.planner import Energy

router = APIRouter(tags=["learning"])
EnergyQuery = Annotated[
    Energy | None,
    Query(description="low drops new material; a different value re-plans an open session."),
]
LimitQuery = Annotated[int, Query(ge=1, le=100)]
WeekQuery = Annotated[int | None, Query(ge=1, description="Show one study week (1-based).")]


@router.get(
    "/today",
    response_model=TodayOut,
    summary="Resume or plan today's next session, with the last transition note",
)
def get_today(settings: SettingsDep, clock: ClockDep, energy: EnergyQuery = None) -> TodayOut:
    return learning.today(settings, clock(), energy)


@router.post(
    "/sessions/{session_id}/close",
    response_model=CloseOut,
    summary="Close a session once: where you stopped, the next step, your answer",
    responses=error_responses(
        (404, "No session with this id"),
        (422, "A required note field is empty or too long"),
    ),
)
def close_session(
    session_id: str, body: CloseRequest, settings: SettingsDep, clock: ClockDep
) -> CloseOut:
    return learning.close(settings, clock(), session_id, body)


@router.get(
    "/reviews/due",
    response_model=DueReviewsOut,
    summary="Cards due now (enrols cards of studied topics first, as `dogma review` does)",
)
def get_due_reviews(
    settings: SettingsDep, clock: ClockDep, limit: LimitQuery = 20
) -> DueReviewsOut:
    return learning.due_reviews(settings, clock(), limit)


@router.post(
    "/reviews/{card_id:path}",
    response_model=ReviewOut,
    summary="Rate a due card (1 Again, 2 Hard, 3 Good, 4 Easy)",
    responses=error_responses(
        (404, "No card with this id"),
        (409, "Stale (schedule or content changed) or not due yet"),
    ),
)
def rate_review(
    card_id: str, body: RatingRequest, settings: SettingsDep, clock: ClockDep
) -> ReviewOut:
    return learning.rate(settings, clock(), card_id, body)


@router.get(
    "/progress",
    response_model=ProgressOut,
    summary="Core path by week: mastery, attempts, best result, and the summary",
    responses=error_responses((404, "The core path has no such week")),
)
def get_progress(settings: SettingsDep, clock: ClockDep, week: WeekQuery = None) -> ProgressOut:
    return learning.progress(settings, clock(), week)
