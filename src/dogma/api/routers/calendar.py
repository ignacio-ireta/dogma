"""The course calendar and its moves: read it, move it, or start the course over (ADR-0019)."""

from __future__ import annotations

from fastapi import APIRouter

from dogma.api.deps import ClockDep, SettingsDep
from dogma.api.openapi import error_responses
from dogma.api.schemas.calendar import (
    CalendarOut,
    CalendarWarpOut,
    CalendarWarpRequest,
    ResetOut,
    ResetRequest,
)
from dogma.api.services import calendar

router = APIRouter(tags=["calendar"])


@router.get(
    "/calendar",
    response_model=CalendarOut,
    summary="The course calendar: the offset, this week, the anchor week and the last move",
    description="Read-only. Nothing is stored, and no database is created.",
)
def get_calendar(settings: SettingsDep, clock: ClockDep) -> CalendarOut:
    return calendar.status(settings, now=clock())


@router.post(
    "/calendar/warp",
    response_model=CalendarWarpOut,
    summary="Move the calendar: sync, to a week, ahead, back, or back to the original dates",
    description="With dry_run the move is previewed and nothing is written.",
    responses=error_responses(
        (409, "Recording is switched off, so a real move is refused"),
        (422, "The move needs its own amount, or the amount is out of range"),
    ),
)
def warp(body: CalendarWarpRequest, settings: SettingsDep, clock: ClockDep) -> CalendarWarpOut:
    return calendar.warp(settings, body, now=clock())


@router.post(
    "/course/reset",
    response_model=ResetOut,
    summary="Back up, clear all progress, and make this week week 1",
    description=(
        "Needs `confirm` set to the word reset. The backup is written first; if it fails, "
        "nothing is cleared."
    ),
    responses=error_responses(
        (409, "Recording is switched off, so nothing is reset"),
        (422, "The confirmation word is missing or wrong"),
    ),
)
def reset(body: ResetRequest, settings: SettingsDep, clock: ClockDep) -> ResetOut:
    return calendar.reset(settings, now=clock())
