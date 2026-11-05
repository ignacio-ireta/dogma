"""Mock interviews over HTTP (ARCHITECTURE §4.4): start, reveal, score, debrief and abandon."""

from __future__ import annotations

from fastapi import APIRouter

from dogma.api.deps import ClockDep, SettingsDep
from dogma.api.openapi import error_responses
from dogma.api.schemas.mocks import (
    AbandonedOut,
    MockDebriefOut,
    MockPlanOut,
    MockReportOut,
    MockStartRequest,
    MockStatusOut,
    RevealOut,
    RevealRequest,
    ScoreRequest,
)
from dogma.api.services import mocks

router = APIRouter(tags=["mocks"])


@router.get(
    "/mocks/current",
    response_model=MockStatusOut,
    summary="The open mock (or null) and the last scored one",
    description="Always 200. Nothing is stored by reading it, and no database is created.",
)
def get_current(settings: SettingsDep) -> MockStatusOut:
    return mocks.status(settings)


@router.post(
    "/mocks",
    response_model=MockPlanOut,
    status_code=201,
    summary="Start a mock: the case, the coding problem, the theory question and the timings",
    responses=error_responses(
        (404, "The case id is not a ready sys-track case"),
        (409, "A mock is already open, or recording is switched off"),
    ),
)
def start_mock(body: MockStartRequest, settings: SettingsDep, clock: ClockDep) -> MockPlanOut:
    return mocks.start(settings, body, now=clock())


@router.post(
    "/mocks/current/reveal",
    response_model=RevealOut,
    summary="Ask the interviewer something about the case (released only on a matching question)",
    responses=error_responses((409, "No mock is open")),
)
def reveal(body: RevealRequest, settings: SettingsDep) -> RevealOut:
    return mocks.reveal(settings, body.question)


@router.post(
    "/mocks/current/score",
    response_model=MockReportOut,
    status_code=201,
    summary="Score the 30 items of the open mock; this closes it as scored",
    description="Every item needs a whole number from 1 to 4. A bad score stores nothing.",
    responses=error_responses(
        (409, "No mock is open, or recording is switched off"),
        (422, "Items missing, unknown, or out of range"),
    ),
)
def score(body: ScoreRequest, settings: SettingsDep, clock: ClockDep) -> MockReportOut:
    return mocks.score(settings, body.scores, now=clock())


@router.post(
    "/mocks/current/abandon",
    response_model=AbandonedOut,
    summary="Close the open mock without scores. Nothing is stored for it",
    responses=error_responses((409, "No mock is open")),
)
def abandon(settings: SettingsDep, clock: ClockDep) -> AbandonedOut:
    return mocks.abandon(settings, now=clock())


@router.get(
    "/mocks/{session_id}/debrief",
    response_model=MockDebriefOut,
    summary="A scored mock's part totals, band, strongest and weakest part, and its case debrief",
    responses=error_responses((404, "No scored mock with this id")),
)
def get_debrief(session_id: str, settings: SettingsDep) -> MockDebriefOut:
    return mocks.debrief(settings, session_id)
