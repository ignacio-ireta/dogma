from __future__ import annotations

from fastapi import APIRouter

from dogma.api.deps import ClockDep, SettingsDep
from dogma.api.openapi import error_responses
from dogma.api.schemas.resolves import ResolveStartedOut, ResolveWindowOut, StartResolveRequest
from dogma.api.services import resolves

router = APIRouter(tags=["resolves"])


@router.post(
    "/resolves",
    response_model=ResolveStartedOut,
    status_code=201,
    summary="Start a timed, hint-free re-solve of a core problem (`dogma resolve`)",
    description=(
        "Opens the window and restores the starter in your workspace; the old file is kept under "
        "var/workspace-history/. The budget is the problem's minutes. The first submit of the "
        "problem decides the window (POST /attempts with mode submit)."
    ),
    responses=error_responses(
        (404, "No problem with this id"),
        (
            409,
            "Not solved yet, not verified or reviewed, recording is off, or another window is open",
        ),
        (422, "The problem is not core"),
    ),
)
def start_resolve(
    body: StartResolveRequest, settings: SettingsDep, clock: ClockDep
) -> ResolveStartedOut:
    return resolves.start(settings, body.problem_id, now=clock())


@router.get(
    "/resolves/current",
    response_model=ResolveWindowOut | None,
    summary="The open timed re-solve, or null when none is open",
    description=(
        "Always 200. The body is the open window (status `open`, with `deadline` and `hints_used`) "
        "or the JSON value `null` when no window is open. A window open for more than 12 hours is "
        "abandoned as this is read."
    ),
)
def current_resolve(settings: SettingsDep, clock: ClockDep) -> ResolveWindowOut | None:
    return resolves.current(settings, now=clock())


@router.post(
    "/resolves/current/abandon",
    response_model=ResolveWindowOut,
    summary="Abandon the open timed re-solve. No penalty: the schedule does not change",
    responses=error_responses((404, "No timed re-solve is open")),
)
def abandon_resolve(settings: SettingsDep, clock: ClockDep) -> ResolveWindowOut:
    return resolves.abandon(settings, now=clock())
