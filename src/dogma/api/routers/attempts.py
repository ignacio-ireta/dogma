from __future__ import annotations

from fastapi import APIRouter

from dogma.api.deps import ClockDep, SettingsDep
from dogma.api.openapi import error_responses
from dogma.api.schemas.attempts import AttemptCreated, AttemptRecord, AttemptRequest
from dogma.api.services import attempts

router = APIRouter(tags=["attempts"])


@router.post(
    "/attempts",
    response_model=AttemptCreated,
    status_code=201,
    summary="Grade the workspace file (run or submit) and record it",
    description=(
        "Synchronous: returns when the sandboxed run finishes. Each run is capped by the problem's "
        "timeout_s (30 s for every current problem). A failing solution is a 201 with status data, "
        "not an error. A submit closes the open timed re-solve on its problem and returns the "
        "verdict in `timed_resolve` (null otherwise); a run never does."
    ),
    responses=error_responses(
        (404, "No problem with this id"),
        (409, "No workspace file yet: save it first"),
    ),
)
def create_attempt(body: AttemptRequest, settings: SettingsDep, clock: ClockDep) -> AttemptCreated:
    return attempts.create(settings, body.problem_id, body.mode, clock=clock)


@router.get(
    "/attempts/{attempt_id}",
    response_model=AttemptRecord,
    summary="A recorded attempt's verdict and counts",
    responses=error_responses((404, "No recorded attempt with this id")),
)
def get_attempt(attempt_id: str, settings: SettingsDep) -> AttemptRecord:
    return attempts.read(settings, attempt_id)
