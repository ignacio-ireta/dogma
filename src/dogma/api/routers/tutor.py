from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from dogma.api.deps import ClockDep, GatewayDep, ModelDep, SettingsDep
from dogma.api.openapi import error_responses
from dogma.api.schemas.tutor import DebugRequest, HintOut, HintRequest, TutorAnswerOut
from dogma.api.services import tutor
from dogma.api.streaming import SSE_MEDIA_TYPE, event_stream

router = APIRouter(tags=["tutor"])
_TUTOR_ERRORS = error_responses(
    (403, "The walkthrough (L4) is locked until two recorded submits fail"),
    (404, "No problem with this id"),
    (503, "The local model is unavailable"),
)
_HINT_EVENTS = (
    "Server-sent events (`text/event-stream`), in this order:\n\n"
    "1. `reviewed`: the authored hint for the level, sent before any model call. Its payload is "
    "`{level, reviewed, hints_note}`. `reviewed` is null at L4 and when hints.md is missing.\n"
    "2. `nudge`, zero or more: `{text}`, a raw piece of the model's answer as it arrives. It has "
    "not been through the output guards.\n"
    "3. Exactly one terminal event:\n"
    "   - `done`: the final answer, the same payload as the JSON `HintOut`. Replace the streamed "
    "nudge text with `done.nudge.text`, or clear it when `done.nudge` is null. Then "
    "`done.nudge_note` says why no nudge was given (held back, or the model answered nothing).\n"
    "   - `unavailable`: `{code, note}`. The model did not answer; the reviewed hint stands.\n"
    "   - `error`: an RFC 9457 problem with the domain `code`.\n\n"
    "Refusals before the stream starts (unknown id, locked walkthrough) are JSON problems with the "
    "status of the JSON route, and no event is sent."
)
_DEBUG_EVENTS = (
    "Server-sent events (`text/event-stream`), in this order:\n\n"
    "1. `answer`, zero or more: `{text}`, a raw piece of the model's answer as it arrives. Replace "
    "it with `done.text` when the stream ends: long code is omitted in the final text.\n"
    "2. Exactly one terminal event: `done` (the same payload as the JSON `TutorAnswerOut`), "
    "`unavailable` (`{code, note}`: the model did not answer), or `error` (a problem).\n\n"
    "Refusals before the first event (no failing attempt, a refused endpoint) are JSON problems "
    "with the status of the JSON route."
)


def _stream_responses(*entries: tuple[int, str]) -> dict[int | str, dict[str, object]]:
    responses: dict[int | str, dict[str, object]] = dict(error_responses(*entries))
    responses[200] = {
        "description": "A stream of server-sent events, ending with one terminal event",
        "content": {SSE_MEDIA_TYPE: {"schema": {"type": "string"}}},
    }
    return responses


@router.post(
    "/tutor/hint",
    response_model=HintOut,
    summary="A hint at level 1-4: the reviewed text first, then the model's nudge",
    description=(
        "Synchronous. The local model can take 30-60 s (I-12 measured 33-42 s), so this is the "
        "slowest route; the reviewed hint is in the same response and never needs the model. "
        "Streaming variant: `/tutor/hint/stream` (ADR-0017)."
    ),
    responses=_TUTOR_ERRORS,
)
def hint(body: HintRequest, settings: SettingsDep, gateway: GatewayDep, clock: ClockDep) -> HintOut:
    return tutor.hint(
        settings,
        gateway,
        body.problem_id,
        level=body.level,
        confirm_walkthrough=body.confirm_walkthrough,
        now=clock(),
    )


@router.post(
    "/tutor/hint/stream",
    response_class=StreamingResponse,
    summary="A hint as server-sent events: the reviewed text at once, then the nudge",
    description=_HINT_EVENTS,
    responses=_stream_responses(
        (403, "The walkthrough (L4) is locked until two recorded submits fail"),
        (404, "No problem with this id"),
    ),
)
def hint_stream(
    body: HintRequest, request: Request, settings: SettingsDep, model: ModelDep, clock: ClockDep
) -> StreamingResponse:
    return event_stream(
        tutor.hint_events(
            settings,
            model,
            body.problem_id,
            level=body.level,
            confirm_walkthrough=body.confirm_walkthrough,
            instance=request.url.path,
            now=clock(),
        )
    )


@router.post(
    "/tutor/debug",
    response_model=TutorAnswerOut,
    summary="Name the first wrong concept or line of the latest failing attempt",
    responses=error_responses(
        (404, "No problem with this id"),
        (409, "No failing recorded attempt for this problem"),
        (503, "The local model is unavailable"),
    ),
)
def debug(
    body: DebugRequest, settings: SettingsDep, gateway: GatewayDep, clock: ClockDep
) -> TutorAnswerOut:
    return tutor.debug(settings, gateway, body.problem_id, now=clock())


@router.post(
    "/tutor/debug/stream",
    response_class=StreamingResponse,
    summary="The debug answer as server-sent events",
    description=_DEBUG_EVENTS,
    responses=_stream_responses(
        (404, "No problem with this id"),
        (409, "No failing recorded attempt for this problem"),
        (503, "The model server refused the endpoint (a configuration error)"),
    ),
)
def debug_stream(
    body: DebugRequest, request: Request, settings: SettingsDep, model: ModelDep, clock: ClockDep
) -> StreamingResponse:
    return event_stream(
        tutor.debug_events(settings, model, body.problem_id, instance=request.url.path, now=clock())
    )
