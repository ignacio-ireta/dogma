"""POST /tutor/hint/stream and /tutor/debug/stream: event order, guards and failure paths.

The fake provider streams in fixed pieces, so the tests see several deltas and tags split across
chunks. Each stream ends with one terminal event, and its ``done`` payload is the JSON answer.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient
from tests.unit.api.support import NOW, PROBLEM, ChunkedFakeProvider, FakeGrader, parse_events

from dogma.api.app import create_app
from dogma.api.deps import LocalModel
from dogma.api.services import tutor as tutor_service
from dogma.core.config import Settings
from dogma.core.errors import ErrorCode
from dogma.grading.models import RunStatus
from dogma.tutor.gateway import call_log_path
from dogma.tutor.models import ProviderError

HINT = "/api/v1/tutor/hint"
HINT_STREAM = "/api/v1/tutor/hint/stream"
DEBUG_STREAM = "/api/v1/tutor/debug/stream"
REPLY = "Think about the scalar that makes the residual orthogonal to b [S1]."
DOWN = ProviderError(ErrorCode.LLM_UNAVAILABLE, "unavailable", "the local model server is down")
NOTE_PREFIX = "Personalised nudge needs the local model:"


@pytest.fixture
def provider() -> ChunkedFakeProvider:
    return ChunkedFakeProvider([REPLY] * 6)


@pytest.fixture
def stream_client(
    provider: ChunkedFakeProvider, make_client: Callable[..., TestClient]
) -> Callable[..., TestClient]:
    return lambda **kwargs: make_client(model=kwargs.pop("model", provider), **kwargs)


def _write_workspace(settings: Settings) -> None:
    path = settings.workspace_dir / "cml" / "vector-operations" / "p01-vector-projection.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("def vector_projection(a, b):\n    return a\n", encoding="utf-8")


def _failing_attempt(settings: Settings, grader: FakeGrader, client: TestClient) -> None:
    _write_workspace(settings)
    grader.status = RunStatus.FAILED
    client.post("/api/v1/attempts", json={"problem_id": PROBLEM, "mode": "run"})


def _without_latency(body: dict[str, Any]) -> dict[str, Any]:
    nudge = body.get("nudge")
    if nudge is not None:
        body = {**body, "nudge": {**nudge, "latency_ms": 0}}
    return body


def _text(events: list[tuple[str, dict[str, Any]]], name: str) -> str:
    """The text of every delta event called ``name`` (``nudge`` for hints, ``answer`` for debug)."""
    return "".join(data["text"] for event, data in events if event == name)


def test_reviewed_comes_first_then_deltas_then_done(
    stream_client: Callable[..., TestClient],
) -> None:
    with stream_client() as client:
        response = client.post(HINT_STREAM, json={"problem_id": PROBLEM, "level": 1})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    events = parse_events(response.text)
    names = [name for name, _ in events]
    assert names[0] == "reviewed"
    assert names[-1] == "done"
    assert set(names[1:-1]) == {"nudge"}
    assert len(names) > 3, "the chunked fake provider must produce several deltas"

    reviewed = events[0][1]
    assert reviewed["level"] == 1
    assert reviewed["reviewed"]["level"] == 1
    assert reviewed["reviewed"]["text"]
    assert reviewed["hints_note"] is None
    assert _text(events, "nudge") == REPLY


def test_done_is_the_same_answer_the_json_route_gives(
    provider: ChunkedFakeProvider, stream_client: Callable[..., TestClient]
) -> None:
    provider.queue(REPLY)
    with stream_client() as client:
        json_body = client.post(HINT, json={"problem_id": PROBLEM, "level": 1}).json()
        events = parse_events(
            client.post(HINT_STREAM, json={"problem_id": PROBLEM, "level": 1}).text
        )

    done = events[-1][1]
    assert _without_latency(done) == _without_latency(json_body)


def test_the_verdict_guard_applies_to_done_and_the_raw_deltas_show_what_was_removed(
    provider: ChunkedFakeProvider, stream_client: Callable[..., TestClient]
) -> None:
    verdict = "Your code is correct, nice work. Think about the scalar [S1]."
    with stream_client(model=ChunkedFakeProvider([verdict])) as client:
        events = parse_events(
            client.post(HINT_STREAM, json={"problem_id": PROBLEM, "level": 1}).text
        )

    assert _text(events, "nudge") == verdict, "deltas are raw, before the guards"
    done = events[-1][1]
    assert done["nudge"]["text"].startswith("Think about the scalar")
    assert "Your code is correct" not in done["nudge"]["text"]
    assert "grader's call" in done["nudge"]["text"]


def test_thinking_blocks_never_reach_the_deltas(
    provider: ChunkedFakeProvider, stream_client: Callable[..., TestClient]
) -> None:
    provider.queue(f"<think>secret working</think>{REPLY}")
    with stream_client() as client:
        events = parse_events(
            client.post(HINT_STREAM, json={"problem_id": PROBLEM, "level": 1}).text
        )

    assert "secret" not in _text(events, "nudge")
    assert _text(events, "nudge") == REPLY
    assert "secret" not in events[-1][1]["nudge"]["text"]


def test_a_reply_that_is_only_reasoning_is_a_done_without_a_nudge_as_on_the_json_route(
    stream_client: Callable[..., TestClient],
) -> None:
    # The model answered, but with nothing usable. The reviewed hint stands, so PreparedHint
    # absorbs the empty answer into a note, exactly as the JSON route does.
    with stream_client(model=ChunkedFakeProvider(["<think>only reasoning</think>"])) as client:
        response = client.post(HINT_STREAM, json={"problem_id": PROBLEM, "level": 1})

    events = parse_events(response.text)
    assert [name for name, _ in events] == ["reviewed", "done"]
    done = events[1][1]
    assert done["nudge"] is None
    assert "empty answer" in done["nudge_note"]
    assert "only reasoning" not in response.text


def test_debug_with_an_empty_answer_ends_in_unavailable_with_its_code(
    settings: Settings,
    grader: FakeGrader,
    stream_client: Callable[..., TestClient],
) -> None:
    with stream_client() as client:
        _failing_attempt(settings, grader, client)
    with stream_client(model=ChunkedFakeProvider(["<think>only reasoning</think>"])) as client:
        response = client.post(DEBUG_STREAM, json={"problem_id": PROBLEM})

    events = parse_events(response.text)
    assert [name for name, _ in events] == ["unavailable"]
    assert events[0][1]["code"] == "LLM_RESPONSE_INVALID"
    assert "only reasoning" not in response.text


def test_the_walkthrough_lock_is_a_problem_before_any_event(
    provider: ChunkedFakeProvider, stream_client: Callable[..., TestClient]
) -> None:
    with stream_client() as client:
        response = client.post(HINT_STREAM, json={"problem_id": PROBLEM, "level": 4})

    assert response.status_code == 403
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == ErrorCode.TUTOR_LOCKED.value
    assert provider.requests == [], "no model call and no reviewed event"


def test_a_confirmed_walkthrough_streams_with_no_reviewed_text(
    stream_client: Callable[..., TestClient],
) -> None:
    with stream_client() as client:
        response = client.post(
            HINT_STREAM,
            json={"problem_id": PROBLEM, "level": 4, "confirm_walkthrough": True},
        )

    events = parse_events(response.text)
    assert events[0] == ("reviewed", {"level": 4, "reviewed": None, "hints_note": None})
    assert events[-1][0] == "done"
    assert events[-1][1]["reviewed"] is None
    assert events[-1][1]["nudge"]["mode"] == "hint"


def test_a_down_model_after_the_reviewed_hint_ends_in_unavailable(
    stream_client: Callable[..., TestClient],
) -> None:
    with stream_client(model=ChunkedFakeProvider([DOWN])) as client:
        response = client.post(HINT_STREAM, json={"problem_id": PROBLEM, "level": 2})

    assert response.status_code == 200
    events = parse_events(response.text)
    assert [name for name, _ in events] == ["reviewed", "unavailable"]
    assert events[0][1]["reviewed"]["text"]
    unavailable = events[1][1]
    assert unavailable["code"] == "LLM_UNAVAILABLE"
    assert unavailable["note"].startswith(NOTE_PREFIX)


def test_a_down_model_without_reviewed_text_still_ends_in_unavailable(
    stream_client: Callable[..., TestClient],
) -> None:
    body = {"problem_id": PROBLEM, "level": 4, "confirm_walkthrough": True}
    with stream_client(model=ChunkedFakeProvider([DOWN])) as client:
        response = client.post(HINT_STREAM, json=body)

    events = parse_events(response.text)
    assert [name for name, _ in events] == ["reviewed", "unavailable"]
    assert events[1][1]["code"] == "LLM_UNAVAILABLE"


def test_a_refused_endpoint_after_reviewed_is_an_error_event_and_is_logged(
    settings: Settings, stream_client: Callable[..., TestClient]
) -> None:
    refused = LocalModel("https://models.example.com/v1")
    with stream_client(model=refused) as client:
        response = client.post(HINT_STREAM, json={"problem_id": PROBLEM, "level": 1})

    events = parse_events(response.text)
    assert [name for name, _ in events] == ["reviewed", "error"]
    error = events[1][1]
    assert error["code"] == "LLM_ENDPOINT_REFUSED"
    assert error["status"] == 503
    assert error["instance"] == HINT_STREAM
    last_line = call_log_path(settings).read_text(encoding="utf-8").splitlines()[-1]
    assert '"outcome": "refused"' in last_line


def test_reviewed_is_sent_before_the_model_is_called(
    settings: Settings, provider: ChunkedFakeProvider
) -> None:
    frames = tutor_service.hint_events(
        settings,
        provider,
        PROBLEM,
        level=1,
        confirm_walkthrough=False,
        instance=HINT_STREAM,
        now=NOW,
    )
    first = next(frames)

    assert first.startswith("event: reviewed\n")
    assert provider.requests == [], "the reviewed hint is ready before any model call"
    assert [name for name, _ in parse_events("".join([first, *frames]))][-1] == "done"
    assert len(provider.requests) == 1


def test_an_unknown_problem_is_not_found_before_the_stream(
    stream_client: Callable[..., TestClient],
) -> None:
    with stream_client() as client:
        response = client.post(
            HINT_STREAM, json={"problem_id": "cml/vector-operations/p99-x", "level": 1}
        )
    assert response.status_code == 404
    assert response.json()["code"] == "CONTENT_NOT_FOUND"


def test_a_level_outside_the_ladder_is_a_validation_error(
    stream_client: Callable[..., TestClient],
) -> None:
    with stream_client() as client:
        response = client.post(HINT_STREAM, json={"problem_id": PROBLEM, "level": 5})
    assert response.status_code == 422
    assert response.json()["code"] == "API_VALIDATION_FAILED"


def test_the_stream_routes_are_in_the_openapi_contract(settings: Settings) -> None:
    schema = create_app(settings).openapi()
    for path in ("/api/v1/tutor/hint/stream", "/api/v1/tutor/debug/stream"):
        content = schema["paths"][path]["post"]["responses"]["200"]["content"]
        assert set(content) == {"text/event-stream"}


def test_debug_stream_names_the_failing_attempt_with_the_final_answer(
    settings: Settings,
    grader: FakeGrader,
    provider: ChunkedFakeProvider,
    stream_client: Callable[..., TestClient],
) -> None:
    provider.queue(REPLY)
    with stream_client() as client:
        _failing_attempt(settings, grader, client)
        response = client.post(DEBUG_STREAM, json={"problem_id": PROBLEM})

    events = parse_events(response.text)
    assert events[0][0] == "answer"
    assert {name for name, _ in events[:-1]} == {"answer"}
    assert events[-1][0] == "done"
    done = events[-1][1]
    assert done["mode"] == "debug"
    assert done["level"] is None
    assert _text(events, "answer") == REPLY


def test_debug_stream_omits_long_code_in_done_only(
    settings: Settings,
    grader: FakeGrader,
    provider: ChunkedFakeProvider,
    stream_client: Callable[..., TestClient],
) -> None:
    block = "Look at this:\n```python\na = 1\nb = 2\nc = 3\nd = 4\n```\nSee [S1]."
    with stream_client(model=ChunkedFakeProvider([block])) as client:
        _failing_attempt(settings, grader, client)
        events = parse_events(client.post(DEBUG_STREAM, json={"problem_id": PROBLEM}).text)

    assert "d = 4" in _text(events, "answer"), "the stream shows the raw code"
    done_text = events[-1][1]["text"]
    assert "d = 4" not in done_text
    assert "[code omitted" in done_text


def test_debug_stream_with_a_down_model_ends_in_unavailable(
    settings: Settings,
    grader: FakeGrader,
    provider: ChunkedFakeProvider,
    stream_client: Callable[..., TestClient],
) -> None:
    with stream_client() as client:
        _failing_attempt(settings, grader, client)
    down = ChunkedFakeProvider([DOWN])
    with stream_client(model=down) as client:
        response = client.post(DEBUG_STREAM, json={"problem_id": PROBLEM})

    assert response.status_code == 200
    events = parse_events(response.text)
    assert [name for name, _ in events] == ["unavailable"]
    assert events[0][1]["code"] == "LLM_UNAVAILABLE"


def test_debug_stream_without_a_failing_attempt_is_a_conflict(
    provider: ChunkedFakeProvider, stream_client: Callable[..., TestClient]
) -> None:
    with stream_client() as client:
        response = client.post(DEBUG_STREAM, json={"problem_id": PROBLEM})

    assert response.status_code == 409
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == "TUTOR_NO_FAILURE"
    assert provider.requests == []


def test_debug_stream_with_a_refused_endpoint_is_a_problem_before_any_event(
    settings: Settings,
    grader: FakeGrader,
    stream_client: Callable[..., TestClient],
) -> None:
    with stream_client() as client:
        _failing_attempt(settings, grader, client)
    refused = LocalModel("https://models.example.com/v1")
    with stream_client(model=refused) as client:
        response = client.post(DEBUG_STREAM, json={"problem_id": PROBLEM})

    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == "LLM_ENDPOINT_REFUSED"
