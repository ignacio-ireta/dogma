"""POST /tutor/hint and /tutor/debug, with a fake provider behind the real gateway."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient
from tests.unit.api.support import PROBLEM, FakeGrader

from dogma.api.deps import get_gateway
from dogma.core.config import Settings
from dogma.core.errors import ErrorCode
from dogma.grading.models import RunStatus
from dogma.tutor.gateway import Gateway
from dogma.tutor.models import ProviderError
from dogma.tutor.providers.fake import FakeProvider

HINT = "/api/v1/tutor/hint"
DEBUG = "/api/v1/tutor/debug"
REPLY = "Think about the scalar that makes the residual orthogonal to b [S1]."


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider([REPLY, REPLY, REPLY, REPLY])


@pytest.fixture
def tutor_client(
    settings: Settings,
    make_client: Callable[..., TestClient],
    provider: FakeProvider,
) -> Callable[[], TestClient]:
    gateway = Gateway(settings, provider=provider)
    return lambda: make_client(gateway=gateway)


def _write_workspace(settings: Settings) -> None:
    path = settings.workspace_dir / "cml" / "vector-operations" / "p01-vector-projection.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("def vector_projection(a, b):\n    return a\n", encoding="utf-8")


def test_level_one_hint_returns_the_reviewed_hint_and_the_nudge(
    tutor_client: Callable[[], TestClient], provider: FakeProvider
) -> None:
    with tutor_client() as client:
        response = client.post(HINT, json={"problem_id": PROBLEM, "level": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["level"] == 1
    assert body["reviewed"]["level"] == 1
    assert body["reviewed"]["text"]
    nudge = body["nudge"]
    assert nudge["mode"] == "hint"
    assert "scalar" in nudge["text"]
    assert isinstance(nudge["sources"], list)
    assert nudge["prompt_version"] >= 1
    assert nudge["model"]
    assert isinstance(nudge["cited"], bool)
    assert provider.requests


def test_hint_level_defaults_to_one(tutor_client: Callable[[], TestClient]) -> None:
    with tutor_client() as client:
        body = client.post(HINT, json={"problem_id": PROBLEM}).json()
    assert body["level"] == 1
    assert body["reviewed"] is not None


@pytest.mark.parametrize("level", [0, 5])
def test_hint_level_outside_the_ladder_is_a_validation_error(
    tutor_client: Callable[[], TestClient], level: int
) -> None:
    with tutor_client() as client:
        response = client.post(HINT, json={"problem_id": PROBLEM, "level": level})
    assert response.status_code == 422
    assert response.json()["code"] == "API_VALIDATION_FAILED"


def test_walkthrough_is_locked_until_two_recorded_submits_fail(
    tutor_client: Callable[[], TestClient], provider: FakeProvider
) -> None:
    with tutor_client() as client:
        response = client.post(HINT, json={"problem_id": PROBLEM, "level": 4})
    assert response.status_code == 403
    assert response.json()["code"] == ErrorCode.TUTOR_LOCKED.value
    assert provider.requests == []


def test_walkthrough_opens_when_the_learner_confirms(
    tutor_client: Callable[[], TestClient],
) -> None:
    with tutor_client() as client:
        response = client.post(
            HINT, json={"problem_id": PROBLEM, "level": 4, "confirm_walkthrough": True}
        )
    assert response.status_code == 200
    body = response.json()
    assert body["level"] == 4
    assert body["reviewed"] is None
    assert body["nudge"]["mode"] == "hint"


def test_debug_without_a_failing_attempt_is_a_conflict(
    tutor_client: Callable[[], TestClient],
) -> None:
    with tutor_client() as client:
        response = client.post(DEBUG, json={"problem_id": PROBLEM})
    assert response.status_code == 409
    assert response.json()["code"] == "TUTOR_NO_FAILURE"


def test_debug_names_the_latest_failing_attempt(
    settings: Settings,
    tutor_client: Callable[[], TestClient],
    grader: FakeGrader,
    provider: FakeProvider,
) -> None:
    _write_workspace(settings)
    grader.status = RunStatus.FAILED
    with tutor_client() as client:
        client.post("/api/v1/attempts", json={"problem_id": PROBLEM, "mode": "run"})
        response = client.post(DEBUG, json={"problem_id": PROBLEM})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "debug"
    assert body["level"] is None
    assert provider.requests


def test_a_down_model_keeps_the_reviewed_hint_and_explains_the_nudge(
    settings: Settings, make_client: Callable[..., TestClient]
) -> None:
    down = FakeProvider([ProviderError(ErrorCode.LLM_UNAVAILABLE, "unavailable", "model is down")])
    with make_client(gateway=Gateway(settings, provider=down)) as client:
        response = client.post(HINT, json={"problem_id": PROBLEM, "level": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["reviewed"]["text"]
    assert body["nudge"] is None
    assert "local model" in body["nudge_note"]


def test_unavailable_model_without_a_reviewed_hint_is_a_503(
    settings: Settings, make_client: Callable[..., TestClient]
) -> None:
    down = FakeProvider([ProviderError(ErrorCode.LLM_UNAVAILABLE, "unavailable", "model is down")])
    with make_client(
        gateway=Gateway(settings, provider=down), raise_server_exceptions=False
    ) as client:
        response = client.post(
            HINT, json={"problem_id": PROBLEM, "level": 4, "confirm_walkthrough": True}
        )
    assert response.status_code == 503
    assert response.json()["code"] == "LLM_UNAVAILABLE"


def test_unknown_problem_is_not_found(tutor_client: Callable[[], TestClient]) -> None:
    with tutor_client() as client:
        response = client.post(HINT, json={"problem_id": "cml/vector-operations/p99-x", "level": 1})
    assert response.status_code == 404
    assert response.json()["code"] == "CONTENT_NOT_FOUND"


def test_gateway_is_resolved_through_the_dependency(
    settings: Settings, make_client: Callable[..., TestClient]
) -> None:
    # The route never builds its own gateway, so tests and the CLI can swap providers.
    seen: list[object] = []

    def build() -> Gateway:
        seen.append(object())
        return Gateway(settings, provider=FakeProvider([REPLY]))

    with make_client() as client:
        client.app.dependency_overrides[get_gateway] = build  # type: ignore[attr-defined]
        client.post(HINT, json={"problem_id": PROBLEM, "level": 1})
    assert seen
