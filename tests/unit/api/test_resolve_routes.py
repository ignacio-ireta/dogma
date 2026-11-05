"""Timed re-solves over HTTP: start, read, abandon, the verdict on submit and hints counted.

The rules are the CLI's (ARCHITECTURE §4.4): these routes call the same ``resolve_service`` and
the same attempt recording, so each test checks the wire shape and the status, not the rules again.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from tests.unit.api.conftest import FakeGrader
from tests.unit.api.support import NOW, PROBLEM, FrozenClock

from dogma.core.config import Settings
from dogma.db.errors import StorageErrorCode
from dogma.db.session import open_session
from dogma.db.session_models import StudySession
from dogma.grading.models import RunStatus
from dogma.tutor.gateway import Gateway
from dogma.tutor.models import CompletionRequest, CompletionResult
from dogma.tutor.providers.fake import FakeProvider
from dogma.tutor.providers.local import LocalProvider

RESOLVES = "/api/v1/resolves"
CURRENT = f"{RESOLVES}/current"
ABANDON = f"{CURRENT}/abandon"
ATTEMPTS = "/api/v1/attempts"
HINT = "/api/v1/tutor/hint"
HINT_STREAM = "/api/v1/tutor/hint/stream"
DEBUG = "/api/v1/tutor/debug"
REPLY = "Think about the scalar that makes the residual orthogonal to b [S1]."
BUDGET_S = 20 * 60  # p01's meta.minutes


@pytest.fixture(autouse=True)
def _fake_grading(grader: FakeGrader) -> FakeGrader:
    """Every submit here is graded by the fake: a passing report unless a test scripts otherwise."""
    return grader


def _solution(settings: Settings) -> Path:
    return settings.workspace_dir / "cml" / "vector-operations" / "p01-vector-projection.py"


def _write(settings: Settings, text: str) -> None:
    path = _solution(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _at(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _first_pass(settings: Settings, client: TestClient) -> None:
    _write(settings, "def vector_projection(a, b):\n    return a\n")
    body = client.post(ATTEMPTS, json={"problem_id": PROBLEM, "mode": "submit"}).json()
    assert body["report"]["status"] == "passed"


def _start(client: TestClient) -> dict:
    response = client.post(RESOLVES, json={"problem_id": PROBLEM})
    assert response.status_code == 201, response.text
    return response.json()


def _tick(clock: FrozenClock, minutes: int) -> None:
    clock.now = clock.now + timedelta(minutes=minutes)


def test_start_needs_a_first_pass(client: TestClient) -> None:
    response = client.post(RESOLVES, json={"problem_id": PROBLEM})
    assert response.status_code == 409
    assert response.json()["code"] == "RESOLVE_NOT_SOLVED"


def test_start_opens_a_window_and_restores_the_starter(
    settings: Settings, client: TestClient
) -> None:
    _first_pass(settings, client)
    _write(settings, "# my old attempt\n")
    body = _start(client)
    window = body["window"]
    assert window["problem_id"] == PROBLEM
    assert window["status"] == "open"
    assert window["budget_s"] == BUDGET_S
    assert window["hints_used"] == 0
    assert window["closed_at"] is None
    assert _at(window["started_at"]) == NOW
    assert _at(window["deadline"]) == NOW + timedelta(seconds=BUDGET_S)
    assert body["previous_file_kept"] is True
    starter = (
        settings.content_dir / "cml/vector-operations/problems/p01-vector-projection/starter.py"
    )
    assert _solution(settings).read_text(encoding="utf-8") == starter.read_text(encoding="utf-8")
    kept = list((settings.var_dir / "workspace-history").rglob("*.py"))
    assert [path.read_text(encoding="utf-8") for path in kept] == ["# my old attempt\n"]


def test_current_is_null_when_no_window_is_open(client: TestClient) -> None:
    response = client.get(CURRENT)
    assert response.status_code == 200
    assert response.json() is None


def test_current_returns_the_open_window(settings: Settings, client: TestClient) -> None:
    _first_pass(settings, client)
    started = _start(client)["window"]
    current = client.get(CURRENT).json()
    assert current["id"] == started["id"]
    assert current["status"] == "open"
    assert current["deadline"] == started["deadline"]


def test_a_second_start_names_the_open_window(settings: Settings, client: TestClient) -> None:
    _first_pass(settings, client)
    _start(client)
    response = client.post(RESOLVES, json={"problem_id": PROBLEM})
    assert response.status_code == 409
    assert response.json()["code"] == "RESOLVE_ALREADY_OPEN"
    assert PROBLEM in response.json()["detail"]


def test_a_non_core_problem_is_refused(settings: Settings, client: TestClient) -> None:
    meta = settings.content_dir / "cml/vector-operations/problems/p01-vector-projection/meta.yaml"
    meta.write_text(meta.read_text(encoding="utf-8").replace("core: true", "core: false"))
    response = client.post(RESOLVES, json={"problem_id": PROBLEM})
    assert response.status_code == 422
    assert response.json()["code"] == "RESOLVE_NOT_CORE"


def test_recording_switched_off_is_refused(
    repo: Path, make_client: Callable[..., TestClient]
) -> None:
    off = Settings(repo_root=repo, record_attempts=False)
    with make_client(app_settings=off) as client:
        response = client.post(RESOLVES, json={"problem_id": PROBLEM})
    assert response.status_code == 409
    assert response.json()["code"] == "RESOLVE_RECORDING_OFF"


def test_an_unknown_problem_is_a_404(client: TestClient) -> None:
    response = client.post(RESOLVES, json={"problem_id": "cml/nope/p01-missing"})
    assert response.status_code == 404
    assert response.json()["code"] == "CONTENT_NOT_FOUND"


def test_start_body_is_strict(client: TestClient) -> None:
    response = client.post(RESOLVES, json={"problem_id": PROBLEM, "budget_s": 1})
    assert response.status_code == 422
    assert response.json()["code"] == "API_VALIDATION_FAILED"


def test_abandon_closes_the_window_without_penalty(settings: Settings, client: TestClient) -> None:
    _first_pass(settings, client)
    _start(client)
    response = client.post(ABANDON)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "abandoned"
    assert body["closed_at"] is not None
    assert client.get(CURRENT).json() is None
    again = client.post(ABANDON)
    assert again.status_code == 404
    assert again.json()["code"] == "RESOLVE_NOT_FOUND"


def test_submit_closes_the_window_and_returns_the_verdict_and_next_due(
    settings: Settings, client: TestClient, clock: FrozenClock, grader: FakeGrader
) -> None:
    _first_pass(settings, client)
    _start(client)
    _tick(clock, 5)
    response = client.post(ATTEMPTS, json={"problem_id": PROBLEM, "mode": "submit"})
    assert response.status_code == 201
    verdict = response.json()["timed_resolve"]
    assert verdict["problem_id"] == PROBLEM
    assert verdict["status"] == "passed"
    assert verdict["used_s"] == 300
    assert verdict["budget_s"] == BUDGET_S
    assert verdict["hints_used"] == 0
    due = verdict["next_due"]
    assert due["problem_id"] == PROBLEM
    assert due["stage"] == 1
    assert due["due_now"] is False
    assert _at(due["due_at"]) == NOW + timedelta(minutes=5, days=21)
    assert "21 days" in due["reason"]
    assert client.get(CURRENT).json() is None


def test_a_run_never_closes_the_window(settings: Settings, client: TestClient) -> None:
    _first_pass(settings, client)
    started = _start(client)["window"]
    body = client.post(ATTEMPTS, json={"problem_id": PROBLEM, "mode": "run"}).json()
    assert body["timed_resolve"] is None
    assert client.get(CURRENT).json()["id"] == started["id"]


def test_a_submit_after_the_budget_is_partial(
    settings: Settings, client: TestClient, clock: FrozenClock
) -> None:
    _first_pass(settings, client)
    _start(client)
    _tick(clock, 25)
    verdict = client.post(ATTEMPTS, json={"problem_id": PROBLEM, "mode": "submit"}).json()[
        "timed_resolve"
    ]
    assert verdict["status"] == "partial"
    assert verdict["used_s"] == 1500


def test_a_failing_submit_is_failed_and_due_again_in_three_days(
    settings: Settings, client: TestClient, clock: FrozenClock, grader: FakeGrader
) -> None:
    _first_pass(settings, client)
    _start(client)
    grader.status = RunStatus.FAILED
    _tick(clock, 2)
    verdict = client.post(ATTEMPTS, json={"problem_id": PROBLEM, "mode": "submit"}).json()[
        "timed_resolve"
    ]
    assert verdict["status"] == "failed"
    assert verdict["next_due"]["stage"] == 0
    assert _at(verdict["next_due"]["due_at"]) == NOW + timedelta(minutes=2, days=3)


def test_a_hint_during_the_window_is_counted_and_makes_the_submit_partial(
    settings: Settings, make_client: Callable[..., TestClient], grader: FakeGrader
) -> None:
    provider = FakeProvider([REPLY, REPLY, REPLY])
    with make_client(gateway=Gateway(settings, provider=provider)) as client:
        _first_pass(settings, client)
        _start(client)
        assert client.post(HINT, json={"problem_id": PROBLEM, "level": 1}).status_code == 200
        assert client.get(CURRENT).json()["hints_used"] == 1
        verdict = client.post(ATTEMPTS, json={"problem_id": PROBLEM, "mode": "submit"}).json()[
            "timed_resolve"
        ]
    assert verdict["status"] == "partial"
    assert verdict["hints_used"] == 1


def test_streamed_hints_and_debug_are_counted_too(
    settings: Settings,
    make_client: Callable[..., TestClient],
    grader: FakeGrader,
    clock: FrozenClock,
) -> None:
    provider = FakeProvider([REPLY, REPLY, REPLY])
    with make_client(gateway=Gateway(settings, provider=provider), model=provider) as client:
        _first_pass(settings, client)
        _start(client)
        grader.status = RunStatus.FAILED
        _tick(clock, 1)  # the failing run is the latest attempt, so debug has something to name
        client.post(ATTEMPTS, json={"problem_id": PROBLEM, "mode": "run"})
        assert client.post(HINT_STREAM, json={"problem_id": PROBLEM, "level": 1}).status_code == 200
        assert client.get(CURRENT).json()["hints_used"] == 1
        assert client.post(DEBUG, json={"problem_id": PROBLEM}).status_code == 200
        assert client.get(CURRENT).json()["hints_used"] == 2


def test_a_hint_with_no_window_creates_no_window(
    settings: Settings,
    make_client: Callable[..., TestClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbid_live_model(self: LocalProvider, req: CompletionRequest) -> CompletionResult:
        pytest.fail("unit tests must use the fake provider instead of the live model")

    monkeypatch.setattr(LocalProvider, "complete", forbid_live_model)
    gateway = Gateway(settings, provider=FakeProvider([REPLY]))
    with make_client(gateway=gateway) as client:
        assert client.post(HINT, json={"problem_id": PROBLEM, "level": 1}).status_code == 200
        assert client.get(CURRENT).json() is None


def test_foreign_origin_is_refused_on_resolve_routes(client: TestClient) -> None:
    response = client.post(
        RESOLVES,
        json={"problem_id": PROBLEM},
        headers={"Origin": "http://evil.example"},
    )
    assert response.status_code == 403
    assert response.json()["code"] == "API_REQUEST_REFUSED"


def test_a_window_open_for_more_than_twelve_hours_reads_as_abandoned(
    settings: Settings, client: TestClient, clock: FrozenClock
) -> None:
    _first_pass(settings, client)
    _start(client)
    clock.now = clock.now + timedelta(hours=13)
    assert client.get(CURRENT).json() is None


@pytest.mark.parametrize("is_open", [False, True])
def test_today_uses_the_persisted_target_independently_of_the_action_text(
    settings: Settings, client: TestClient, is_open: bool
) -> None:
    _first_pass(settings, client)
    if is_open:
        _start(client)
    assert client.get("/api/v1/today").status_code == 200
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as db:
        row = db.scalars(select(StudySession)).one()
        row.timed_problem_id = PROBLEM
        row.first_action = "Continue your practice on the Problem page."
    current = client.get("/api/v1/today").json()["current"]
    assert current["timed_resolve"]["problem_id"] == PROBLEM
    assert current["timed_resolve"]["open"] is is_open
    assert current["first_action"] == "Continue your practice on the Problem page."
