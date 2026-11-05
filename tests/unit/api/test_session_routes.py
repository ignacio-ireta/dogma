"""GET /today plans or resumes a session; POST /sessions/{id}/close writes its transition note."""

from __future__ import annotations

from fastapi.testclient import TestClient
from tests.unit.api.support import TOPIC


def _note(**overrides: str) -> dict[str, str]:
    base = {"stopped_where": "Practice, example 2", "next_step": "Solve p01 again"}
    base.update(overrides)
    return base


def test_today_plans_the_first_session_of_the_day(client: TestClient) -> None:
    body = client.get("/api/v1/today").json()
    assert body["local_date"] == "2026-10-26"
    assert body["sessions_per_day"] >= 1
    assert body["closed_today"] == 0
    assert body["last_note"] is None
    current = body["current"]
    assert current is not None
    assert current["slot"] == 1
    assert current["kind"] == "learn"
    assert current["energy"] == "normal"
    assert current["topic_id"] == TOPIC
    assert current["first_action"]
    assert sum(block["minutes"] for block in current["blocks"]) == current["minutes"]
    assert body["resumed"] is False
    assert body["replanned"] is False


def test_a_repeat_visit_resumes_the_same_session(client: TestClient) -> None:
    first = client.get("/api/v1/today").json()
    again = client.get("/api/v1/today").json()
    assert again["current"]["session_id"] == first["current"]["session_id"]
    assert again["resumed"] is True
    assert again["replanned"] is False


def test_low_energy_replans_the_open_session_in_place(client: TestClient) -> None:
    first = client.get("/api/v1/today").json()
    low = client.get("/api/v1/today", params={"energy": "low"}).json()
    assert low["current"]["session_id"] == first["current"]["session_id"]
    assert low["current"]["energy"] == "low"
    assert low["replanned"] is True


def test_invalid_energy_is_a_validation_error(client: TestClient) -> None:
    response = client.get("/api/v1/today", params={"energy": "loud"})
    assert response.status_code == 422
    assert response.json()["code"] == "API_VALIDATION_FAILED"


def test_close_stores_the_note_and_enrols_the_topic(client: TestClient) -> None:
    session = client.get("/api/v1/today").json()["current"]
    response = client.post(
        f"/api/v1/sessions/{session['session_id']}/close",
        json=_note(answer="A projection is a shadow.", blocker="none"),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["already_closed"] is False
    assert body["introduced_topic"] == TOPIC
    assert body["new_cards"] > 0
    assert body["note"]["stopped_where"] == "Practice, example 2"
    assert body["note"]["next_step"] == "Solve p01 again"
    assert body["note"]["blocker"] == "none"
    assert body["note"]["kind"] == "learn"


def test_second_close_is_idempotent_and_keeps_the_first_note(client: TestClient) -> None:
    session_id = client.get("/api/v1/today").json()["current"]["session_id"]
    client.post(f"/api/v1/sessions/{session_id}/close", json=_note())
    again = client.post(
        f"/api/v1/sessions/{session_id}/close", json=_note(next_step="Something else")
    )
    assert again.status_code == 200
    body = again.json()
    assert body["already_closed"] is True
    assert body["note"]["next_step"] == "Solve p01 again"
    assert body["new_cards"] == 0


def test_today_after_close_shows_the_transition_note(client: TestClient) -> None:
    session_id = client.get("/api/v1/today").json()["current"]["session_id"]
    client.post(f"/api/v1/sessions/{session_id}/close", json=_note())
    body = client.get("/api/v1/today").json()
    assert body["closed_today"] == 1
    assert body["last_note"]["next_step"] == "Solve p01 again"
    assert body["current"] is None or body["current"]["session_id"] != session_id


def test_empty_required_note_is_rejected_by_the_service_rules(client: TestClient) -> None:
    session_id = client.get("/api/v1/today").json()["current"]["session_id"]
    response = client.post(f"/api/v1/sessions/{session_id}/close", json=_note(stopped_where="   "))
    assert response.status_code == 422
    assert response.json()["code"] == "SESSION_NOTE_INVALID"


def test_overlong_note_is_rejected(client: TestClient) -> None:
    session_id = client.get("/api/v1/today").json()["current"]["session_id"]
    response = client.post(f"/api/v1/sessions/{session_id}/close", json=_note(next_step="x" * 501))
    assert response.status_code == 422
    assert response.json()["code"] == "SESSION_NOTE_INVALID"


def test_missing_note_field_is_a_validation_error(client: TestClient) -> None:
    session_id = client.get("/api/v1/today").json()["current"]["session_id"]
    response = client.post(f"/api/v1/sessions/{session_id}/close", json={"stopped_where": "here"})
    assert response.status_code == 422
    assert response.json()["code"] == "API_VALIDATION_FAILED"


def test_unknown_session_is_not_found(client: TestClient) -> None:
    response = client.post(f"/api/v1/sessions/{'0' * 32}/close", json=_note())
    assert response.status_code == 404
    assert response.json()["code"] == "SESSION_NOT_FOUND"


def test_malformed_session_id_is_not_found(client: TestClient) -> None:
    response = client.post("/api/v1/sessions/..%2F..%2Fetc/close", json=_note())
    assert response.status_code == 404
    assert response.json()["code"] in {"SESSION_NOT_FOUND", "API_NOT_FOUND"}


def test_session_id_that_is_not_a_hex_token_is_not_found(client: TestClient) -> None:
    response = client.post("/api/v1/sessions/not-a-session/close", json=_note())
    assert response.status_code == 404
    assert response.json()["code"] == "SESSION_NOT_FOUND"
