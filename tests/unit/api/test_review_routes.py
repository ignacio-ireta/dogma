"""Review queue and ratings: the same stale and not-due rules as `dogma review`."""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from urllib.parse import quote

from fastapi.testclient import TestClient
from tests.unit.api.support import TOPIC, FrozenClock


def _close_learn_session(client: TestClient) -> None:
    session_id = client.get("/api/v1/today").json()["current"]["session_id"]
    client.post(
        f"/api/v1/sessions/{session_id}/close",
        json={"stopped_where": "Concept", "next_step": "Practice"},
    )


def _due(client: TestClient, limit: int = 20) -> list[dict[str, Any]]:
    body: dict[str, Any] = client.get("/api/v1/reviews/due", params={"limit": limit}).json()
    cards: list[dict[str, Any]] = body["cards"]
    return cards


def _rate(client: TestClient, card: dict[str, Any], rating: int = 3) -> Any:
    return client.post(
        f"/api/v1/reviews/{quote(card['card_id'], safe='')}",
        json={
            "rating": rating,
            "expected_due": card["due"],
            "expected_content_hash": card["content_hash"],
        },
    )


def test_queue_is_empty_before_any_topic_is_studied(client: TestClient) -> None:
    body = client.get("/api/v1/reviews/due").json()
    assert body["cards"] == []
    assert body["enrolled"] == 0


def test_closing_a_learn_session_enrols_the_topic_cards_into_the_queue(client: TestClient) -> None:
    _close_learn_session(client)
    cards = _due(client)
    assert cards
    first = cards[0]
    assert first["card_id"].startswith(f"{TOPIC}#")
    assert first["topic_id"] == TOPIC
    assert first["question"]
    assert first["answer"]
    assert len(first["content_hash"]) == 64


def test_queue_respects_the_limit(client: TestClient) -> None:
    _close_learn_session(client)
    assert len(_due(client, limit=2)) == 2


def test_rating_reschedules_the_card(client: TestClient) -> None:
    _close_learn_session(client)
    card = _due(client)[0]
    response = _rate(client, card, rating=3)
    assert response.status_code == 200
    body = response.json()
    assert body["card_id"] == card["card_id"]
    assert body["rating"] == 3
    assert body["due"] > card["due"]
    assert body["mastery"] in {"introduced", "consolidating", "solved"}


def test_stale_rating_is_a_conflict(client: TestClient) -> None:
    _close_learn_session(client)
    card = _due(client)[0]
    assert _rate(client, card).status_code == 200
    response = _rate(client, card)
    assert response.status_code == 409
    assert response.json()["code"] == "REVIEW_STALE"


def test_changed_content_is_stale(client: TestClient) -> None:
    _close_learn_session(client)
    card = _due(client)[0]
    card = {**card, "content_hash": "0" * 64}
    response = _rate(client, card)
    assert response.status_code == 409
    assert response.json()["code"] == "REVIEW_STALE"


def test_rating_before_the_card_is_due_is_a_conflict(
    client: TestClient, clock: FrozenClock
) -> None:
    _close_learn_session(client)
    card = _due(client)[0]
    clock.now = clock.now - timedelta(hours=1)
    response = _rate(client, card)
    assert response.status_code == 409
    assert response.json()["code"] == "REVIEW_NOT_DUE"


def test_unknown_card_is_not_found(client: TestClient) -> None:
    card = {
        "card_id": f"{TOPIC}#999",
        "due": "2026-10-26T09:00:00Z",
        "content_hash": "0" * 64,
    }
    response = _rate(client, card)
    assert response.status_code == 404
    assert response.json()["code"] == "REVIEW_NOT_FOUND"


def test_malformed_card_id_is_not_found(client: TestClient) -> None:
    response = client.post(
        "/api/v1/reviews/..%2F..%2Fetc",
        json={"rating": 3, "expected_due": "2026-10-26T09:00:00Z", "expected_content_hash": "x"},
    )
    assert response.status_code == 404
    assert response.json()["code"] == "REVIEW_NOT_FOUND"


def test_rating_outside_one_to_four_is_a_validation_error(client: TestClient) -> None:
    _close_learn_session(client)
    card = _due(client)[0]
    response = _rate(client, card, rating=5)
    assert response.status_code == 422
    assert response.json()["code"] == "API_VALIDATION_FAILED"


def test_expected_due_must_carry_a_timezone(client: TestClient) -> None:
    _close_learn_session(client)
    card = _due(client)[0]
    response = client.post(
        f"/api/v1/reviews/{quote(card['card_id'], safe='')}",
        json={"rating": 3, "expected_due": "2026-10-26T09:00:00", "expected_content_hash": "x"},
    )
    assert response.status_code == 422


def test_limit_outside_bounds_is_a_validation_error(client: TestClient) -> None:
    assert client.get("/api/v1/reviews/due", params={"limit": 0}).status_code == 422
    assert client.get("/api/v1/reviews/due", params={"limit": 101}).status_code == 422
