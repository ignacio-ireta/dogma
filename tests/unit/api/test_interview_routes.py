"""Interview questions, rubrics and self-scored answers over HTTP (`dogma answer`).

Each test checks the wire shape and the status. The mastery rule itself is tested in
tests/unit/learning/test_answers.py, and the routes here call the same service.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.unit.api.support import NOW, TOPIC, FrozenClock

from dogma.core.config import Settings
from dogma.db.errors import StorageErrorCode
from dogma.db.models import Mastery, TopicState
from dogma.db.session import open_session

INTERVIEW = f"/api/v1/topics/{TOPIC}/interview"
ANSWERS = f"/api/v1/topics/{TOPIC}/answers"
TOPIC_URL = f"/api/v1/topics/{TOPIC}"
DIMENSIONS = ("accuracy", "structure", "trade-offs", "communication")


def _all(score: int) -> dict[str, int]:
    return dict.fromkeys(DIMENSIONS, score)


def _retain(settings: Settings, clock: FrozenClock) -> None:
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        session.add(TopicState(topic_id=TOPIC, mastery=Mastery.RETAINED.value, retained_at=NOW))
    clock.now = NOW


def test_the_interview_lists_question_titles_without_their_answers(client: TestClient) -> None:
    body = client.get(INTERVIEW).json()
    assert body["topic_id"] == TOPIC
    assert body["title"]
    assert body["mastery"] == "unseen"
    assert [item["number"] for item in body["questions"]] == [1, 2, 3]
    assert all(set(item) == {"number", "title"} for item in body["questions"])
    assert body["questions"][0]["title"].startswith("What does the dot product")


def test_the_interview_carries_the_topic_answer_rubric_with_anchors(client: TestClient) -> None:
    rubric = client.get(INTERVIEW).json()["rubric"]
    assert rubric["id"] == "topic-answer"
    assert rubric["ready_score"] == 3
    assert rubric["maximum"] == 16
    assert rubric["criteria"] == list(DIMENSIONS)
    for dimension in rubric["dimensions"]:
        levels = [anchor["level"] for anchor in dimension["anchors"]]
        assert levels == [1, 2, 3, 4]
        assert all(anchor["text"] for anchor in dimension["anchors"])


def test_a_topic_without_questions_is_not_found(repo: Path, client: TestClient) -> None:
    (repo / "content" / TOPIC / "interview.md").unlink()
    response = client.get(INTERVIEW)
    assert response.status_code == 404
    assert response.json()["code"] == "CONTENT_NOT_FOUND"


def test_an_unknown_topic_is_not_found(client: TestClient) -> None:
    response = client.get("/api/v1/topics/cml/no-such-topic/interview")
    assert response.status_code == 404
    assert response.json()["code"] == "CONTENT_NOT_FOUND"


def test_the_topic_route_still_answers_the_topic_not_the_interview(client: TestClient) -> None:
    body = client.get(TOPIC_URL).json()
    assert body["id"] == TOPIC
    assert "questions" not in body


def test_a_written_answer_is_stored_and_says_what_it_did(client: TestClient) -> None:
    response = client.post(
        ANSWERS,
        json={"question": 1, "text": "The dot product measures alignment.", "scores": _all(3)},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["topic_id"] == TOPIC
    assert body["prompt_ref"] == "interview.md#Q1"
    assert body["spoken"] is False
    assert body["scores"] == _all(3)
    assert body["before"] == "unseen"
    assert body["after"] == "unseen"
    assert body["promoted"] is False
    assert body["ready_at"] is None
    assert "not started" in body["reason"].lower()


def test_a_spoken_answer_keeps_no_text_and_still_scores(client: TestClient) -> None:
    response = client.post(ANSWERS, json={"question": 2, "spoken": True, "scores": _all(2)})
    assert response.status_code == 201
    assert response.json()["spoken"] is True
    assert response.json()["prompt_ref"] == "interview.md#Q2"


def test_a_retained_topic_becomes_interview_ready_on_a_qualifying_answer(
    settings: Settings, client: TestClient, clock: FrozenClock
) -> None:
    _retain(settings, clock)
    response = client.post(ANSWERS, json={"text": "Alignment, scaled.", "scores": _all(3)})
    body = response.json()
    assert body["before"] == "retained"
    assert body["after"] == "interview-ready"
    assert body["promoted"] is True
    assert body["ready_at"] is not None
    assert "Interview-ready" in body["reason"]
    assert client.get(INTERVIEW).json()["mastery"] == "interview-ready"


def test_a_low_dimension_keeps_a_retained_topic_retained(
    settings: Settings, client: TestClient, clock: FrozenClock
) -> None:
    _retain(settings, clock)
    scores = {**_all(4), "trade-offs": 2}
    body = client.post(ANSWERS, json={"text": "Alignment.", "scores": scores}).json()
    assert body["after"] == "retained"
    assert body["promoted"] is False
    assert "trade-offs" in body["reason"]


@pytest.mark.parametrize(
    ("payload", "code", "status"),
    [
        ({"text": "x", "spoken": True, "scores": _all(3)}, "ANSWER_TEXT_INVALID", 422),
        ({"scores": _all(3)}, "ANSWER_TEXT_INVALID", 422),
        ({"text": "   ", "scores": _all(3)}, "ANSWER_TEXT_INVALID", 422),
        ({"text": "x", "scores": {"accuracy": 3}}, "ANSWER_SCORE_INVALID", 422),
        ({"text": "x", "scores": {**_all(3), "accuracy": 5}}, "ANSWER_SCORE_INVALID", 422),
        ({"text": "x", "scores": {**_all(3), "fluency": 3}}, "ANSWER_SCORE_INVALID", 422),
        ({"text": "x", "scores": {**_all(3), "accuracy": "3"}}, "API_VALIDATION_FAILED", 422),
        ({"text": "x", "scores": {**_all(3), "accuracy": True}}, "API_VALIDATION_FAILED", 422),
        ({"text": "x", "scores": _all(3), "extra": 1}, "API_VALIDATION_FAILED", 422),
        ({"text": "x", "question": 9, "scores": _all(3)}, "CONTENT_NOT_FOUND", 404),
    ],
)
def test_bad_answers_are_refused_with_a_domain_code(
    client: TestClient, payload: dict[str, object], code: str, status: int
) -> None:
    response = client.post(ANSWERS, json=payload)
    assert response.status_code == status, response.text
    assert response.json()["code"] == code


def test_an_unknown_topic_cannot_take_an_answer(client: TestClient) -> None:
    response = client.post(
        "/api/v1/topics/cml/no-such-topic/answers", json={"text": "x", "scores": _all(3)}
    )
    assert response.status_code == 404
    assert response.json()["code"] == "CONTENT_NOT_FOUND"


def test_recording_switched_off_refuses_an_answer(
    repo: Path, make_client: Callable[..., TestClient]
) -> None:
    off = Settings(repo_root=repo, record_attempts=False)
    with make_client(app_settings=off) as client:
        response = client.post(ANSWERS, json={"text": "x", "scores": _all(3)})
    assert response.status_code == 409
    assert response.json()["code"] == "ANSWER_RECORDING_OFF"


def test_the_mock_rubric_lists_thirty_items_in_six_parts(client: TestClient) -> None:
    rubric = client.get("/api/v1/rubrics/mock").json()
    assert rubric["id"] == "mock"
    assert rubric["kind"] == "mock"
    assert len(rubric["criteria"]) == 30
    assert [part["id"] for part in rubric["parts"]] == [
        "coding",
        "reframe",
        "ask",
        "structure",
        "synthesize",
        "communication",
    ]
    assert rubric["bands"][0]["low"] == 0
    assert rubric["bands"][-1]["high"] == rubric["maximum"]
    assert len(rubric["anchors"]) == 4


def test_an_unknown_rubric_is_not_found(client: TestClient) -> None:
    response = client.get("/api/v1/rubrics/nope")
    assert response.status_code == 404
    assert response.json()["code"] == "CONTENT_NOT_FOUND"
