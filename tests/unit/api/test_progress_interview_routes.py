"""GET /progress adds interview-ready counts to readiness: additive, counts only (ADR-0013)."""

from __future__ import annotations

from fastapi.testclient import TestClient
from tests.unit.api.support import NOW, TOPIC, FrozenClock

from dogma.core.config import Settings
from dogma.db.errors import StorageErrorCode
from dogma.db.models import Mastery, TopicState
from dogma.db.session import open_session

ANSWERS = f"/api/v1/topics/{TOPIC}/answers"
DIMENSIONS = ("accuracy", "structure", "trade-offs", "communication")


def test_readiness_counts_zero_interview_ready_topics_when_nothing_is_retained(
    client: TestClient,
) -> None:
    ready = client.get("/api/v1/progress").json()["readiness"]["interview_ready"]
    assert ready["ready"] == 0
    assert ready["total"] > 0
    assert len(ready["weeks"]) >= 1
    assert all(item["ready"] == 0 for item in ready["weeks"])
    assert ready["weeks"][0]["week"] == 1


def test_a_qualifying_answer_on_a_retained_topic_counts_in_its_week(
    settings: Settings, client: TestClient, clock: FrozenClock
) -> None:
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        session.add(TopicState(topic_id=TOPIC, mastery=Mastery.RETAINED.value, retained_at=NOW))
    clock.now = NOW
    scores = dict.fromkeys(DIMENSIONS, 3)
    assert client.post(ANSWERS, json={"text": "Alignment.", "scores": scores}).status_code == 201
    ready = client.get("/api/v1/progress").json()["readiness"]["interview_ready"]
    assert ready["ready"] == 1
    week_one = next(item for item in ready["weeks"] if item["week"] == 1)
    assert week_one["ready"] == 1
    assert all(item["ready"] == 0 for item in ready["weeks"][1:])
