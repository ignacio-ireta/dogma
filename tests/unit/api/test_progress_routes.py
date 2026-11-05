"""GET /progress: the same weekly view and summary rules as `dogma progress`, read-only."""

from __future__ import annotations

from datetime import timedelta

from fastapi.testclient import TestClient
from tests.unit.api.conftest import FakeGrader
from tests.unit.api.support import PROBLEM, TOPIC, FrozenClock

from dogma.core.config import Settings

WORKSPACE_URL = f"/api/v1/workspace/{PROBLEM}"
ATTEMPTS = "/api/v1/attempts"


def _submit_passing(settings: Settings, client: TestClient, grader: FakeGrader) -> None:
    path = settings.workspace_dir / "cml" / "vector-operations" / "p01-vector-projection.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("def vector_projection(a, b):\n    return a\n", encoding="utf-8")
    client.post(ATTEMPTS, json={"problem_id": PROBLEM, "mode": "submit"})
    assert grader.calls


def test_empty_progress_has_every_week_and_no_solved_topics(
    settings: Settings, client: TestClient
) -> None:
    body = client.get("/api/v1/progress").json()
    assert body["course"]["id"] == "core"
    assert body["summary"]["solved_topics"] == 0
    assert body["summary"]["attempts_this_week"] == 0
    assert body["summary"]["current_week"] == 1
    assert body["summary"]["week_start"] == "2026-10-26"
    assert body["summary"]["total_topics"] > 0
    first = body["weeks"][0]
    assert first["week"] == 1
    row = next(topic for topic in first["topics"] if topic["topic_id"] == TOPIC)
    assert row["mastery"] == "unseen"
    assert row["attempts"] == 0
    assert row["best"] is None
    assert row["last_attempt_at"] is None


def test_reading_progress_creates_no_database(settings: Settings, client: TestClient) -> None:
    client.get("/api/v1/progress")
    assert not (settings.var_dir / "dogma.db").exists()


def test_a_passing_core_submit_marks_the_topic_solved(
    settings: Settings, client: TestClient, grader: FakeGrader
) -> None:
    _submit_passing(settings, client, grader)
    body = client.get("/api/v1/progress").json()
    assert body["summary"]["solved_topics"] == 1
    assert body["summary"]["attempts_this_week"] == 1
    row = next(topic for topic in body["weeks"][0]["topics"] if topic["topic_id"] == TOPIC)
    assert row["mastery"] == "solved"
    assert row["solved"] is True
    assert row["attempts"] == 1
    assert row["best"] == {"passed": 2, "total": 2, "mode": "submit"}
    assert row["last_attempt_at"] is not None


def test_week_filter_returns_only_that_week(client: TestClient) -> None:
    body = client.get("/api/v1/progress", params={"week": 1}).json()
    assert [week["week"] for week in body["weeks"]] == [1]


def test_summary_covers_the_whole_path_even_when_filtered(
    settings: Settings, client: TestClient, grader: FakeGrader
) -> None:
    _submit_passing(settings, client, grader)
    filtered = client.get("/api/v1/progress", params={"week": 1}).json()
    assert filtered["summary"]["solved_topics"] == 1


def test_week_outside_the_path_is_not_found(client: TestClient) -> None:
    response = client.get("/api/v1/progress", params={"week": 999})
    assert response.status_code == 404
    assert response.json()["code"] == "CONTENT_NOT_FOUND"


def test_week_must_be_positive(client: TestClient) -> None:
    response = client.get("/api/v1/progress", params={"week": 0})
    assert response.status_code == 422


def test_reading_progress_leaves_the_workspace_alone(client: TestClient) -> None:
    client.get("/api/v1/progress")
    assert client.get(WORKSPACE_URL).json()["exists"] is False


def test_readiness_is_empty_before_any_timed_pass(client: TestClient) -> None:
    readiness = client.get("/api/v1/progress").json()["readiness"]
    # interview_ready (I-18e) is additive; the timed-re-solve part is unchanged.
    assert {key: readiness[key] for key in ("next_due", "recent_results")} == {
        "next_due": [],
        "recent_results": [],
    }


def test_readiness_lists_the_next_due_and_the_last_timed_results(
    settings: Settings, client: TestClient, clock: FrozenClock, grader: FakeGrader
) -> None:
    _submit_passing(settings, client, grader)
    assert client.post("/api/v1/resolves", json={"problem_id": PROBLEM}).status_code == 201
    clock.now = clock.now + timedelta(minutes=5)
    client.post(ATTEMPTS, json={"problem_id": PROBLEM, "mode": "submit"})
    readiness = client.get("/api/v1/progress").json()["readiness"]
    assert [item["problem_id"] for item in readiness["next_due"]] == [PROBLEM]
    due = readiness["next_due"][0]
    assert due["stage"] == 1
    assert due["due_now"] is False
    assert "reason" in due
    result = readiness["recent_results"][0]
    assert result["problem_id"] == PROBLEM
    assert result["status"] == "passed"
    assert result["used_s"] == 300
    assert result["budget_s"] == 1200
