"""One learner day over HTTP with the real grader: catalog, edit, submit, today, close, review.

Runs in a scratch repo copy (never the real var/ or workspace/). Marked e2e: `make test-all`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.unit.api.support import PROBLEM, TOPIC, FrozenClock, copy_repo

from dogma.api.app import create_app
from dogma.api.deps import get_clock
from dogma.core.config import Settings

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

REFERENCE = (
    Path(__file__).resolve().parents[3] / "content" / TOPIC / "problems" / "p01-vector-projection"
)


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    settings = Settings(repo_root=copy_repo(tmp_path))
    app = create_app(settings)
    clock = FrozenClock()
    app.dependency_overrides[get_clock] = lambda: clock
    return TestClient(app, base_url="http://127.0.0.1")


def test_a_learner_day_over_http(client: TestClient) -> None:
    catalog = client.get("/api/v1/catalog").json()
    assert any(topic["id"] == TOPIC for topic in catalog["topics"])

    problem = client.get(f"/api/v1/problems/{PROBLEM}").json()
    assert problem["starter"]

    solution = (REFERENCE / "solution.py").read_text(encoding="utf-8")
    saved = client.put(f"/api/v1/workspace/{PROBLEM}", json={"content": solution})
    assert saved.json()["exists"] is True

    submitted = client.post("/api/v1/attempts", json={"problem_id": PROBLEM, "mode": "submit"})
    assert submitted.status_code == 201, submitted.text
    assert submitted.json()["report"]["status"] == "passed"

    today = client.get("/api/v1/today").json()
    session_id = today["current"]["session_id"]
    closed = client.post(
        f"/api/v1/sessions/{session_id}/close",
        json={"stopped_where": "Practice done", "next_step": "Review cards"},
    )
    assert closed.status_code == 200

    due = client.get("/api/v1/reviews/due").json()["cards"]
    assert due
    card = due[0]
    rated = client.post(
        f"/api/v1/reviews/{card['card_id'].replace('#', '%23')}",
        json={
            "rating": 3,
            "expected_due": card["due"],
            "expected_content_hash": card["content_hash"],
        },
    )
    assert rated.status_code == 200
    assert rated.json()["due"] > card["due"]

    progress = client.get("/api/v1/progress").json()
    assert progress["summary"]["solved_topics"] == 1
