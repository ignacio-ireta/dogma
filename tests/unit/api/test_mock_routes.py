"""Mock interviews over HTTP: start, plan, reveal, score, debrief and abandon.

The rules are the CLI's (`dogma mock`), tested in tests/unit/learning/test_mocks.py. These tests
check the wire shapes and the status codes, on a scratch repo that holds the sys-track cases.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from dogma.core.config import Settings

REPO = Path(__file__).resolve().parents[3]
MOCKS = "/api/v1/mocks"
CURRENT = f"{MOCKS}/current"
REVEAL = f"{CURRENT}/reveal"
SCORE = f"{CURRENT}/score"
ABANDON = f"{CURRENT}/abandon"
CHURN = "sys/churn-prediction-case"
MISSING_ID = "0" * 32
NOTHING = "The interviewer has nothing on that."


@pytest.fixture(autouse=True)
def _cases(repo: Path) -> None:
    """The mock needs the sys-track cases, which the default scratch repo does not copy."""
    shutil.copytree(REPO / "content" / "sys", repo / "content" / "sys")


def _items(client: TestClient) -> list[str]:
    rubric = client.get("/api/v1/rubrics/mock").json()
    return list(rubric["criteria"])


def _start(client: TestClient, **payload: object) -> dict[str, object]:
    response = client.post(MOCKS, json=payload)
    assert response.status_code == 201, response.text
    body: dict[str, object] = response.json()
    return body


def _score_all(client: TestClient, score: int) -> dict[str, object]:
    response = client.post(SCORE, json={"scores": dict.fromkeys(_items(client), score)})
    assert response.status_code == 201, response.text
    body: dict[str, object] = response.json()
    return body


def test_start_picks_a_case_with_its_brief_and_the_part_timings(client: TestClient) -> None:
    plan = _start(client)
    case = plan["case"]
    assert isinstance(case, dict)
    assert case["id"] == CHURN
    assert case["brief"]
    assert plan["coding"] is None
    assert plan["theory"] is None
    assert plan["focus"] is None
    parts = plan["parts"]
    assert isinstance(parts, list)
    assert next(part["title"] for part in parts) == "Live coding"
    assert all(part["minutes"] for part in parts)


def test_the_current_mock_is_the_one_started(client: TestClient) -> None:
    plan = _start(client)
    status = client.get(CURRENT).json()
    assert status["open"]["session_id"] == plan["session_id"]
    assert status["last_scored"] is None


def test_reading_the_current_mock_with_no_database_creates_none(
    settings: Settings, client: TestClient
) -> None:
    body = client.get(CURRENT).json()
    assert body == {"open": None, "last_scored": None}
    assert not (settings.var_dir / "dogma.db").exists()


def test_a_second_start_while_one_is_open_is_refused(client: TestClient) -> None:
    _start(client)
    response = client.post(MOCKS, json={})
    assert response.status_code == 409
    assert response.json()["code"] == "MOCK_ALREADY_OPEN"


def test_an_unknown_case_is_not_found(client: TestClient) -> None:
    response = client.post(MOCKS, json={"case_id": "sys/no-such-case"})
    assert response.status_code == 404
    assert response.json()["code"] == "CONTENT_NOT_FOUND"


def test_start_body_is_strict(client: TestClient) -> None:
    response = client.post(MOCKS, json={"minutes": 40})
    assert response.status_code == 422
    assert response.json()["code"] == "API_VALIDATION_FAILED"


def test_a_reveal_is_released_only_for_a_matching_question(repo: Path, client: TestClient) -> None:
    _start(client, case_id=CHURN)
    reveals = yaml.safe_load((repo / "content" / CHURN / "reveals.yaml").read_text("utf-8"))
    matching = reveals[0]
    hit = client.post(REVEAL, json={"question": matching["when_asked"]}).json()
    assert hit["matched"] is True
    assert hit["text"] == matching["answer"]
    miss = client.post(REVEAL, json={"question": "What is the weather on Mars?"}).json()
    assert miss == {"text": NOTHING, "matched": False, "exhibit": None}


def test_a_reveal_with_no_open_mock_is_refused(client: TestClient) -> None:
    response = client.post(REVEAL, json={"question": "What are the competitors doing?"})
    assert response.status_code == 409
    assert response.json()["code"] == "MOCK_NOT_OPEN"


def test_an_empty_reveal_question_is_a_validation_error(client: TestClient) -> None:
    _start(client, case_id=CHURN)
    response = client.post(REVEAL, json={"question": ""})
    assert response.status_code == 422
    assert response.json()["code"] == "API_VALIDATION_FAILED"


@pytest.mark.parametrize(
    ("payload", "code", "status"),
    [
        ("missing", "MOCK_SCORE_INVALID", 422),
        ("out-of-range", "MOCK_SCORE_INVALID", 422),
        ("text", "API_VALIDATION_FAILED", 422),
        ("unknown", "MOCK_SCORE_INVALID", 422),
    ],
)
def test_bad_scores_store_nothing_and_leave_the_mock_open(
    client: TestClient, payload: str, code: str, status: int
) -> None:
    _start(client, case_id=CHURN)
    items = _items(client)
    scores: dict[str, object] = dict.fromkeys(items, 3)
    if payload == "missing":
        del scores[items[0]]
    elif payload == "out-of-range":
        scores[items[0]] = 5
    elif payload == "text":
        scores[items[0]] = "3"
    else:
        scores["not.an-item"] = 3
    response = client.post(SCORE, json={"scores": scores})
    assert response.status_code == status, response.text
    assert response.json()["code"] == code
    assert client.get(CURRENT).json()["open"] is not None


def test_scoring_closes_the_mock_and_returns_the_report(client: TestClient) -> None:
    plan = _start(client, case_id=CHURN)
    report = _score_all(client, 3)
    assert report["session_id"] == plan["session_id"]
    assert report["total"] == 90  # thirty items at 3
    assert report["maximum"] == client.get("/api/v1/rubrics/mock").json()["maximum"]
    band = report["band"]
    assert isinstance(band, dict)
    assert band["low"] <= 90 <= band["high"]
    parts = report["parts"]
    assert isinstance(parts, list)
    assert len(parts) == 6
    assert all(0 <= part["percent"] <= 100 for part in parts)
    status = client.get(CURRENT).json()
    assert status["open"] is None
    assert status["last_scored"]["session_id"] == plan["session_id"]


def test_the_debrief_shows_the_report_and_the_case_debrief(client: TestClient) -> None:
    plan = _start(client, case_id=CHURN)
    _score_all(client, 4)
    body = client.get(f"{MOCKS}/{plan['session_id']}/debrief").json()
    assert body["case_id"] == CHURN
    assert body["report"]["total"] == 120
    assert body["case_debrief"]
    assert body["report"]["strongest"]["percent"] == 100.0


def test_a_debrief_for_an_unknown_or_malformed_id_is_not_found(client: TestClient) -> None:
    _start(client, case_id=CHURN)
    for session_id in (MISSING_ID, "not-an-id"):
        response = client.get(f"{MOCKS}/{session_id}/debrief")
        assert response.status_code == 404
        assert response.json()["code"] == "MOCK_NOT_FOUND"


def test_an_open_mock_has_no_debrief_yet(client: TestClient) -> None:
    plan = _start(client, case_id=CHURN)
    response = client.get(f"{MOCKS}/{plan['session_id']}/debrief")
    assert response.status_code == 404
    assert response.json()["code"] == "MOCK_NOT_FOUND"


def test_abandon_closes_the_open_mock_without_scores(client: TestClient) -> None:
    plan = _start(client, case_id=CHURN)
    response = client.post(ABANDON)
    assert response.status_code == 200
    assert response.json() == {"session_id": plan["session_id"], "status": "abandoned"}
    status = client.get(CURRENT).json()
    assert status["open"] is None
    assert status["last_scored"] is None


def test_abandon_with_nothing_open_is_refused(client: TestClient) -> None:
    response = client.post(ABANDON)
    assert response.status_code == 409
    assert response.json()["code"] == "MOCK_NOT_OPEN"


def test_recording_switched_off_refuses_a_mock(
    repo: Path, make_client: Callable[..., TestClient]
) -> None:
    off = Settings(repo_root=repo, record_attempts=False)
    with make_client(app_settings=off) as client:
        response = client.post(MOCKS, json={"case_id": CHURN})
    assert response.status_code == 409
    assert response.json()["code"] == "MOCK_RECORDING_OFF"


def test_a_foreign_origin_is_refused_on_a_mock_write(client: TestClient) -> None:
    response = client.post(
        MOCKS, json={"case_id": CHURN}, headers={"Origin": "https://evil.example"}
    )
    assert response.status_code == 403
    assert response.json()["code"] == "API_REQUEST_REFUSED"
