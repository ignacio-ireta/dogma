"""POST /attempts grades the learner's workspace file and records it exactly as the CLI does."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.unit.api.conftest import FakeGrader
from tests.unit.api.support import PROBLEM, TOPIC

from dogma.core.config import Settings
from dogma.db.errors import StorageError, StorageErrorCode
from dogma.grading.models import RunStatus

URL = "/api/v1/attempts"
REFERENCE = Path(__file__).resolve().parents[3] / "content" / TOPIC / "problems"


def _workspace(settings: Settings) -> Path:
    path = settings.workspace_dir / "cml" / "vector-operations" / "p01-vector-projection.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("def vector_projection(a, b):\n    return a\n", encoding="utf-8")
    return path


def test_run_grades_the_workspace_file_and_records_it(
    settings: Settings, client: TestClient, grader: FakeGrader
) -> None:
    path = _workspace(settings)
    response = client.post(URL, json={"problem_id": PROBLEM, "mode": "run"})
    assert response.status_code == 201
    body = response.json()
    assert body["mode"] == "run"
    assert body["recorded"] is True
    assert body["record_warning"] is None
    assert body["report"]["status"] == "passed"
    assert body["report"]["problem_id"] == PROBLEM
    assert grader.calls == [(path, False)]


def test_the_created_attempt_can_be_read_back(
    settings: Settings, client: TestClient, grader: FakeGrader
) -> None:
    _workspace(settings)
    created = client.post(URL, json={"problem_id": PROBLEM, "mode": "run"}).json()
    attempt_id = created["report"]["attempt_id"]
    response = client.get(f"/api/v1/attempts/{attempt_id}")
    assert response.status_code == 200
    record = response.json()
    assert record["id"] == attempt_id
    assert record["problem_id"] == PROBLEM
    assert record["mode"] == "run"
    assert record["status"] == "passed"
    assert record["public_passed"] == 1
    assert record["public_total"] == 1
    assert record["hidden_total"] == 0
    assert record["hints_used"] == 0


def test_submit_includes_hidden_tests_and_records_submit_mode(
    settings: Settings, client: TestClient, grader: FakeGrader
) -> None:
    _workspace(settings)
    body = client.post(URL, json={"problem_id": PROBLEM, "mode": "submit"}).json()
    assert grader.calls[-1][1] is True
    record = client.get(f"/api/v1/attempts/{body['report']['attempt_id']}").json()
    assert record["mode"] == "submit"
    assert record["hidden_passed"] == 1


def test_a_failing_attempt_is_a_created_report_not_an_error(
    settings: Settings, client: TestClient, grader: FakeGrader
) -> None:
    _workspace(settings)
    grader.status = RunStatus.FAILED
    response = client.post(URL, json={"problem_id": PROBLEM, "mode": "run"})
    assert response.status_code == 201
    assert response.json()["report"]["status"] == "failed"
    assert response.json()["recorded"] is True


def test_missing_workspace_file_is_a_conflict_and_grades_nothing(
    client: TestClient, grader: FakeGrader
) -> None:
    response = client.post(URL, json={"problem_id": PROBLEM, "mode": "run"})
    assert response.status_code == 409
    assert response.json()["code"] == "API_WORKSPACE_MISSING"
    assert grader.calls == []


def test_unknown_problem_is_not_found(client: TestClient, grader: FakeGrader) -> None:
    response = client.post(
        URL, json={"problem_id": "cml/vector-operations/p99-missing", "mode": "run"}
    )
    assert response.status_code == 404
    assert response.json()["code"] == "CONTENT_NOT_FOUND"
    assert grader.calls == []


def test_grader_choice_is_not_client_controlled(
    settings: Settings, client: TestClient, grader: FakeGrader
) -> None:
    _workspace(settings)
    response = client.post(
        URL, json={"problem_id": PROBLEM, "mode": "run", "submission": "/etc/passwd"}
    )
    assert response.status_code == 422
    assert grader.calls == []


def test_unknown_mode_is_a_validation_error(
    settings: Settings, client: TestClient, grader: FakeGrader
) -> None:
    _workspace(settings)
    response = client.post(URL, json={"problem_id": PROBLEM, "mode": "record"})
    assert response.status_code == 422
    assert response.json()["code"] == "API_VALIDATION_FAILED"


def test_recording_off_keeps_the_grade_and_writes_no_row(
    settings: Settings, make_client: Callable[..., TestClient], grader: FakeGrader
) -> None:
    _workspace(settings)
    off = Settings(repo_root=settings.repo_root, record_attempts=False)
    with make_client(app_settings=off) as client:
        body = client.post(URL, json={"problem_id": PROBLEM, "mode": "run"}).json()
        assert body["recorded"] is False
        assert body["record_warning"] == "attempt recording is turned off"
        assert body["report"]["status"] == "passed"
        attempt_id = body["report"]["attempt_id"]
        assert client.get(f"/api/v1/attempts/{attempt_id}").status_code == 404
    assert not (settings.var_dir / "dogma.db").exists()


def test_a_storage_failure_keeps_the_grade(
    settings: Settings,
    client: TestClient,
    grader: FakeGrader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace(settings)

    def broken(*args: object, **kwargs: object) -> None:
        raise StorageError(StorageErrorCode.DB_WRITE_FAILED, "disk is full")

    monkeypatch.setattr("dogma.api.services.attempts.record_attempt", broken)
    response = client.post(URL, json={"problem_id": PROBLEM, "mode": "run"})
    assert response.status_code == 201
    body = response.json()
    assert body["recorded"] is False
    assert "DB_WRITE_FAILED" in body["record_warning"]
    assert body["report"]["status"] == "passed"


@pytest.mark.parametrize("ref", ["att-missing", "bad id with spaces", "..%2Fetc"])
def test_unknown_or_malformed_attempt_ids_are_not_found(client: TestClient, ref: str) -> None:
    response = client.get(f"/api/v1/attempts/{ref}")
    assert response.status_code == 404
    # An encoded slash never matches the one-segment route, so routing answers it instead.
    assert response.json()["code"] in {"API_ATTEMPT_NOT_FOUND", "API_NOT_FOUND"}


@pytest.mark.integration
def test_real_grader_passes_the_reference_solution(settings: Settings, client: TestClient) -> None:
    """The real sandboxed runner on the reference solution, copied into a scratch workspace."""
    solution = (REFERENCE / "p01-vector-projection" / "solution.py").read_text(encoding="utf-8")
    path = settings.workspace_dir / "cml" / "vector-operations" / "p01-vector-projection.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(solution, encoding="utf-8")
    response = client.post(URL, json={"problem_id": PROBLEM, "mode": "submit"})
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["report"]["status"] == "passed"
    assert body["recorded"] is True
    progress = client.get("/api/v1/progress").json()
    assert progress["summary"]["solved_topics"] == 1
