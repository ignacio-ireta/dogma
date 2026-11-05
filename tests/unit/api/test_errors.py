"""RFC 9457 problem bodies: the domain-code to status table and the handlers that use it."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dogma.api.errors import STATUS_BY_CODE, problem_for_code, status_for
from dogma.core.errors import ErrorCode
from dogma.db.errors import StorageErrorCode
from dogma.runner import attempt


@pytest.mark.parametrize(
    ("code", "status"),
    [
        (ErrorCode.CONTENT_NOT_FOUND, 404),
        (ErrorCode.REVIEW_NOT_FOUND, 404),
        (ErrorCode.SESSION_NOT_FOUND, 404),
        (ErrorCode.API_ATTEMPT_NOT_FOUND, 404),
        (ErrorCode.REVIEW_STALE, 409),
        (ErrorCode.REVIEW_NOT_DUE, 409),
        (ErrorCode.TUTOR_NO_FAILURE, 409),
        (ErrorCode.API_WORKSPACE_MISSING, 409),
        (ErrorCode.TUTOR_LOCKED, 403),
        (ErrorCode.REVIEW_RATING_INVALID, 422),
        (ErrorCode.SESSION_NOTE_INVALID, 422),
        (ErrorCode.CORPUS_QUERY_INVALID, 422),
        (ErrorCode.LLM_UNAVAILABLE, 503),
        (ErrorCode.CORPUS_INDEX_MISSING, 503),
        (ErrorCode.RUNNER_RUNTIME_UNAVAILABLE, 503),
        (ErrorCode.LLM_RESPONSE_INVALID, 502),
        (ErrorCode.RUNNER_TIMEOUT, 500),
        (ErrorCode.CONTENT_SCHEMA_INVALID, 500),
        (ErrorCode.RESOLVE_NOT_CORE, 422),
        (ErrorCode.RESOLVE_NOT_READY, 409),
        (ErrorCode.RESOLVE_NOT_SOLVED, 409),
        (ErrorCode.RESOLVE_ALREADY_OPEN, 409),
        (ErrorCode.RESOLVE_RECORDING_OFF, 409),
        (ErrorCode.RESOLVE_NOT_FOUND, 404),
    ],
)
def test_domain_codes_map_to_their_status(code: ErrorCode, status: int) -> None:
    assert status_for(code) == status


def test_every_domain_code_has_a_deliberate_status() -> None:
    # status_for falls back to 500, so the table itself is what must list every code.
    for code in [*ErrorCode, *StorageErrorCode]:
        assert code.value in STATUS_BY_CODE, code
        assert 400 <= STATUS_BY_CODE[code.value] <= 599, code


def test_storage_codes_map_to_status() -> None:
    assert status_for(StorageErrorCode.DB_UNAVAILABLE) == 503
    assert status_for(StorageErrorCode.DB_MIGRATION_FAILED) == 503
    assert status_for(StorageErrorCode.DB_READ_FAILED) == 500


def test_problem_body_carries_the_code_and_message() -> None:
    problem = problem_for_code(
        "CONTENT_NOT_FOUND", "no problem 'x'", instance="/api/v1/problems/x", status=404
    )
    assert problem.type == "urn:dogma:error:CONTENT_NOT_FOUND"
    assert problem.title == "Not Found"
    assert problem.status == 404
    assert problem.detail == "no problem 'x'"
    assert problem.instance == "/api/v1/problems/x"
    assert problem.code == "CONTENT_NOT_FOUND"


def test_validation_errors_name_the_field_and_do_not_echo_input(client: TestClient) -> None:
    response = client.get("/api/v1/today", params={"energy": "loud"})
    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert body["code"] == "API_VALIDATION_FAILED"
    fields = [item["field"] for item in body["errors"]]
    assert any(field.endswith("energy") for field in fields)
    assert "loud" not in response.text


def test_unexpected_failure_is_a_500_without_a_traceback(
    make_client: Callable[..., TestClient],
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(attempt, "grade_attempt", explode)
    workspace = repo / "workspace" / "cml" / "vector-operations" / "p01-vector-projection.py"
    workspace.parent.mkdir(parents=True)
    workspace.write_text("def vector_projection(a, b):\n    return a\n", encoding="utf-8")
    with make_client(raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/attempts",
            json={"problem_id": "cml/vector-operations/p01-vector-projection", "mode": "run"},
        )
    assert response.status_code == 500
    body = response.json()
    assert body["code"] == "API_INTERNAL_ERROR"
    assert "secret internal detail" not in response.text
    assert "Traceback" not in response.text
