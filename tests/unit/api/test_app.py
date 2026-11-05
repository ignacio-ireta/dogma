"""The app shell: routing prefix, OpenAPI contract, Host/Origin guards, CORS and body limits."""

from __future__ import annotations

from collections.abc import Callable

from fastapi.testclient import TestClient
from tests.unit.api.support import ORIGIN

from dogma.api.app import API_PREFIX, create_app
from dogma.core.config import Settings

EXPECTED_PATHS = [
    "/api/v1/attempts",
    "/api/v1/attempts/{attempt_id}",
    "/api/v1/calendar",
    "/api/v1/calendar/warp",
    "/api/v1/catalog",
    "/api/v1/course/reset",
    "/api/v1/coverage",
    "/api/v1/health",
    "/api/v1/mocks",
    "/api/v1/mocks/current",
    "/api/v1/mocks/current/abandon",
    "/api/v1/mocks/current/reveal",
    "/api/v1/mocks/current/score",
    "/api/v1/mocks/{session_id}/debrief",
    "/api/v1/problems/{problem_id}",
    "/api/v1/progress",
    "/api/v1/resolves",
    "/api/v1/resolves/current",
    "/api/v1/resolves/current/abandon",
    "/api/v1/reviews/due",
    "/api/v1/reviews/{card_id}",
    "/api/v1/rubrics/{rubric_id}",
    "/api/v1/search",
    "/api/v1/sessions/{session_id}/close",
    "/api/v1/today",
    "/api/v1/topics/{topic_id}",
    "/api/v1/topics/{topic_id}/answers",
    "/api/v1/topics/{topic_id}/interview",
    "/api/v1/tutor/debug",
    "/api/v1/tutor/debug/stream",
    "/api/v1/tutor/hint",
    "/api/v1/tutor/hint/stream",
    "/api/v1/workspace/{problem_id}",
]


def test_prefix_is_v1() -> None:
    assert API_PREFIX == "/api/v1"


def test_openapi_path_list_is_stable(settings: Settings) -> None:
    schema = create_app(settings).openapi()
    assert sorted(schema["paths"]) == EXPECTED_PATHS


def test_openapi_is_deterministic(settings: Settings) -> None:
    first = create_app(settings).openapi()
    second = create_app(settings).openapi()
    assert first == second


def test_openapi_is_served_under_api_prefix(client: TestClient) -> None:
    response = client.get("/api/v1/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"]["title"] == "dogma API"


def test_no_interactive_docs_that_load_cdn_assets(client: TestClient) -> None:
    # Swagger UI and ReDoc fetch their assets from a CDN; the app must work offline.
    assert client.get("/api/v1/docs").status_code == 404
    assert client.get("/docs").status_code == 404


def test_health_reports_ok(client: TestClient) -> None:
    body = client.get("/api/v1/health").json()
    assert body == {"status": "ok", "version": "0.1.0"}


def test_unknown_route_is_problem_json(client: TestClient) -> None:
    response = client.get("/api/v1/nope")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert body["code"] == "API_NOT_FOUND"
    assert body["status"] == 404
    assert body["instance"] == "/api/v1/nope"


def test_wrong_method_is_problem_json(client: TestClient) -> None:
    response = client.delete("/api/v1/health")
    assert response.status_code == 405
    assert response.json()["code"] == "API_METHOD_NOT_ALLOWED"


def test_foreign_host_header_is_refused(client: TestClient) -> None:
    # DNS rebinding: a page on another name must not reach the loopback API.
    response = client.get("/api/v1/health", headers={"Host": "evil.example"})
    assert response.status_code == 403
    assert response.json()["code"] == "API_REQUEST_REFUSED"


def test_loopback_host_with_port_is_accepted(client: TestClient) -> None:
    assert client.get("/api/v1/health", headers={"Host": "127.0.0.1:8765"}).status_code == 200
    assert client.get("/api/v1/health", headers={"Host": "localhost:8765"}).status_code == 200


def test_foreign_origin_is_refused_even_for_simple_requests(client: TestClient) -> None:
    response = client.get("/api/v1/health", headers={"Origin": "https://evil.example"})
    assert response.status_code == 403
    assert response.json()["code"] == "API_REQUEST_REFUSED"


def test_dev_origin_gets_cors_headers(client: TestClient) -> None:
    response = client.get("/api/v1/health", headers={"Origin": ORIGIN})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ORIGIN
    assert "access-control-allow-credentials" not in response.headers


def test_preflight_allows_only_the_configured_origin(client: TestClient) -> None:
    allowed = client.options(
        "/api/v1/workspace/cml/vector-operations/p01-vector-projection",
        headers={
            "Origin": ORIGIN,
            "Access-Control-Request-Method": "PUT",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == ORIGIN
    assert "PUT" in allowed.headers["access-control-allow-methods"]


def test_cors_never_uses_a_wildcard(make_client: Callable[..., TestClient]) -> None:
    with make_client(origins=("http://127.0.0.1:5173",)) as client:
        response = client.get("/api/v1/health", headers={"Origin": "http://127.0.0.1:5173"})
    assert response.headers.get("access-control-allow-origin") != "*"


def test_configured_origin_is_used(make_client: Callable[..., TestClient]) -> None:
    other = "http://localhost:4000"
    with make_client(origins=(other,)) as client:
        response = client.get("/api/v1/health", headers={"Origin": other})
        refused = client.get("/api/v1/health", headers={"Origin": ORIGIN})
    assert response.headers["access-control-allow-origin"] == other
    assert refused.status_code == 403


def test_oversized_request_body_is_refused(client: TestClient) -> None:
    big = "x" * (1024 * 1024 + 10)
    response = client.put(
        "/api/v1/workspace/cml/vector-operations/p01-vector-projection", json={"content": big}
    )
    assert response.status_code == 413
    assert response.json()["code"] == "API_REQUEST_TOO_LARGE"
