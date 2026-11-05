"""The built web app at ``/``: index and client routes, asset files, and no effect on the API.

Each test builds its own fake ``dist`` under tmp_path, so the result never depends on whether a
real ``web/dist`` exists in the checkout.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dogma.api.app import create_app
from dogma.api.static import page_for
from dogma.core.config import Settings

INDEX_HTML = "<!doctype html><title>dogma shell</title><div id=root></div>"
SENTINEL = "outside-the-build-secret"
ROUTE = "/problem/cml/vector-operations/p01-vector-projection"


@pytest.fixture
def dist(tmp_path: Path) -> Path:
    build = tmp_path / "web-dist"
    (build / "assets").mkdir(parents=True)
    (build / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (build / "assets" / "app-abc123.js").write_text("console.log('app')", encoding="utf-8")
    (build / "assets" / "app-abc123.css").write_text("body{}", encoding="utf-8")
    (tmp_path / "secret.txt").write_text(SENTINEL, encoding="utf-8")
    return build


@pytest.fixture
def served(settings: Settings, dist: Path) -> Iterator[Callable[..., TestClient]]:
    def _make(**kwargs: object) -> TestClient:
        app = create_app(settings, web_dist=dist, **kwargs)  # type: ignore[arg-type]
        return TestClient(app, base_url="http://127.0.0.1")

    yield _make


def test_the_index_is_served_at_the_root_with_a_strict_policy(served: Callable[..., TestClient]):
    response = served().get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "dogma shell" in response.text
    policy = response.headers["content-security-policy"]
    assert "script-src 'self'" in policy
    assert "default-src 'self'" in policy


def test_a_client_route_gets_the_index_so_a_reload_keeps_the_page(
    served: Callable[..., TestClient],
):
    response = served().get(ROUTE)
    assert response.status_code == 200
    assert "dogma shell" in response.text


def test_a_built_asset_is_served_with_its_own_type_and_caching(
    served: Callable[..., TestClient],
):
    response = served().get("/assets/app-abc123.js")
    assert response.status_code == 200
    assert response.text == "console.log('app')"
    assert response.headers["content-type"].startswith("text/javascript")
    assert "immutable" in response.headers["cache-control"]


def test_a_missing_asset_is_a_problem_not_the_index(served: Callable[..., TestClient]):
    response = served().get("/assets/missing-999.js")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == "API_NOT_FOUND"


def test_an_unknown_api_route_stays_a_problem(served: Callable[..., TestClient]):
    response = served().get("/api/v1/nope")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/problem+json")
    assert "dogma shell" not in response.text


@pytest.mark.parametrize("path", ["/docs", "/docs/oauth2-redirect", "/redoc"])
def test_disabled_api_docs_stay_missing_with_a_build(served: Callable[..., TestClient], path: str):
    response = served().get(path)
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/problem+json")


def test_the_api_routes_are_unaffected_by_the_build(served: Callable[..., TestClient]):
    client = served()
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    wrong_method = client.post("/api/v1/health")
    assert wrong_method.status_code == 405
    assert wrong_method.json()["code"] == "API_METHOD_NOT_ALLOWED"


def test_a_traversal_out_of_the_build_is_never_served(served: Callable[..., TestClient]):
    response = served().get("/%2e%2e/%2e%2e/secret.txt")
    assert response.status_code == 404
    assert SENTINEL not in response.text


def test_nothing_is_served_without_a_build(settings: Settings, tmp_path: Path):
    app = create_app(settings, web_dist=tmp_path / "no-such-build")
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/").status_code == 404
        assert client.get(ROUTE).status_code == 404
        assert client.get("/api/v1/health").status_code == 200


def test_the_served_page_may_call_its_own_server(served: Callable[..., TestClient]):
    client = served()
    own = client.get("/api/v1/health", headers={"Origin": "http://127.0.0.1"})
    assert own.status_code == 200
    with_port = client.get(
        "/api/v1/health",
        headers={"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:8765"},
    )
    assert with_port.status_code == 200


def test_a_foreign_origin_is_still_refused_when_the_ui_is_served(
    served: Callable[..., TestClient],
):
    response = served().get("/api/v1/health", headers={"Origin": "https://evil.example"})
    assert response.status_code == 403
    assert response.json()["code"] == "API_REQUEST_REFUSED"


def test_an_origin_that_matches_a_non_loopback_host_is_refused(
    served: Callable[..., TestClient],
):
    response = served().get(
        "/api/v1/health",
        headers={"Host": "evil.example:8765", "Origin": "http://evil.example:8765"},
    )
    assert response.status_code == 403


def test_the_same_origin_exemption_needs_the_build(settings: Settings, tmp_path: Path):
    app = create_app(settings, web_dist=tmp_path / "no-such-build")
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.get("/api/v1/health", headers={"Origin": "http://127.0.0.1"})
    assert response.status_code == 403


def test_page_for_maps_paths_without_leaving_the_build(dist: Path):
    root = dist.resolve()
    assert page_for("/", root) == root / "index.html"
    assert page_for("/problem/a/b/c", root) == root / "index.html"
    assert page_for("/assets/app-abc123.js", root) == root / "assets" / "app-abc123.js"
    assert page_for("/assets/nope.js", root) is None
    assert page_for("/api/v1/whatever", root) is None
    assert page_for("/../secret.txt", root) is None


def test_an_index_symlink_outside_the_build_is_never_served(settings: Settings, dist: Path):
    index = dist / "index.html"
    index.unlink()
    index.symlink_to(dist.parent / "secret.txt")

    assert page_for("/", dist.resolve()) is None
    assert page_for(ROUTE, dist.resolve()) is None
    with TestClient(create_app(settings, web_dist=dist), base_url="http://127.0.0.1") as client:
        response = client.get("/")
    assert response.status_code == 404
    assert SENTINEL not in response.text
