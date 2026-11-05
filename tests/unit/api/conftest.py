"""API fixtures: a scratch repo, a TestClient with a frozen clock, and a fake grader.

Every test gets its own repo under tmp_path; nothing touches the real content/, var/ or workspace/.
The client's Host is 127.0.0.1, as the Host guard requires for a local server.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.unit.api.support import ORIGIN, FakeGrader, FrozenClock, copy_repo

from dogma.api.app import create_app
from dogma.api.deps import get_clock, get_gateway, get_model
from dogma.core.config import Settings
from dogma.runner import attempt
from dogma.tutor.gateway import Gateway, LLMProvider


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return copy_repo(tmp_path)


@pytest.fixture
def settings(repo: Path) -> Settings:
    return Settings(repo_root=repo)


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock()


@pytest.fixture
def grader(monkeypatch: pytest.MonkeyPatch) -> FakeGrader:
    fake = FakeGrader()
    monkeypatch.setattr(attempt, "grade_attempt", fake)
    return fake


@pytest.fixture
def make_client(settings: Settings, clock: FrozenClock) -> Callable[..., TestClient]:
    """Build a client for ``settings``; tests may pass their own origins or a gateway."""

    def _make(
        *,
        app_settings: Settings | None = None,
        origins: tuple[str, ...] = (ORIGIN,),
        gateway: Gateway | None = None,
        model: LLMProvider | None = None,
        raise_server_exceptions: bool = True,
    ) -> TestClient:
        app = create_app(app_settings or settings, allowed_origins=origins)
        app.dependency_overrides[get_clock] = lambda: clock
        if gateway is not None:
            app.dependency_overrides[get_gateway] = lambda: gateway
        if model is not None:
            app.dependency_overrides[get_model] = lambda: model
        return TestClient(
            app, base_url="http://127.0.0.1", raise_server_exceptions=raise_server_exceptions
        )

    return _make


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> Iterator[TestClient]:
    with make_client() as test_client:
        yield test_client
