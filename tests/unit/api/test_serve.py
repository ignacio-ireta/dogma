"""`dogma serve`: loopback only (ADR-0001). Other hosts are refused before anything binds."""

from __future__ import annotations

from typing import Any

import pytest
from typer.testing import CliRunner

from dogma.cli import app
from dogma.cli.context import CliError
from dogma.cli.serve import check_host
from dogma.core.config import Settings


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost", "127.0.0.2"])
def test_loopback_hosts_are_accepted(host: str) -> None:
    check_host(host)


# Non-loopback addresses are the point of these tests.
@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "::", "192.168.1.10", "10.0.0.5", "example.com", ""],  # noqa: S104
)
def test_other_hosts_are_refused(host: str) -> None:
    with pytest.raises(CliError, match="loopback"):
        check_host(host)


@pytest.fixture
def run_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_run(application: object, **kwargs: Any) -> None:
        calls.append({"app": application, **kwargs})

    monkeypatch.setattr("uvicorn.run", fake_run)
    return calls


def test_non_loopback_host_is_refused_and_nothing_starts(
    settings: Settings, run_calls: list[dict[str, Any]]
) -> None:
    result = CliRunner().invoke(app, ["serve", "--host", "0.0.0.0"], obj=settings)  # noqa: S104
    assert result.exit_code == 2
    assert "loopback" in result.output
    assert run_calls == []


def test_default_serve_binds_loopback_and_prints_the_urls(
    settings: Settings, run_calls: list[dict[str, Any]]
) -> None:
    result = CliRunner().invoke(app, ["serve"], obj=settings, env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert len(run_calls) == 1
    assert run_calls[0]["host"] == "127.0.0.1"
    assert run_calls[0]["port"] == 8765
    assert "http://127.0.0.1:8765/api/v1" in result.output
    assert "http://127.0.0.1:8765/api/v1/openapi.json" in result.output


def test_port_and_allowed_origin_are_passed_through(
    settings: Settings, run_calls: list[dict[str, Any]]
) -> None:
    result = CliRunner().invoke(
        app,
        ["serve", "--port", "9001", "--allow-origin", "http://127.0.0.1:4000"],
        obj=settings,
    )
    assert result.exit_code == 0, result.output
    assert run_calls[0]["port"] == 9001
    assert "http://127.0.0.1:9001/api/v1" in result.output


@pytest.mark.parametrize("port", ["0", "70000"])
def test_invalid_port_is_a_usage_error(
    settings: Settings, run_calls: list[dict[str, Any]], port: str
) -> None:
    result = CliRunner().invoke(app, ["serve", "--port", port], obj=settings)
    assert result.exit_code != 0
    assert run_calls == []
