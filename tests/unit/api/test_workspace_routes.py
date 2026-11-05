"""The learner's workspace file: the browser editor and the CLI share this one file."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from tests.unit.api.support import PROBLEM

from dogma.content.resolve import resolve_problem, workspace_file
from dogma.core.config import Settings

URL = f"/api/v1/workspace/{PROBLEM}"
CODE = "def vector_projection(a, b):\n    return a\n"


def test_missing_workspace_file_reads_as_not_existing(client: TestClient) -> None:
    body = client.get(URL).json()
    assert body == {"problem_id": PROBLEM, "exists": False, "content": None}


def test_put_then_get_round_trips_the_text(client: TestClient) -> None:
    saved = client.put(URL, json={"content": CODE})
    assert saved.status_code == 200
    assert saved.json() == {"problem_id": PROBLEM, "exists": True, "content": CODE}
    assert client.get(URL).json()["content"] == CODE


def test_put_writes_the_file_the_cli_grades(settings: Settings, client: TestClient) -> None:
    client.put(URL, json={"content": CODE})
    ref = resolve_problem(settings, PROBLEM)
    path = workspace_file(settings, ref)
    assert path.read_text(encoding="utf-8") == CODE
    assert path.is_relative_to(settings.workspace_dir)


def test_put_overwrites_and_leaves_no_temp_files(settings: Settings, client: TestClient) -> None:
    client.put(URL, json={"content": CODE})
    client.put(URL, json={"content": "def vector_projection(a, b):\n    return b\n"})
    directory = settings.workspace_dir / "cml" / "vector-operations"
    names = sorted(path.name for path in directory.iterdir())
    assert names == ["p01-vector-projection.py"]


def test_oversized_content_is_refused_and_nothing_is_written(
    settings: Settings, client: TestClient
) -> None:
    big = "#" * (256 * 1024 + 1)
    response = client.put(URL, json={"content": big})
    assert response.status_code == 413
    assert response.json()["code"] == "API_WORKSPACE_TOO_LARGE"
    assert not (settings.workspace_dir / "cml").exists()


def test_content_at_the_limit_is_accepted(client: TestClient) -> None:
    exact = "#" * (256 * 1024)
    assert client.put(URL, json={"content": exact}).status_code == 200


def test_non_utf8_file_is_reported_not_silently_overwritten(
    settings: Settings, client: TestClient
) -> None:
    path = _workspace_path(settings)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\xff\xfe\x00bad")
    response = client.get(URL)
    assert response.status_code == 422
    assert response.json()["code"] == "API_WORKSPACE_INVALID"


def test_unknown_problem_is_404_and_writes_nothing(settings: Settings, client: TestClient) -> None:
    response = client.put(
        "/api/v1/workspace/cml/vector-operations/p99-missing", json={"content": "x"}
    )
    assert response.status_code == 404
    assert response.json()["code"] == "CONTENT_NOT_FOUND"
    assert not (settings.workspace_dir / "cml").exists()


def test_traversal_id_cannot_write_outside_the_workspace(
    tmp_path: Path, settings: Settings, client: TestClient
) -> None:
    outside = tmp_path / "escape.py"
    ref = "cml%2F..%2F..%2F..%2F" + "escape"
    response = client.put(f"/api/v1/workspace/{ref}", json={"content": CODE})
    assert response.status_code == 404
    assert not outside.exists()
    assert not (settings.repo_root.parent / "escape.py").exists()


def test_missing_content_field_is_a_validation_error(client: TestClient) -> None:
    response = client.put(URL, json={"text": CODE})
    assert response.status_code == 422
    assert response.json()["code"] == "API_VALIDATION_FAILED"


def _workspace_path(settings: Settings) -> Path:
    return settings.workspace_dir / "cml" / "vector-operations" / "p01-vector-projection.py"
