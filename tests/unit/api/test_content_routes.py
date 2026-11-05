"""Catalog, topic and problem reads: what the pages show, and what they must never show."""

from __future__ import annotations

from pathlib import Path

import yaml
from fastapi.testclient import TestClient
from tests.unit.api.support import PROBLEM, TOPIC

from dogma.content.catalog import build_catalog, write_catalog
from dogma.core.config import Settings

SENTINEL = "SENTINEL-REFERENCE-SOLUTION-7f3a"
TEST_SENTINEL = "SENTINEL-HIDDEN-TEST-9c1e"


def _problem_dir(repo: Path) -> Path:
    return repo / "content" / TOPIC / "problems" / "p01-vector-projection"


def test_catalog_is_built_from_content_when_no_file_exists(client: TestClient) -> None:
    response = client.get("/api/v1/catalog")
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == 1
    topics = {topic["id"]: topic for topic in body["topics"]}
    assert TOPIC in topics
    problem_ids = [problem["id"] for problem in topics[TOPIC]["problems"]]
    assert problem_ids == [PROBLEM]


def test_catalog_prefers_the_compiled_file(settings: Settings, client: TestClient) -> None:
    catalog = build_catalog(settings)
    renamed = catalog.model_copy(
        update={
            "topics": tuple(
                topic.model_copy(update={"title": "From the compiled file"})
                for topic in catalog.topics
            )
        }
    )
    write_catalog(renamed, settings.var_dir / "catalog.json")
    topics = client.get("/api/v1/catalog").json()["topics"]
    assert topics[0]["title"] == "From the compiled file"


def test_corrupt_compiled_catalog_falls_back_to_building(
    settings: Settings, client: TestClient
) -> None:
    settings.var_dir.mkdir(parents=True, exist_ok=True)
    (settings.var_dir / "catalog.json").write_text("{not json", encoding="utf-8")
    response = client.get("/api/v1/catalog")
    assert response.status_code == 200
    assert response.json()["topics"][0]["id"] == TOPIC


def test_catalog_with_invalid_content_is_a_server_error(
    settings: Settings, client: TestClient
) -> None:
    topic_yaml = settings.content_dir / TOPIC / "topic.yaml"
    topic_yaml.write_text("id: cml/vector-operations\n", encoding="utf-8")
    response = client.get("/api/v1/catalog")
    assert response.status_code == 500
    assert response.json()["code"] == "CONTENT_SCHEMA_INVALID"


def test_topic_detail_has_lesson_meta_cards_and_problems(repo: Path, client: TestClient) -> None:
    body = client.get(f"/api/v1/topics/{TOPIC}").json()
    assert body["id"] == TOPIC
    assert body["track"] == "cml"
    assert body["status"] == "reviewed"
    assert body["requires"] == []
    assert body["lesson"]["title"] == "Vector operations"
    assert body["lesson"]["markdown"].lstrip().startswith("## The system")
    assert "---" not in body["lesson"]["markdown"].split("\n")[0]
    cards = yaml.safe_load((repo / "content" / TOPIC / "cards.yaml").read_text(encoding="utf-8"))
    assert body["cards_count"] == len(cards["cards"])
    assert [problem["id"] for problem in body["problems"]] == [PROBLEM]


def test_unknown_topic_is_404(client: TestClient) -> None:
    response = client.get("/api/v1/topics/cml/does-not-exist")
    assert response.status_code == 404
    assert response.json()["code"] == "CONTENT_NOT_FOUND"


def test_problem_detail_has_statement_starter_and_hint_count(
    repo: Path, client: TestClient
) -> None:
    body = client.get(f"/api/v1/problems/{PROBLEM}").json()
    assert body["id"] == PROBLEM
    assert body["topic_id"] == TOPIC
    assert body["language"] == "python"
    assert body["entry"] == "vector_projection"
    assert body["timeout_s"] == 30
    assert body["problem_md"].strip() != ""
    starter = (_problem_dir(repo) / "starter.py").read_text(encoding="utf-8")
    assert body["starter"] == starter
    assert body["hints_count"] == 3


def test_problem_responses_never_contain_solution_or_hidden_tests(
    repo: Path, client: TestClient
) -> None:
    problem = _problem_dir(repo)
    (problem / "solution.py").write_text(f"# {SENTINEL}\n", encoding="utf-8")
    (problem / "test_hidden.py").write_text(f"# {TEST_SENTINEL}\n", encoding="utf-8")
    for url in (f"/api/v1/problems/{PROBLEM}", f"/api/v1/topics/{TOPIC}", "/api/v1/catalog"):
        text = client.get(url).text
        assert SENTINEL not in text
        assert TEST_SENTINEL not in text
    body = client.get(f"/api/v1/problems/{PROBLEM}").json()
    assert not {"solution", "solution_py", "tests", "hints"} & set(body)


def test_unknown_problem_is_404(client: TestClient) -> None:
    response = client.get("/api/v1/problems/cml/vector-operations/p99-missing")
    assert response.status_code == 404
    assert response.json()["code"] == "CONTENT_NOT_FOUND"


def test_path_traversal_ids_are_not_found(client: TestClient) -> None:
    attempts = [
        "cml%2F..%2F..%2F..%2Fetc%2Fpasswd",
        "..%2F..%2Fetc",
        "cml%2Fvector-operations%2F..%2Fp01-vector-projection",
        "%2Fetc%2Fpasswd",
        "cml%2Fvector-operations%2Fp01-vector-projection%2Fproblem.md",
    ]
    for ref in attempts:
        for prefix in ("problems", "topics"):
            response = client.get(f"/api/v1/{prefix}/{ref}")
            assert response.status_code == 404, (prefix, ref)
            assert response.json()["code"] == "CONTENT_NOT_FOUND"
