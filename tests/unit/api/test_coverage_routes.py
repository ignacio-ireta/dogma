"""GET /coverage: per-track and per-tier counts, with topic rows once a filter is given."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from tests.unit.api.support import REPO_ROOT

from dogma.content.loader import load_syllabus

COVERAGE = "/api/v1/coverage"
EXAMPLE = "cml/vector-operations"


def _syllabus():
    return load_syllabus(REPO_ROOT / "curriculum" / "syllabus.yaml")


def test_unfiltered_coverage_has_track_and_tier_summaries_and_no_rows(client: TestClient) -> None:
    body = client.get(COVERAGE).json()
    syllabus = _syllabus()

    assert [item["key"] for item in body["by_track"]] == [track.code for track in syllabus.tracks]
    assert [item["key"] for item in body["by_tier"]] == ["A", "B", "C"]
    assert body["rows"] is None
    assert body["unreadable_topics"] == []
    titles = {item["key"]: item["title"] for item in body["by_track"]}
    assert titles["cml"] == next(track.title for track in syllabus.tracks if track.code == "cml")
    total = sum(item["topics"] for item in body["by_tier"])
    assert total == sum(track.topic_count for track in syllabus.tracks)


def test_summaries_count_the_authored_example(client: TestClient) -> None:
    body = client.get(COVERAGE).json()

    cml = next(item for item in body["by_track"] if item["key"] == "cml")
    assert cml["with_package"] == 1
    assert cml["with_lesson"] == 1
    assert cml["with_problems"] == 1
    assert cml["statuses"]["missing"] == cml["topics"] - 1
    assert sum(cml["statuses"].values()) == cml["topics"]


def test_a_tier_filter_returns_the_rows_of_that_tier(client: TestClient) -> None:
    syllabus = _syllabus()
    tier = next(
        topic.tier for track in syllabus.tracks for topic in track.topics if topic.id == EXAMPLE
    )

    body = client.get(COVERAGE, params={"tier": tier}).json()

    assert body["tier"] == tier
    assert {row["tier"] for row in body["rows"]} == {tier}
    by_tier = {item["key"]: item for item in body["by_tier"]}
    assert len(body["rows"]) == by_tier[tier]["topics"]
    assert all(item["topics"] == 0 for key, item in by_tier.items() if key != tier)
    example = next(row for row in body["rows"] if row["id"] == EXAMPLE)
    assert example["has_package"] is True
    assert example["has_lesson"] is True
    assert example["n_problems"] >= 1


def test_a_track_filter_scopes_the_summaries_and_the_rows(client: TestClient) -> None:
    body = client.get(COVERAGE, params={"track": "cml"}).json()

    assert [item["key"] for item in body["by_track"]] == ["cml"]
    assert body["rows"], "a filter returns topic rows"
    assert {row["track"] for row in body["rows"]} == {"cml"}
    assert len(body["rows"]) == body["by_track"][0]["topics"]


def test_an_unknown_track_is_not_found(client: TestClient) -> None:
    response = client.get(COVERAGE, params={"track": "zzz"})

    assert response.status_code == 404
    assert response.json()["code"] == "CONTENT_NOT_FOUND"


def test_a_malformed_track_or_tier_is_a_validation_error(client: TestClient) -> None:
    bad_track = client.get(COVERAGE, params={"track": "Not A Track"})
    bad_tier = client.get(COVERAGE, params={"tier": "D"})
    lower_tier = client.get(COVERAGE, params={"tier": "a"})

    assert bad_track.status_code == 422
    assert bad_tier.status_code == 422
    assert bad_tier.json()["code"] == "API_VALIDATION_FAILED"
    assert lower_tier.status_code == 422, "tiers are the enum values A, B and C"


def test_an_unreadable_package_is_listed_and_not_counted(repo: Path, client: TestClient) -> None:
    broken = repo / "content" / "cml" / "broken-topic"
    broken.mkdir(parents=True)
    (broken / "topic.yaml").write_text("id: [unclosed\n", encoding="utf-8")

    body = client.get(COVERAGE, params={"track": "cml"}).json()

    assert body["unreadable_topics"] == ["cml/broken-topic"]
    assert all(row["id"] != "cml/broken-topic" for row in body["rows"])
