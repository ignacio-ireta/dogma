"""GET /search: the corpus index, read-only, with literal terms and topic filters."""

from __future__ import annotations

from fastapi.testclient import TestClient
from tests.unit.api.support import TOPIC

from dogma.core.config import Settings
from dogma.corpus.index import connect, replace_source
from dogma.corpus.models import Chunk


def _index(settings: Settings) -> None:
    chunks = (
        Chunk(
            source_id="lib-linear",
            path="corpus/md/linear.md",
            page=12,
            headings=("Projections",),
            ordinal=1,
            text="The orthogonal projection of a onto b uses the dot product.",
            topic_ids=(TOPIC,),
        ),
        Chunk(
            source_id="lib-stats",
            path="corpus/md/stats.md",
            page=None,
            headings=("Bias",),
            ordinal=2,
            text="Bias measures systematic error in an estimator.",
            topic_ids=("math/bias",),
        ),
    )
    conn = connect(settings, create=True)
    try:
        replace_source(conn, "lib-linear", "corpus/md/linear.md", "d1", (chunks[0],))
        replace_source(conn, "lib-stats", "corpus/md/stats.md", "d2", (chunks[1],))
    finally:
        conn.close()


def test_missing_index_is_unavailable_with_a_hint(client: TestClient) -> None:
    response = client.get("/api/v1/search", params={"q": "projection"})
    assert response.status_code == 503
    assert response.json()["code"] == "CORPUS_INDEX_MISSING"


def test_search_returns_cited_passages(settings: Settings, client: TestClient) -> None:
    _index(settings)
    body = client.get("/api/v1/search", params={"q": "projection"}).json()
    assert body["query"] == "projection"
    hit = body["hits"][0]
    assert "orthogonal projection" in hit["text"]
    assert hit["source_id"] == "lib-linear"
    assert hit["page"] == 12
    assert hit["headings"] == ["Projections"]
    assert hit["topic_ids"] == [TOPIC]
    assert hit["citation"].startswith("[lib-linear p.12]")
    assert isinstance(hit["score"], float)


def test_topic_filter_narrows_the_hits(settings: Settings, client: TestClient) -> None:
    _index(settings)
    assert client.get("/api/v1/search", params={"q": "bias", "topic": TOPIC}).json()["hits"] == []
    hits = client.get("/api/v1/search", params={"q": "bias", "topic": "math/bias"}).json()["hits"]
    assert [hit["source_id"] for hit in hits] == ["lib-stats"]


def test_limit_caps_the_number_of_hits(settings: Settings, client: TestClient) -> None:
    _index(settings)
    hits = client.get("/api/v1/search", params={"q": "the", "limit": 1}).json()["hits"]
    assert len(hits) <= 1


def test_limit_bounds_are_validated(client: TestClient) -> None:
    assert client.get("/api/v1/search", params={"q": "x", "limit": 0}).status_code == 422
    assert client.get("/api/v1/search", params={"q": "x", "limit": 101}).status_code == 422


def test_query_operators_are_literal_text(settings: Settings, client: TestClient) -> None:
    _index(settings)
    response = client.get("/api/v1/search", params={"q": 'projection" OR * NOT (x'})
    assert response.status_code == 200
    assert response.json()["hits"] == []


def test_empty_query_returns_no_hits(settings: Settings, client: TestClient) -> None:
    _index(settings)
    assert client.get("/api/v1/search", params={"q": ""}).json()["hits"] == []


def test_overlong_query_is_a_validation_error(client: TestClient) -> None:
    response = client.get("/api/v1/search", params={"q": "x" * 4097})
    assert response.status_code == 422
    assert response.json()["code"] == "API_VALIDATION_FAILED"
