"""``dogma catalog build``: writes var/catalog.json only from valid content, idempotently."""

from __future__ import annotations

import json


def test_build_writes_the_catalog_and_a_rerun_writes_nothing(invoke, repo):
    first = invoke("catalog", "build")
    catalog = repo / "var" / "catalog.json"

    assert first.exit_code == 0, first.output
    assert "Wrote" in first.output
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    assert [topic["id"] for topic in payload["topics"]] == ["cml/vector-operations"]

    second = invoke("catalog", "build")
    assert second.exit_code == 0
    assert "up to date" in second.output


def test_invalid_content_refuses_to_build_and_prints_every_issue(invoke, repo):
    lesson = repo / "content" / "cml" / "vector-operations" / "lesson.md"
    lesson.write_text(lesson.read_text().replace("## Connections", "## Links"))

    outcome = invoke("catalog", "build")

    assert outcome.exit_code == 1
    assert "Catalog not built" in outcome.output
    assert "LESSON_SECTIONS" in outcome.output
    assert not (repo / "var" / "catalog.json").exists()
