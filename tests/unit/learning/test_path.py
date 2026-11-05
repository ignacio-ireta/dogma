"""Topic rows for the core path: missing, invalid, or read from the package on disk."""

from __future__ import annotations

from dogma.core.config import Settings
from dogma.learning.path import READY, topic_row


def test_only_verified_and_reviewed_topics_are_ready() -> None:
    assert {"verified", "reviewed"} == READY


def test_a_topic_without_a_package_is_missing(settings: Settings) -> None:
    row = topic_row(settings, "cml/not-written-yet")

    assert row.status == "missing"
    assert row.title == ""
    assert row.problems == ()


def test_a_topic_whose_package_does_not_load_is_invalid(settings: Settings) -> None:
    broken = settings.content_dir / "cml" / "broken"
    broken.mkdir(parents=True)
    (broken / "topic.yaml").write_text("id: [unclosed\n", encoding="utf-8")

    row = topic_row(settings, "cml/broken")

    assert row.status == "invalid"
    assert row.problems == ()


def test_a_readable_topic_reports_its_title_status_and_problems(settings: Settings) -> None:
    row = topic_row(settings, "cml/vector-operations")

    assert row.title
    assert row.status in {"draft", "verified", "reviewed", "retired"}
    assert "p01-vector-projection" in row.problems
