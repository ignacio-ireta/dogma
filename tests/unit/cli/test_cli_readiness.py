"""``dogma progress`` shows interview-ready counts per path week, and never a streak."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from typer.testing import CliRunner

from dogma.cli import app
from dogma.core.config import Settings
from dogma.db.errors import StorageErrorCode
from dogma.db.models import Mastery, TopicState
from dogma.db.session import open_session

TOPIC = "cml/vector-operations"
READY = "accuracy=3,structure=3,trade-offs=3,communication=4"
NOW = datetime(2026, 10, 27, 9, tzinfo=UTC)


def _text(output: str) -> str:
    return " ".join(output.split())


def test_the_readiness_section_counts_zero_interview_ready_topics_for_a_new_learner(
    cli: CliRunner, settings: Settings
) -> None:
    outcome = cli.invoke(app, ["progress", "--week", "1"], obj=settings, env={"COLUMNS": "200"})

    text = _text(outcome.output)
    assert outcome.exit_code == 0, outcome.output
    assert "Readiness" in text
    assert "Week 1 · interview-ready 0 of 12 topics" in text
    assert "streak" not in text.lower()


def test_a_topic_that_is_interview_ready_is_counted_in_its_week(
    cli: CliRunner, settings: Settings
) -> None:
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        session.add(
            TopicState(
                topic_id=TOPIC,
                mastery=Mastery.INTERVIEW_READY.value,
                solved_at=NOW - timedelta(days=12),
                retained_at=NOW - timedelta(days=3),
                ready_at=NOW,
            )
        )

    outcome = cli.invoke(app, ["progress", "--week", "1"], obj=settings, env={"COLUMNS": "200"})

    text = _text(outcome.output)
    assert outcome.exit_code == 0, outcome.output
    assert "Week 1 · interview-ready 1 of 12 topics" in text
