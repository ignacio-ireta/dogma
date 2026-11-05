"""``dogma close``: flags for scripts, prompts for people, idempotent, and never a solve."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select
from tests.unit.learning.support import (
    ALPHA,
    NOON,
    build_repo,
    fix_clock,
    utc_timezone,  # noqa: F401  autouse: pins the zone to UTC for these tests
)

from dogma.cli import app
from dogma.core.config import Settings
from dogma.db.errors import StorageErrorCode
from dogma.db.models import Mastery, TopicState
from dogma.db.review_models import Card
from dogma.db.session import open_session
from dogma.db.session_models import StudySession


@pytest.fixture
def plan_settings(tmp_path: Path) -> Settings:
    return build_repo(tmp_path / "repo")


def run(cli, settings: Settings, *args: str, input_text: str | None = None):  # type: ignore[no-untyped-def]
    return cli.invoke(app, list(args), obj=settings, input=input_text, env={"COLUMNS": "200"})


def stored_session(settings: Settings) -> tuple[str, str | None, str | None, str | None]:
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as db:
        row = db.scalars(select(StudySession)).one()
        return (row.status, row.stopped_where, row.next_step, row.self_answer)


def count(settings: Settings, model: type) -> int:  # type: ignore[type-arg]
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as db:
        return db.scalar(select(func.count()).select_from(model)) or 0


def test_close_with_flags_records_the_note_and_introduces_the_topic(
    cli, plan_settings, monkeypatch
) -> None:
    fix_clock(monkeypatch, NOON)
    run(cli, plan_settings, "today")

    result = run(
        cli,
        plan_settings,
        "close",
        "--stopped-where",
        "Predict done",
        "--next-step",
        "Read example 1",
        "--answer",
        "A projection keeps the part of a along b.",
    )

    assert result.exit_code == 0, result.output
    assert "Closed Learn session 1 of" in result.output
    assert "Next time you start: Read example 1" in result.output
    assert f"Introduced {ALPHA}." in result.output
    assert "Added 2 new review cards." in result.output
    assert stored_session(plan_settings) == (
        "closed",
        "Predict done",
        "Read example 1",
        "A projection keeps the part of a along b.",
    )
    assert count(plan_settings, Card) == 2


def test_close_never_marks_a_topic_solved(cli, plan_settings, monkeypatch) -> None:
    fix_clock(monkeypatch, NOON)
    run(cli, plan_settings, "today")

    run(cli, plan_settings, "close", "--stopped-where", "a", "--next-step", "b")

    with open_session(plan_settings, on_failure=StorageErrorCode.DB_READ_FAILED) as db:
        assert db.get(TopicState, ALPHA).mastery == Mastery.INTRODUCED


def test_a_second_close_says_so_and_keeps_the_stored_note(cli, plan_settings, monkeypatch) -> None:
    fix_clock(monkeypatch, NOON)
    run(cli, plan_settings, "today")
    run(cli, plan_settings, "close", "--stopped-where", "first", "--next-step", "first next")
    before = stored_session(plan_settings)
    fix_clock(monkeypatch, NOON + timedelta(minutes=30))

    result = run(cli, plan_settings, "close", "--stopped-where", "second", "--next-step", "second")

    assert result.exit_code == 0, result.output
    assert "Already closed" in result.output
    assert "Next time you start: first next" in result.output
    assert stored_session(plan_settings) == before


def test_close_without_a_session_today_is_a_structured_error(
    cli, plan_settings, monkeypatch
) -> None:
    fix_clock(monkeypatch, NOON)

    result = run(cli, plan_settings, "close", "--stopped-where", "a", "--next-step", "b")

    assert result.exit_code == 1
    assert "SESSION_NOT_FOUND" in result.output


def test_close_asks_for_missing_fields_and_keeps_the_answers(
    cli, plan_settings, monkeypatch
) -> None:
    fix_clock(monkeypatch, NOON)
    run(cli, plan_settings, "today")

    result = run(
        cli,
        plan_settings,
        "close",
        input_text="Predict done\nRead example 1\n\nA shadow on b.\n",
    )

    assert result.exit_code == 0, result.output
    assert "Where did you stop?" in result.output
    assert "The exact next step" in result.output
    assert stored_session(plan_settings) == (
        "closed",
        "Predict done",
        "Read example 1",
        "A shadow on b.",
    )


def test_optional_prompts_at_end_of_input_are_skipped(cli, plan_settings, monkeypatch) -> None:
    fix_clock(monkeypatch, NOON)
    run(cli, plan_settings, "today")

    result = run(cli, plan_settings, "close", input_text="Predict done\nRead example 1\n")

    assert result.exit_code == 0, result.output
    assert stored_session(plan_settings) == ("closed", "Predict done", "Read example 1", None)


def test_required_fields_without_a_terminal_are_a_usage_error_and_save_nothing(
    cli, plan_settings, monkeypatch
) -> None:
    fix_clock(monkeypatch, NOON)
    run(cli, plan_settings, "today")

    result = run(cli, plan_settings, "close", input_text="")

    assert result.exit_code == 2
    assert "pass --stopped-where" in result.output
    assert stored_session(plan_settings)[0] == "open"


def test_an_empty_flag_value_is_refused_and_saves_nothing(cli, plan_settings, monkeypatch) -> None:
    fix_clock(monkeypatch, NOON)
    run(cli, plan_settings, "today")

    result = run(cli, plan_settings, "close", "--stopped-where", "a", "--next-step", "   ")

    assert result.exit_code == 1
    assert "SESSION_NOTE_INVALID" in result.output
    assert stored_session(plan_settings)[0] == "open"


def test_close_help_ends_with_an_example(cli, plan_settings) -> None:
    result = run(cli, plan_settings, "close", "--help")

    assert result.exit_code == 0
    assert "Example:" in result.output
