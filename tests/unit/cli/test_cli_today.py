"""``dogma today``: the note first, one first action, resume on repeat, low energy."""

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
from dogma.db.session import open_session
from dogma.db.session_models import StudySession

TOMORROW = NOON + timedelta(days=1)


@pytest.fixture
def plan_settings(tmp_path: Path) -> Settings:
    return build_repo(tmp_path / "repo")


def run(cli, settings: Settings, *args: str, input_text: str | None = None):  # type: ignore[no-untyped-def]
    return cli.invoke(app, list(args), obj=settings, input=input_text, env={"COLUMNS": "200"})


def session_count(settings: Settings) -> int:
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as db:
        return db.scalar(select(func.count()).select_from(StudySession)) or 0


def test_the_first_today_shows_the_learn_plan_and_exactly_one_first_action(
    cli, plan_settings, monkeypatch
) -> None:
    fix_clock(monkeypatch, NOON)

    result = run(cli, plan_settings, "today")

    assert result.exit_code == 0, result.output
    assert "No earlier session yet" in result.output
    assert "session 1 of 3" in result.output
    assert "Learn" in result.output and "45 min" in result.output
    for block in ("Retrieve", "Predict", "Learn", "Practice", "Close"):
        assert block in result.output
    assert result.output.count("First action:") == 1
    assert f"open content/{ALPHA}/lesson.md" in result.output


def test_the_last_transition_note_is_printed_before_the_plan(
    cli, plan_settings, monkeypatch
) -> None:
    fix_clock(monkeypatch, NOON)
    run(cli, plan_settings, "today")
    run(
        cli,
        plan_settings,
        "close",
        "--stopped-where",
        "Practice, example 2",
        "--next-step",
        "Solve example 3",
        "--blocker",
        "numpy shape error",
    )
    fix_clock(monkeypatch, TOMORROW)

    result = run(cli, plan_settings, "today")

    assert result.exit_code == 0, result.output
    note_at = result.output.index("Last session")
    plan_at = result.output.index("Today ·")
    assert note_at < plan_at
    assert "Stopped at: Practice, example 2" in result.output
    assert "Next step:  Solve example 3" in result.output
    assert "Blocker:    numpy shape error" in result.output


def test_repeating_today_shows_the_same_plan_and_stores_one_session(
    cli, plan_settings, monkeypatch
) -> None:
    fix_clock(monkeypatch, NOON)
    first = run(cli, plan_settings, "today")
    fix_clock(monkeypatch, NOON + timedelta(hours=1))

    second = run(cli, plan_settings, "today")

    assert second.exit_code == 0, second.output
    assert "Resuming the session you planned earlier." in second.output
    first_action = [line for line in first.output.splitlines() if "First action:" in line]
    assert first_action and first_action == [
        line for line in second.output.splitlines() if "First action:" in line
    ]
    assert session_count(plan_settings) == 1


def test_low_energy_prints_the_minimum_session_with_no_new_material(
    cli, plan_settings, monkeypatch
) -> None:
    fix_clock(monkeypatch, NOON)

    result = run(cli, plan_settings, "today", "--energy", "low")

    assert result.exit_code == 0, result.output
    assert "Minimum viable" in result.output and "20 min" in result.output
    assert "Review" in result.output and "Re-solve" in result.output
    assert "lesson.md" not in result.output
    assert "Nothing is due and no solved problem" in result.output


def test_an_unknown_energy_level_is_a_usage_error(cli, plan_settings, monkeypatch) -> None:
    fix_clock(monkeypatch, NOON)

    result = run(cli, plan_settings, "today", "--energy", "high")

    assert result.exit_code == 2


def test_a_finished_day_says_so_and_plans_nothing_more(cli, plan_settings, monkeypatch) -> None:
    for offset in (0, 60, 120):
        fix_clock(monkeypatch, NOON + timedelta(minutes=offset))
        run(cli, plan_settings, "today")
        run(
            cli,
            plan_settings,
            "close",
            "--stopped-where",
            "done",
            "--next-step",
            "next",
        )
    fix_clock(monkeypatch, NOON + timedelta(hours=4))

    result = run(cli, plan_settings, "today")

    assert result.exit_code == 0, result.output
    assert "All 3 sessions for" in result.output
    assert "the next session starts tomorrow" in result.output
    assert session_count(plan_settings) == 3


def test_an_unclosed_earlier_session_is_resumed_and_can_be_closed(
    cli, plan_settings, monkeypatch
) -> None:
    fix_clock(monkeypatch, NOON)
    run(cli, plan_settings, "today")
    fix_clock(monkeypatch, TOMORROW)

    result = run(cli, plan_settings, "today")

    assert result.exit_code == 0, result.output
    assert "Resuming the session" in result.output
    assert "Originally planned for" in result.output
    assert session_count(plan_settings) == 1
    closed = run(
        cli,
        plan_settings,
        "close",
        "--stopped-where",
        "Practice",
        "--next-step",
        "Submit the solution",
    )
    assert closed.exit_code == 0, closed.output
    next_day = run(cli, plan_settings, "today")
    assert next_day.exit_code == 0, next_day.output
    assert "Next step:  Submit the solution" in next_day.output
    assert session_count(plan_settings) == 2


def test_today_help_ends_with_an_example(cli, plan_settings) -> None:
    result = run(cli, plan_settings, "today", "--help")

    assert result.exit_code == 0
    assert "Example:" in result.output
