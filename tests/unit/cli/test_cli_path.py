"""``dogma path``: week selection, topic statuses and problem ids."""

from __future__ import annotations

from datetime import date

from dogma.content.loader import load_topic
from dogma.content.schemas import PathWeek
from dogma.learning.path import READY, current_week


def _week(number: int, starts: date) -> PathWeek:
    return PathWeek(week=number, starts=starts, theme=f"theme {number}", topics=("cml/x",))


def test_current_week_is_the_last_started_week():
    weeks = (_week(1, date(2026, 10, 26)), _week(2, date(2026, 11, 2)), _week(3, date(2026, 11, 9)))
    assert current_week(weeks, date(2026, 11, 3)).week == 2
    assert current_week(weeks, date(2026, 11, 9)).week == 3
    assert current_week(weeks, date(2027, 1, 1)).week == 3


def test_before_the_path_starts_the_first_week_is_current():
    weeks = (_week(1, date(2026, 10, 26)), _week(2, date(2026, 11, 2)))
    assert current_week(weeks, date(2026, 10, 1)).week == 1


def test_only_verified_and_reviewed_topics_are_ready():
    assert {"verified", "reviewed"} == READY


def test_week_shows_theme_dates_status_and_problem_ids(invoke, settings):
    outcome = invoke("path", "--week", "1")

    assert outcome.exit_code == 0, outcome.output
    assert "Week 1" in outcome.output
    assert "Linear algebra & numpy fluency" in outcome.output
    assert "starts Mon 26 Oct 2026" in outcome.output
    status = load_topic(settings.content_dir / "cml/vector-operations").meta.status
    assert status in outcome.output
    assert "p01-vector-projection" in outcome.output
    assert "missing" in outcome.output, "topics with no package yet are missing"
    assert "1/12 topics ready" in " ".join(outcome.output.split())


def test_an_unknown_week_is_a_usage_error(invoke):
    outcome = invoke("path", "--week", "99")

    assert outcome.exit_code == 2
    assert "not 99" in outcome.output


def _freeze_today(monkeypatch, day: date) -> None:
    """``dogma path`` reads the system date; pin it so the test does not expire."""

    class _Frozen(date):
        @classmethod
        def today(cls) -> date:
            return day

    monkeypatch.setattr("dogma.cli.path_view.date", _Frozen)


def test_default_week_is_the_current_one(invoke, monkeypatch):
    # Before the core path starts, the default is week 1.
    _freeze_today(monkeypatch, date(2026, 10, 1))
    outcome = invoke("path")

    assert outcome.exit_code == 0, outcome.output
    assert "Week 1 ·" in outcome.output


def test_default_week_follows_the_date(invoke, monkeypatch):
    # Tue 3 Nov 2026 is in authored week 2 (Mon 2 Nov to Sun 8 Nov).
    _freeze_today(monkeypatch, date(2026, 11, 3))
    outcome = invoke("path")

    assert outcome.exit_code == 0, outcome.output
    assert "Week 2 ·" in outcome.output
    assert "Week 1 ·" not in outcome.output
