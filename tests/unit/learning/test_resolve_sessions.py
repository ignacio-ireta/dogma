"""Consolidate plans a due timed re-solve from stored state, through the real today and close flow.

The synthetic repository from ``support`` has a solved ALPHA topic. Its core problem's first pass
sets the schedule, so seven days later its timed re-solve is due and Consolidate plans it.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from tests.unit.learning.support import (
    ALPHA,
    NOON,
    build_repo,
    core_problem,
    solve,
    utc_timezone,  # noqa: F401  autouse: pins the zone to UTC for these tests
)

from dogma.core.config import Settings
from dogma.db.errors import StorageErrorCode
from dogma.db.resolve_store import start_window
from dogma.db.session import open_session
from dogma.learning import sessions
from dogma.learning.planner import Energy
from dogma.learning.sessions import CloseNote


@pytest.fixture
def repo(tmp_path: Path) -> Settings:
    return build_repo(tmp_path / "repo")


def _close_all_learn_sessions(settings: Settings, when: datetime) -> None:
    """Close the first two Learn sessions so the third (Consolidate) is planned next."""
    for _ in range(2):
        view = sessions.today(settings, now=when, energy=Energy.NORMAL)
        assert view.current is not None
        sessions.close_session(
            settings,
            view.current.session_id,
            CloseNote(stopped_where="Practice", next_step="Continue"),
            answer=None,
            now=when,
        )


def _open_window(settings: Settings, when: datetime) -> None:
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        start_window(
            session,
            problem_id=core_problem(ALPHA),
            problem_version=1,
            budget_s=20 * 60,
            now=when,
        )


def test_consolidate_plans_the_timed_re_solve_once_it_is_due(repo: Settings) -> None:
    solve(repo, ALPHA, when=NOON)
    seven_days = NOON + timedelta(days=7, hours=1)

    _close_all_learn_sessions(repo, seven_days)
    view = sessions.today(repo, now=seven_days, energy=Energy.NORMAL)

    assert view.current is not None
    assert view.current.plan.kind.value == "consolidate"
    detail = " ".join(block.detail for block in view.current.plan.blocks)
    assert f"dogma resolve {core_problem(ALPHA)}" in detail
    assert "budget 20 min" in detail
    assert "no hints" in detail
    assert f"dogma resolve {core_problem(ALPHA)}" in view.current.plan.first_action


def test_before_the_due_time_consolidate_stays_untimed(repo: Settings) -> None:
    solve(repo, ALPHA, when=NOON)
    too_early = NOON + timedelta(days=6)

    _close_all_learn_sessions(repo, too_early)
    view = sessions.today(repo, now=too_early, energy=Energy.NORMAL)

    assert view.current is not None
    detail = " ".join(block.detail for block in view.current.plan.blocks)
    assert "dogma resolve" not in detail
    assert "Untimed" in detail


def test_an_open_window_is_planned_as_running_not_as_a_new_start(repo: Settings) -> None:
    solve(repo, ALPHA, when=NOON)
    seven_days = NOON + timedelta(days=7, hours=1)
    _open_window(repo, seven_days)

    _close_all_learn_sessions(repo, seven_days)
    view = sessions.today(repo, now=seven_days, energy=Energy.NORMAL)

    assert view.current is not None
    assert view.current.plan.first_action.startswith("Finish the open timed re-solve")


def test_minimum_viable_sessions_never_plan_a_timed_re_solve(repo: Settings) -> None:
    solve(repo, ALPHA, when=NOON)
    seven_days = NOON + timedelta(days=7, hours=1)

    view = sessions.today(repo, now=seven_days, energy=Energy.LOW)

    assert view.current is not None
    detail = " ".join(block.detail for block in view.current.plan.blocks)
    assert "dogma resolve" not in detail
