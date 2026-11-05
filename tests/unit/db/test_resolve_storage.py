"""Timed re-solve windows in SQLite: one open window at a time, closing on a submit, hints, and
the 12-hour abandonment rule. Every window write goes through the store or the storage helpers.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError
from tests.unit.db.support import PROBLEM, make_report

from dogma.core.config import Settings
from dogma.core.errors import DogmaError, ErrorCode
from dogma.db.errors import StorageErrorCode
from dogma.db.models import AttemptMode, Mastery, TopicState
from dogma.db.resolve_models import ResolveWindow, WindowStatus
from dogma.db.resolve_store import (
    abandon_window,
    all_windows,
    count_hint,
    first_passes,
    open_window,
    start_window,
    windows_for,
)
from dogma.db.session import open_session, upgrade_database
from dogma.db.store import record_attempt
from dogma.grading.models import RunStatus
from dogma.learning.resolves import ClosedWindow

TOPIC = "cml/vector-operations"
PASS_AT = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
BUDGET_S = 20 * 60


def _passing(settings: Settings, when: datetime, attempt: str | None = None) -> None:
    record_attempt(
        settings,
        make_report(
            RunStatus.PASSED, public=(4, 4), hidden=(11, 11), attempt_id=attempt or uuid4().hex
        ),
        mode=AttemptMode.SUBMIT,
        core=True,
        started_at=when - timedelta(seconds=2),
        submitted_at=when,
    )


def _seed_topic(settings: Settings, mastery: Mastery, solved_at: datetime) -> None:
    """Set the topic's state, whether or not a passing submit has already created it."""
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        state = session.get(TopicState, TOPIC)
        if state is None:
            session.add(TopicState(topic_id=TOPIC, mastery=mastery.value, solved_at=solved_at))
        else:
            state.mastery, state.solved_at = mastery.value, solved_at


def _start(settings: Settings, at: datetime) -> None:
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        start_window(session, problem_id=PROBLEM, problem_version=2, budget_s=BUDGET_S, now=at)


def _close(
    settings: Settings, at: datetime, *, passed: bool, attempt: str = "a-close"
) -> ClosedWindow | None:
    """Submit through the real recording path, so the attempt row exists before the window."""
    if passed:
        report = make_report(RunStatus.PASSED, public=(4, 4), hidden=(11, 11), attempt_id=attempt)
    else:
        report = make_report(RunStatus.FAILED, public=(3, 4), hidden=(0, 11), attempt_id=attempt)
    return record_attempt(
        settings,
        report,
        mode=AttemptMode.SUBMIT,
        core=True,
        started_at=at - timedelta(seconds=1),
        submitted_at=at,
    )


def _mastery(settings: Settings) -> tuple[str, datetime | None]:
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        state = session.get(TopicState, TOPIC)
        assert state is not None
        return state.mastery, state.retained_at


def test_the_database_refuses_a_second_open_window(settings: Settings) -> None:
    upgrade_database(settings)
    _passing(settings, PASS_AT)
    _start(settings, PASS_AT + timedelta(days=7))
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        session.add(
            ResolveWindow(
                id=uuid4().hex,
                problem_id=PROBLEM,
                problem_version=2,
                topic_id=TOPIC,
                started_at=PASS_AT,
                budget_s=BUDGET_S,
                hints_used=0,
                status=WindowStatus.OPEN.value,
            )
        )
        with pytest.raises(IntegrityError):
            session.flush()


def test_starting_while_one_is_open_names_it(settings: Settings) -> None:
    upgrade_database(settings)
    _passing(settings, PASS_AT)
    _start(settings, PASS_AT + timedelta(days=7))

    with pytest.raises(DogmaError) as caught:
        _start(settings, PASS_AT + timedelta(days=7, hours=1))

    assert caught.value.code is ErrorCode.RESOLVE_ALREADY_OPEN
    assert PROBLEM in caught.value.message


def test_a_window_open_for_twelve_hours_is_still_open_and_longer_is_abandoned(
    settings: Settings,
) -> None:
    upgrade_database(settings)
    _passing(settings, PASS_AT)
    started = PASS_AT + timedelta(days=7)
    _start(settings, started)

    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        still = open_window(session, now=started + timedelta(hours=12))
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        gone = open_window(session, now=started + timedelta(hours=12, seconds=1))
        history = windows_for(session, PROBLEM, now=started + timedelta(hours=12, seconds=1))

    assert still is not None
    assert gone is None
    assert [item.status for item in history] == [WindowStatus.ABANDONED]
    assert history[0].closed_at == started + timedelta(hours=12, seconds=1)


def test_a_submit_closes_the_open_window_with_its_verdict(settings: Settings) -> None:
    upgrade_database(settings)
    _passing(settings, PASS_AT)
    started = PASS_AT + timedelta(days=7)
    _start(settings, started)

    closed = _close(settings, started + timedelta(minutes=10), passed=True)

    assert closed is not None
    assert closed.status is WindowStatus.PASSED
    assert closed.used_s == 600
    assert closed.budget_s == BUDGET_S
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        (row,) = all_windows(session, now=started)
    assert row.attempt_id == "a-close"
    assert row.closed_at == started + timedelta(minutes=10)


def test_a_late_or_hinted_pass_closes_as_partial(settings: Settings) -> None:
    upgrade_database(settings)
    _passing(settings, PASS_AT)
    started = PASS_AT + timedelta(days=7)
    _start(settings, started)
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        assert count_hint(session, problem_id=PROBLEM, now=started + timedelta(minutes=1))

    closed = _close(settings, started + timedelta(minutes=30), passed=True)

    assert closed is not None
    assert closed.status is WindowStatus.PARTIAL
    assert closed.hints_used == 1


def test_only_the_first_submit_decides_and_a_later_one_changes_nothing(settings: Settings) -> None:
    upgrade_database(settings)
    _passing(settings, PASS_AT)
    started = PASS_AT + timedelta(days=7)
    _start(settings, started)
    _close(settings, started + timedelta(minutes=5), passed=False, attempt="first")

    later = _close(settings, started + timedelta(minutes=6), passed=True, attempt="second")

    assert later is None
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        (row,) = all_windows(session, now=started)
    assert row.status is WindowStatus.FAILED
    assert row.attempt_id == "first"


def test_a_failed_window_on_a_retained_topic_drops_it_to_consolidating(settings: Settings) -> None:
    upgrade_database(settings)
    _passing(settings, PASS_AT)
    _seed_topic(settings, Mastery.CONSOLIDATING, PASS_AT)
    first = PASS_AT + timedelta(days=8)
    _start(settings, first)
    _close(settings, first + timedelta(minutes=10), passed=True, attempt="first-pass")
    assert _mastery(settings)[0] == Mastery.RETAINED.value

    second = first + timedelta(days=7)
    _start(settings, second)
    _close(settings, second + timedelta(minutes=10), passed=False, attempt="fail")

    mastery, retained_at = _mastery(settings)
    assert mastery == Mastery.CONSOLIDATING.value
    assert retained_at == first + timedelta(minutes=10)


def test_a_passed_window_retains_a_consolidating_topic_seven_days_after_solving(
    settings: Settings,
) -> None:
    upgrade_database(settings)
    _passing(settings, PASS_AT)
    _seed_topic(settings, Mastery.CONSOLIDATING, PASS_AT)
    started = PASS_AT + timedelta(days=7, hours=1)
    _start(settings, started)

    _close(settings, started + timedelta(minutes=10), passed=True)

    assert _mastery(settings) == (Mastery.RETAINED.value, started + timedelta(minutes=10))


def test_a_pass_under_seven_days_after_solving_leaves_the_topic_alone(settings: Settings) -> None:
    upgrade_database(settings)
    _passing(settings, PASS_AT)
    _seed_topic(settings, Mastery.CONSOLIDATING, PASS_AT)
    started = PASS_AT + timedelta(days=6)
    _start(settings, started)

    _close(settings, started + timedelta(minutes=10), passed=True)

    assert _mastery(settings) == (Mastery.CONSOLIDATING.value, None)


def test_abandoning_leaves_the_schedule_and_mastery_alone(settings: Settings) -> None:
    upgrade_database(settings)
    _passing(settings, PASS_AT)
    started = PASS_AT + timedelta(days=7)
    _start(settings, started)
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        row = abandon_window(session, now=started + timedelta(minutes=3), problem_id=PROBLEM)
        assert row.status is WindowStatus.ABANDONED
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        passes = first_passes(session)

    assert passes == {PROBLEM: PASS_AT}


def test_hints_count_only_while_a_window_is_open_on_that_problem(settings: Settings) -> None:
    upgrade_database(settings)
    _passing(settings, PASS_AT)
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        assert count_hint(session, problem_id=PROBLEM, now=PASS_AT) is False
    started = PASS_AT + timedelta(days=7)
    _start(settings, started)

    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        assert count_hint(session, problem_id=PROBLEM, now=started) is True
        assert count_hint(session, problem_id="cml/other/p01-x", now=started) is False
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        (row,) = all_windows(session, now=started)
    assert row.hints_used == 1


def test_a_submit_returns_the_next_due_time_and_a_run_closes_nothing(settings: Settings) -> None:
    upgrade_database(settings)
    _passing(settings, PASS_AT)
    started = PASS_AT + timedelta(days=7)
    _start(settings, started)
    record_attempt(
        settings,
        make_report(RunStatus.PASSED, public=(4, 4), hidden=(0, 11), attempt_id="run-only"),
        mode=AttemptMode.RUN,
        core=True,
        started_at=started,
        submitted_at=started + timedelta(minutes=1),
    )
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        assert open_window(session, now=started) is not None

    closed = record_attempt(
        settings,
        make_report(RunStatus.PASSED, public=(4, 4), hidden=(11, 11), attempt_id="submit"),
        mode=AttemptMode.SUBMIT,
        core=True,
        started_at=started,
        submitted_at=started + timedelta(minutes=12),
    )

    assert closed is not None
    assert closed.status is WindowStatus.PASSED
    assert closed.next_due is not None
    assert closed.next_due.stage == 1
    assert closed.next_due.due_at == started + timedelta(minutes=12) + timedelta(days=21)
