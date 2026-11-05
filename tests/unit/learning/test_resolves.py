"""Timed re-solves as pure rules: the verdict, the derived schedule, mastery and readiness.

Nothing here touches a database or a clock. Each table row is one learner history, written as
windows, and the expected stage, due time, mastery or verdict is spelled out.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from dogma.db.models import Mastery
from dogma.db.resolve_models import WindowStatus
from dogma.learning.resolves import (
    WindowRecord,
    build_schedules,
    judge_window,
    reevaluate_mastery,
    timed_results,
    upcoming,
)

PROBLEM = "cml/vector-operations/p01-vector-projection"
TOPIC = "cml/vector-operations"
PASS_AT = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
BUDGET_S = 20 * 60


def window(
    closed: datetime | None,
    status: WindowStatus,
    *,
    started: datetime | None = None,
    hints: int = 0,
    problem: str = PROBLEM,
) -> WindowRecord:
    """A closed window that closed at ``closed``. ``started`` defaults to 20 minutes earlier."""
    begun = started if started is not None else (closed or PASS_AT) - timedelta(minutes=20)
    return WindowRecord(
        window_id=f"w{begun.isoformat()}",
        problem_id=problem,
        topic_id=TOPIC,
        started_at=begun,
        budget_s=BUDGET_S,
        hints_used=hints,
        status=status,
        closed_at=closed,
        attempt_id=None if status in {WindowStatus.ABANDONED, WindowStatus.OPEN} else "a1",
    )


# --- the verdict of one timed submit -----------------------------------------------------------


@pytest.mark.parametrize(
    ("passed", "elapsed_s", "hints", "expected"),
    [
        (True, 600, 0, WindowStatus.PASSED),
        (True, BUDGET_S, 0, WindowStatus.PASSED),  # exactly on the budget is within it
        (True, BUDGET_S + 1, 0, WindowStatus.PARTIAL),
        (True, 600, 1, WindowStatus.PARTIAL),
        (True, BUDGET_S * 3, 2, WindowStatus.PARTIAL),
        (False, 600, 0, WindowStatus.FAILED),
        (False, BUDGET_S * 3, 2, WindowStatus.FAILED),
    ],
)
def test_the_first_submit_decides_passed_partial_or_failed(
    passed: bool, elapsed_s: float, hints: int, expected: WindowStatus
) -> None:
    verdict = judge_window(passed=passed, elapsed_s=elapsed_s, budget_s=BUDGET_S, hints_used=hints)

    assert verdict is expected


# --- the derived schedule ----------------------------------------------------------------------


def schedule_of(windows: list[WindowRecord], first: datetime | None = PASS_AT):
    passes = {PROBLEM: first} if first is not None else {}
    return build_schedules(passes, windows)[PROBLEM]


def test_a_first_pass_with_no_windows_is_due_seven_days_later() -> None:
    result = schedule_of([])

    assert result.stage == 0
    assert result.due_at == PASS_AT + timedelta(days=7)
    assert "First timed re-solve" in result.reason


@pytest.mark.parametrize(
    ("passes", "expected_stage", "interval_days"),
    [(1, 1, 21), (2, 2, 60), (3, 3, 120), (4, 3, 120), (7, 3, 120)],
)
def test_each_passed_window_moves_the_stage_and_the_interval(
    passes: int, expected_stage: int, interval_days: int
) -> None:
    closes = [PASS_AT + timedelta(days=10 * (index + 1)) for index in range(passes)]
    history = [window(closed, WindowStatus.PASSED) for closed in closes]

    result = schedule_of(history)

    assert result.stage == expected_stage
    assert result.due_at == closes[-1] + timedelta(days=interval_days)


def test_maintenance_stage_says_so_in_its_reason() -> None:
    history = [window(PASS_AT + timedelta(days=d), WindowStatus.PASSED) for d in (10, 40, 100)]

    assert "Maintenance" in schedule_of(history).reason


@pytest.mark.parametrize("status", [WindowStatus.PARTIAL, WindowStatus.FAILED])
def test_a_partial_or_failed_window_is_due_again_three_days_after_it_closed(
    status: WindowStatus,
) -> None:
    closed = PASS_AT + timedelta(days=8)

    result = schedule_of([window(closed, status)])

    assert result.stage == 0
    assert result.due_at == closed + timedelta(days=3)
    assert status.value in result.reason


def test_a_retry_keeps_the_stage_of_the_passes_before_it() -> None:
    passed_at = PASS_AT + timedelta(days=8)
    retry_closed = passed_at + timedelta(days=21)
    history = [
        window(passed_at, WindowStatus.PASSED),
        window(retry_closed, WindowStatus.PARTIAL),
    ]

    result = schedule_of(history)

    assert result.stage == 1
    assert result.due_at == retry_closed + timedelta(days=3)


def test_a_pass_after_a_retry_anchors_the_next_interval_on_that_pass() -> None:
    retry_closed = PASS_AT + timedelta(days=9)
    passed_at = retry_closed + timedelta(days=3)
    history = [
        window(retry_closed, WindowStatus.FAILED),
        window(passed_at, WindowStatus.PASSED),
    ]

    result = schedule_of(history)

    assert result.stage == 1
    assert result.due_at == passed_at + timedelta(days=21)


def test_abandoned_and_open_windows_do_not_move_the_schedule() -> None:
    history = [
        window(PASS_AT + timedelta(days=2), WindowStatus.ABANDONED),
        window(None, WindowStatus.OPEN),
    ]

    result = schedule_of(history)

    assert result.stage == 0
    assert result.due_at == PASS_AT + timedelta(days=7)


def test_a_problem_never_passed_has_no_schedule() -> None:
    assert build_schedules({}, [window(None, WindowStatus.OPEN)]) == {}


def test_a_schedule_reports_whether_it_is_due() -> None:
    result = schedule_of([])

    assert result.is_due(PASS_AT + timedelta(days=7))
    assert not result.is_due(PASS_AT + timedelta(days=6, hours=23))


# --- mastery after a window or a card lapse ----------------------------------------------------

SOLVED_AT = datetime(2026, 10, 1, 12, tzinfo=UTC)


def reevaluate(
    current: Mastery,
    windows: list[WindowRecord],
    *,
    retained_at: datetime | None = None,
    solved_at: datetime | None = SOLVED_AT,
):
    return reevaluate_mastery(
        current, solved_at=solved_at, retained_at=retained_at, windows=windows
    )


def test_a_pass_seven_days_after_solving_retains_a_consolidating_topic() -> None:
    closed = SOLVED_AT + timedelta(days=7, hours=1)

    change = reevaluate(Mastery.CONSOLIDATING, [window(closed, WindowStatus.PASSED)])

    assert change.mastery is Mastery.RETAINED
    assert change.retained_at == closed


def test_a_pass_under_seven_days_after_solving_does_not_retain() -> None:
    closed = SOLVED_AT + timedelta(days=6, hours=23)

    change = reevaluate(Mastery.CONSOLIDATING, [window(closed, WindowStatus.PASSED)])

    assert change.mastery is Mastery.CONSOLIDATING
    assert change.retained_at is None


def test_a_solved_topic_is_not_retained_by_a_window_alone() -> None:
    closed = SOLVED_AT + timedelta(days=10)

    change = reevaluate(Mastery.SOLVED, [window(closed, WindowStatus.PASSED)])

    assert change.mastery is Mastery.SOLVED


def test_a_partial_window_never_retains() -> None:
    closed = SOLVED_AT + timedelta(days=10)

    change = reevaluate(Mastery.CONSOLIDATING, [window(closed, WindowStatus.PARTIAL)])

    assert change.mastery is Mastery.CONSOLIDATING


def test_a_failed_window_after_retention_drops_to_consolidating_and_keeps_the_time() -> None:
    retained = SOLVED_AT + timedelta(days=8)
    failed = retained + timedelta(days=2)

    change = reevaluate(
        Mastery.RETAINED,
        [window(retained, WindowStatus.PASSED), window(failed, WindowStatus.FAILED)],
        retained_at=retained,
    )

    assert change.mastery is Mastery.CONSOLIDATING
    assert change.retained_at == retained


def test_a_failure_never_drops_a_topic_below_consolidating() -> None:
    failed = SOLVED_AT + timedelta(days=8)

    change = reevaluate(Mastery.CONSOLIDATING, [window(failed, WindowStatus.FAILED)])

    assert change.mastery is Mastery.CONSOLIDATING


def test_a_failure_on_a_solved_topic_leaves_it_solved() -> None:
    failed = SOLVED_AT + timedelta(days=8)

    change = reevaluate(Mastery.SOLVED, [window(failed, WindowStatus.FAILED)])

    assert change.mastery is Mastery.SOLVED


def test_a_pass_before_a_failure_does_not_count_after_it() -> None:
    passed = SOLVED_AT + timedelta(days=8)
    failed = passed + timedelta(days=1)

    change = reevaluate(
        Mastery.CONSOLIDATING,
        [window(passed, WindowStatus.PASSED), window(failed, WindowStatus.FAILED)],
    )

    assert change.mastery is Mastery.CONSOLIDATING


def test_a_pass_after_a_failure_retains_at_that_later_pass() -> None:
    passed = SOLVED_AT + timedelta(days=8)
    failed = passed + timedelta(days=1)
    repassed = failed + timedelta(days=5)

    change = reevaluate(
        Mastery.CONSOLIDATING,
        [
            window(passed, WindowStatus.PASSED),
            window(failed, WindowStatus.FAILED),
            window(repassed, WindowStatus.PASSED),
        ],
    )

    assert change.mastery is Mastery.RETAINED
    assert change.retained_at == repassed


def test_a_retained_topic_with_no_later_failure_keeps_its_time() -> None:
    retained = SOLVED_AT + timedelta(days=8)

    change = reevaluate(
        Mastery.RETAINED, [window(retained, WindowStatus.PASSED)], retained_at=retained
    )

    assert change.mastery is Mastery.RETAINED
    assert change.retained_at == retained


@pytest.mark.parametrize("current", [Mastery.UNSEEN, Mastery.INTRODUCED])
def test_topics_before_solved_are_never_changed_by_windows(current: Mastery) -> None:
    closed = SOLVED_AT + timedelta(days=10)

    change = reevaluate(current, [window(closed, WindowStatus.PASSED)], solved_at=None)

    assert change.mastery is current


# --- readiness views ---------------------------------------------------------------------------


def test_recent_timed_results_are_newest_first_and_limited_to_five() -> None:
    history = [window(PASS_AT + timedelta(days=d), WindowStatus.PASSED) for d in range(1, 8)]

    results = timed_results(history)

    assert len(results) == 5
    assert results[0].closed_at == PASS_AT + timedelta(days=7)
    assert results[0].used_s == 20 * 60
    assert results[0].budget_s == BUDGET_S


def test_abandoned_windows_are_not_timed_results() -> None:
    assert timed_results([window(PASS_AT, WindowStatus.ABANDONED)]) == ()


def test_upcoming_lists_the_soonest_due_first_and_limits_to_three() -> None:
    problems = [f"{TOPIC}/p0{n}-x" for n in range(1, 6)]
    passes = {p: PASS_AT + timedelta(days=n) for n, p in enumerate(problems)}
    schedules = build_schedules(passes, [])

    soonest = upcoming(schedules)

    assert [item.problem_id for item in soonest] == problems[:3]
