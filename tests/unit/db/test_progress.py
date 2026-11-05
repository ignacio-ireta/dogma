"""Per-topic progress from a snapshot: best result, counts, last attempt, and the week windows."""

from __future__ import annotations

from datetime import date, datetime

from dogma.db.models import Mastery
from dogma.db.progress import AttemptRow, Snapshot, attempts_between, topic_progress

TOPIC = "cml/vector-operations"
PROBLEM = f"{TOPIC}/p01-vector-projection"


def _local(day: int) -> datetime:
    """Noon on the given October day in this machine's zone, so the local date is always ``day``."""
    return datetime(2026, 10, day, 12, 0).astimezone()


def _row(
    passed: int,
    total: int,
    *,
    mode: str = "submit",
    status: str = "failed",
    day: int = 24,
    problem: str = PROBLEM,
) -> AttemptRow:
    return AttemptRow(
        problem_id=problem,
        mode=mode,
        status=status,
        passed=passed,
        total=total,
        submitted_at=_local(day),
    )


def test_a_topic_nobody_touched_is_unseen_with_nothing_to_show() -> None:
    progress = topic_progress(TOPIC, Snapshot(mastery={}, attempts=()))

    assert progress.mastery is Mastery.UNSEEN
    assert progress.attempts == 0
    assert progress.best is None
    assert progress.last_attempt_at is None
    assert not progress.solved


def test_a_passing_submit_is_the_best_result_even_against_a_perfect_run() -> None:
    snapshot = Snapshot(
        mastery={TOPIC: "solved"},
        attempts=(
            _row(4, 4, mode="run", status="passed"),
            _row(14, 15, status="failed"),
            _row(15, 15, status="passed"),
        ),
    )

    progress = topic_progress(TOPIC, snapshot)

    assert progress.best is not None
    assert (progress.best.passed, progress.best.total, progress.best.mode) == (15, 15, "submit")
    assert progress.attempts == 3
    assert progress.solved


def test_the_best_result_is_the_highest_share_of_tests_passed() -> None:
    snapshot = Snapshot(mastery={}, attempts=(_row(4, 4, mode="run", status="passed"), _row(5, 15)))

    progress = topic_progress(TOPIC, snapshot)

    assert progress.best is not None
    assert (progress.best.passed, progress.best.total, progress.best.mode) == (4, 4, "run")


def test_a_rejected_attempt_counts_as_an_attempt_but_not_as_a_result() -> None:
    snapshot = Snapshot(mastery={}, attempts=(_row(0, 0, status="rejected"),))

    progress = topic_progress(TOPIC, snapshot)

    assert progress.attempts == 1
    assert progress.best is None


def test_attempts_belong_to_their_topic_by_whole_path_segment() -> None:
    snapshot = Snapshot(
        mastery={},
        attempts=(_row(4, 4), _row(4, 4, problem="cml/vector-operations-extra/p01-other")),
    )

    assert topic_progress(TOPIC, snapshot).attempts == 1


def test_the_last_attempt_is_the_latest_one() -> None:
    snapshot = Snapshot(mastery={}, attempts=(_row(1, 4, day=24), _row(2, 4, day=26)))

    progress = topic_progress(TOPIC, snapshot)

    assert progress.last_attempt_at == _local(26)


def test_attempts_between_counts_from_the_start_day_up_to_but_not_including_the_end() -> None:
    snapshot = Snapshot(mastery={}, attempts=(_row(1, 4, day=24), _row(1, 4, day=26)))

    assert attempts_between(snapshot, date(2026, 10, 24), None) == 2
    assert attempts_between(snapshot, date(2026, 10, 25), date(2026, 10, 27)) == 1
    assert attempts_between(snapshot, date(2026, 10, 27), None) == 0
