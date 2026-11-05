"""Model round-trips, UTC timestamps, and the CHECK constraints that protect the two tables."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta, timezone

import pytest
from sqlalchemy.dialects import sqlite

from dogma.core.config import Settings
from dogma.db.base import UtcDateTime
from dogma.db.errors import StorageError, StorageErrorCode
from dogma.db.models import Attempt, Mastery, TopicState, mastery_rank
from dogma.db.session import database_path, open_session, upgrade_database
from dogma.db.session_models import SessionBlock, StudySession

WHEN = datetime(2026, 10, 24, 9, 0, tzinfo=UTC)


def _attempt(**overrides: object) -> Attempt:
    values: dict[str, object] = {
        "id": "a-1",
        "problem_id": "cml/vector-operations/p01-vector-projection",
        "problem_version": 2,
        "mode": "submit",
        "status": "passed",
        "public_passed": 4,
        "public_total": 4,
        "hidden_passed": 11,
        "hidden_total": 11,
        "hints_used": 0,
        "run_dir": "var/runs/a-1",
        "started_at": WHEN,
        "submitted_at": WHEN + timedelta(milliseconds=250),
        "duration_ms": 250,
    }
    values.update(overrides)
    return Attempt(**values)  # type: ignore[arg-type]


def test_attempt_and_topic_state_round_trip(settings: Settings) -> None:
    upgrade_database(settings)
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        session.add(_attempt())
        session.add(TopicState(topic_id="cml/vector-operations", mastery="solved", solved_at=WHEN))

    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        attempt = session.get(Attempt, "a-1")
        state = session.get(TopicState, "cml/vector-operations")
        assert attempt is not None
        assert state is not None
        assert attempt.problem_version == 2
        assert attempt.started_at == WHEN
        assert attempt.submitted_at == WHEN + timedelta(milliseconds=250)
        assert attempt.submitted_at.tzinfo is not None
        assert state.mastery == "solved"
        assert state.solved_at == WHEN
        assert state.introduced_at is None


def test_timestamps_are_stored_as_utc_text_whatever_offset_they_came_with(
    settings: Settings,
) -> None:
    upgrade_database(settings)
    local = WHEN.astimezone(timezone(timedelta(hours=2)))  # 11:00 at +02:00, 09:00 in UTC
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        session.add(_attempt(started_at=local, submitted_at=local))

    connection = sqlite3.connect(database_path(settings))
    try:
        stored = connection.execute("SELECT started_at FROM attempt").fetchone()[0]
    finally:
        connection.close()
    assert stored == "2026-10-24T09:00:00.000000+00:00"


def test_a_naive_datetime_is_refused_rather_than_guessed_at() -> None:
    naive = datetime(2026, 1, 1)

    with pytest.raises(ValueError, match="timezone-aware"):
        UtcDateTime().process_bind_param(naive, sqlite.dialect())


def test_the_database_refuses_an_unknown_attempt_mode(settings: Settings) -> None:
    upgrade_database(settings)

    with (
        pytest.raises(StorageError) as caught,
        open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session,
    ):
        session.add(_attempt(mode="practice"))

    assert caught.value.code is StorageErrorCode.DB_WRITE_FAILED


def test_the_database_refuses_a_mastery_state_that_is_never_stored(settings: Settings) -> None:
    upgrade_database(settings)

    with (
        pytest.raises(StorageError),
        open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session,
    ):
        session.add(TopicState(topic_id="cml/x", mastery=Mastery.UNSEEN.value))


def test_mastery_ranks_follow_the_path_in_architecture_order() -> None:
    order = [
        Mastery.UNSEEN,
        Mastery.INTRODUCED,
        Mastery.SOLVED,
        Mastery.CONSOLIDATING,
        Mastery.RETAINED,
        Mastery.INTERVIEW_READY,
    ]

    ranks = [mastery_rank(state.value) for state in order]

    assert ranks == sorted(ranks)
    assert len(set(ranks)) == len(order)
    with pytest.raises(ValueError, match="not-a-state"):
        mastery_rank("not-a-state")


def _session(**overrides: object) -> StudySession:
    values: dict[str, object] = {
        "id": "s-1",
        "plan_date": "2026-10-24",
        "slot": 1,
        "kind": "learn",
        "energy": "normal",
        "status": "open",
        "topic_id": "cml/vector-operations",
        "must_do": "cml/vector-operations/p01-vector-projection",
        "first_action": "Write your Predict guess",
        "plan_notes": None,
        "created_at": WHEN,
        "closed_at": None,
        "stopped_where": None,
        "next_step": None,
        "blocker": None,
        "self_answer": None,
    }
    values.update(overrides)
    return StudySession(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overrides",
    [
        {"status": "closed"},  # closed without a note, time or next step
        {"status": "closed", "closed_at": WHEN, "stopped_where": "here"},  # no next step
        {"kind": "practice"},
        {"energy": "high"},
        {"slot": 0},
    ],
    ids=["closed-without-note", "closed-without-next", "unknown-kind", "unknown-energy", "slot-0"],
)
def test_the_database_refuses_an_invalid_study_session(
    settings: Settings, overrides: dict[str, object]
) -> None:
    upgrade_database(settings)

    with (
        pytest.raises(StorageError) as caught,
        open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session,
    ):
        session.add(_session(**overrides))

    assert caught.value.code is StorageErrorCode.DB_WRITE_FAILED


def test_only_one_session_may_hold_a_slot_on_a_local_day(settings: Settings) -> None:
    upgrade_database(settings)
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        session.add(_session())

    with (
        pytest.raises(StorageError),
        open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session,
    ):
        session.add(_session(id="s-2"))


def test_a_session_block_needs_a_real_session_and_positive_minutes(settings: Settings) -> None:
    upgrade_database(settings)
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        session.add(_session())
        session.flush()
        session.add(
            SessionBlock(session_id="s-1", position=1, name="Learn", minutes=12, detail="Read it")
        )

    for bad in (
        SessionBlock(session_id="missing", position=2, name="Learn", minutes=5, detail="x"),
        SessionBlock(session_id="s-1", position=3, name="Learn", minutes=0, detail="x"),
    ):
        with (
            pytest.raises(StorageError),
            open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session,
        ):
            session.add(bad)


def test_a_closed_session_round_trips_its_transition_note(settings: Settings) -> None:
    upgrade_database(settings)
    closed = WHEN + timedelta(hours=1)
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        session.add(
            _session(
                status="closed",
                closed_at=closed,
                stopped_where="Practice, example 2",
                next_step="Solve example 3",
                blocker="numpy shape error",
                self_answer="A projection keeps the part along b.",
            )
        )

    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        row = session.get(StudySession, "s-1")
        assert row is not None
        assert row.closed_at == closed and row.closed_at.tzinfo is not None
        assert (row.stopped_where, row.next_step, row.blocker) == (
            "Practice, example 2",
            "Solve example 3",
            "numpy shape error",
        )
