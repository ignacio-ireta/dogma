"""Recording attempts and the mastery rules: a passing core submit solves, nothing moves back."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from tests.unit.db.support import ATTEMPT, PROBLEM, make_report

from dogma.core.config import Settings
from dogma.db.errors import StorageError, StorageErrorCode
from dogma.db.models import Attempt, AttemptMode, TopicState
from dogma.db.session import open_session, upgrade_database
from dogma.db.store import record_attempt, topic_of
from dogma.grading.models import RunStatus

TOPIC = "cml/vector-operations"
WHEN = datetime(2026, 10, 24, 9, 0, tzinfo=UTC)


def _record(
    settings: Settings,
    report_: object,
    *,
    mode: AttemptMode = AttemptMode.SUBMIT,
    core: bool = True,
) -> None:
    record_attempt(
        settings,
        report_,  # type: ignore[arg-type]
        mode=mode,
        core=core,
        started_at=WHEN,
        submitted_at=WHEN + timedelta(seconds=2),
    )


def _topic_state(settings: Settings, topic: str = TOPIC) -> TopicState | None:
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        state = session.get(TopicState, topic)
        if state is None:
            return None
        return TopicState(topic_id=state.topic_id, mastery=state.mastery, solved_at=state.solved_at)


def _seed_state(settings: Settings, mastery: str) -> None:
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        session.add(TopicState(topic_id=TOPIC, mastery=mastery, solved_at=WHEN))


def _attempts(settings: Settings) -> list[dict[str, object]]:
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        return [
            {
                "id": row.id,
                "mode": row.mode,
                "status": row.status,
                "public": (row.public_passed, row.public_total),
                "hidden": (row.hidden_passed, row.hidden_total),
                "run_dir": row.run_dir,
                "duration_ms": row.duration_ms,
            }
            for row in session.query(Attempt).order_by(Attempt.id)
        ]


@pytest.fixture
def ready(settings: Settings) -> Settings:
    upgrade_database(settings)
    return settings


def test_a_passing_submit_of_a_core_problem_marks_its_topic_solved(ready: Settings) -> None:
    _record(ready, make_report(public=(4, 4), hidden=(11, 11)), core=True)

    state = _topic_state(ready)
    assert state is not None
    assert state.mastery == "solved"
    assert state.solved_at == WHEN + timedelta(seconds=2)


def test_a_passing_run_never_marks_a_topic_solved(ready: Settings) -> None:
    _record(ready, make_report(public=(4, 4)), mode=AttemptMode.RUN, core=True)

    assert _topic_state(ready) is None


def test_a_passing_submit_of_a_non_core_problem_leaves_mastery_alone(ready: Settings) -> None:
    _record(ready, make_report(public=(4, 4), hidden=(11, 11)), core=False)

    assert _topic_state(ready) is None
    assert len(_attempts(ready)) == 1


def test_a_failing_submit_of_a_core_problem_does_not_solve_it(ready: Settings) -> None:
    _record(ready, make_report(RunStatus.FAILED, public=(4, 4), hidden=(9, 11)), core=True)

    assert _topic_state(ready) is None


def test_a_later_failure_never_downgrades_a_solved_topic(ready: Settings) -> None:
    _record(ready, make_report(public=(4, 4), hidden=(11, 11)), core=True)
    later = make_report(
        RunStatus.FAILED, public=(2, 4), hidden=(0, 11), attempt_id="20261024T090100Z-bbbbbbbb"
    )
    _record(ready, later, core=True)

    state = _topic_state(ready)
    assert state is not None
    assert state.mastery == "solved"
    assert len(_attempts(ready)) == 2


def test_a_topic_already_further_along_is_not_pulled_back_to_solved(ready: Settings) -> None:
    _seed_state(ready, "consolidating")

    _record(ready, make_report(public=(4, 4), hidden=(11, 11)), core=True)

    state = _topic_state(ready)
    assert state is not None
    assert state.mastery == "consolidating"


def test_a_topic_at_introduced_moves_forward_to_solved(ready: Settings) -> None:
    _seed_state(ready, "introduced")

    _record(ready, make_report(public=(4, 4), hidden=(11, 11)), core=True)

    state = _topic_state(ready)
    assert state is not None
    assert state.mastery == "solved"


def test_each_attempt_stores_public_and_hidden_counts_and_its_mode(ready: Settings) -> None:
    _record(ready, make_report(RunStatus.FAILED, public=(4, 4), hidden=(10, 11)))

    [row] = _attempts(ready)
    assert row["mode"] == "submit"
    assert row["status"] == "failed"
    assert row["public"] == (4, 4)
    assert row["hidden"] == (10, 11)
    assert row["duration_ms"] == 2000


def test_a_rejected_attempt_ran_nothing_so_it_has_no_run_dir(ready: Settings) -> None:
    _record(ready, make_report(RunStatus.REJECTED, public=(0, 0)), core=True)

    [row] = _attempts(ready)
    assert row["status"] == "rejected"
    assert row["run_dir"] is None
    assert row["public"] == (0, 0)


def test_the_run_dir_is_stored_relative_to_the_repo(ready: Settings) -> None:
    _record(ready, make_report(public=(4, 4)), mode=AttemptMode.RUN)

    [row] = _attempts(ready)
    assert row["run_dir"] == f"var/runs/{ATTEMPT}"


def test_the_topic_is_the_problem_id_without_its_last_part() -> None:
    assert topic_of(PROBLEM) == TOPIC


def test_a_database_that_cannot_be_created_is_a_storage_error(settings: Settings) -> None:
    settings.repo_root.mkdir(parents=True)
    settings.var_dir.write_text("a file where the folder should be")

    with pytest.raises(StorageError) as caught:
        _record(settings, make_report(public=(4, 4)))

    assert caught.value.code is StorageErrorCode.DB_UNAVAILABLE


def test_a_corrupt_database_is_a_migration_storage_error(settings: Settings) -> None:
    path: Path = settings.var_dir / "dogma.db"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not sqlite" * 50)

    with pytest.raises(StorageError) as caught:
        _record(settings, make_report(public=(4, 4)))

    assert caught.value.code is StorageErrorCode.DB_MIGRATION_FAILED
