"""Sessions: plan, resume, re-plan, close once, and carry the transition note to the next day."""

from __future__ import annotations

import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select
from tests.unit.learning.support import (
    ALPHA,
    BETA,
    DELTA,
    NOON,
    build_repo,
    core_problem,
    solve,
    utc_timezone,  # noqa: F401  autouse: pins the zone to UTC for these tests
)

from dogma.core.config import Settings
from dogma.core.errors import DogmaError, ErrorCode
from dogma.db.engine import make_engine
from dogma.db.errors import StorageError, StorageErrorCode
from dogma.db.models import Mastery, TopicState
from dogma.db.progress import read_snapshot, topic_progress
from dogma.db.review_models import Card
from dogma.db.session import database_path, open_session
from dogma.db.session_models import SessionBlock, StudySession
from dogma.learning import sessions
from dogma.learning.planner import Energy, SessionKind
from dogma.learning.sessions import CloseNote, CloseResult

TOMORROW = NOON + timedelta(days=1)
LEARN_BLOCKS = ["Retrieve", "Predict", "Learn", "Practice", "Close"]


@pytest.fixture
def repo(tmp_path: Path) -> Settings:
    return build_repo(tmp_path / "repo")


def rows(settings: Settings) -> dict[str, int]:
    """Row counts for the tables these tests read. Read-only."""
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as db:
        return {
            "sessions": db.scalar(select(func.count()).select_from(StudySession)) or 0,
            "blocks": db.scalar(select(func.count()).select_from(SessionBlock)) or 0,
            "topic_states": db.scalar(select(func.count()).select_from(TopicState)) or 0,
            "cards": db.scalar(select(func.count()).select_from(Card)) or 0,
        }


def mastery_of(settings: Settings, topic_id: str) -> str | None:
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as db:
        state = db.get(TopicState, topic_id)
        return None if state is None else state.mastery


def close_current(settings: Settings, *, now: datetime, answer: str | None = None) -> CloseResult:
    target = sessions.close_target(settings, now=now)
    note = CloseNote(stopped_where="Predict done", next_step="Read example 1")
    return sessions.close_session(settings, target.session_id, note, answer=answer, now=now)


def close_with(
    settings: Settings,
    *,
    now: datetime,
    stopped: str,
    step: str,
    blocker: str | None = None,
) -> CloseResult:
    target = sessions.close_target(settings, now=now)
    note = CloseNote(stopped_where=stopped, next_step=step, blocker=blocker)
    return sessions.close_session(settings, target.session_id, note, answer=None, now=now)


# --- planning and resuming -------------------------------------------------------------------


def test_the_first_today_plans_a_learn_session_with_one_first_action(repo: Settings) -> None:
    view = sessions.today(repo, now=NOON)

    assert view.last_note is None
    assert view.resumed is False and view.replanned is False
    assert view.current is not None
    plan = view.current.plan
    assert view.current.slot == 1
    assert plan.kind is SessionKind.LEARN
    assert plan.topic_id == ALPHA
    assert plan.must_do == core_problem(ALPHA)
    assert [block.name for block in plan.blocks] == LEARN_BLOCKS
    assert plan.minutes == 45
    assert plan.first_action == (
        f"Write your Predict guess (3 min), then open content/{ALPHA}/lesson.md."
    )


def test_planning_never_marks_anything_introduced_or_solved(repo: Settings) -> None:
    sessions.today(repo, now=NOON)

    assert rows(repo)["topic_states"] == 0


def test_repeating_today_resumes_the_same_plan_without_a_duplicate(repo: Settings) -> None:
    first = sessions.today(repo, now=NOON)
    second = sessions.today(repo, now=NOON + timedelta(hours=2))

    assert first.current is not None and second.current is not None
    assert second.resumed is True and second.replanned is False
    assert second.current.session_id == first.current.session_id
    assert second.current.plan == first.current.plan
    assert rows(repo)["sessions"] == 1
    assert rows(repo)["blocks"] == 5


def test_a_resumed_plan_is_stable_when_its_content_changes_afterwards(repo: Settings) -> None:
    first = sessions.today(repo, now=NOON)
    topic_file = repo.content_dir / ALPHA / "topic.yaml"
    topic_file.write_text(topic_file.read_text().replace("status: reviewed", "status: draft"))

    again = sessions.today(repo, now=NOON + timedelta(minutes=5))

    assert first.current is not None and again.current is not None
    assert again.current.plan == first.current.plan


def test_the_stored_blocks_keep_their_order(repo: Settings) -> None:
    sessions.today(repo, now=NOON)

    with open_session(repo, on_failure=StorageErrorCode.DB_READ_FAILED) as db:
        names = db.scalars(select(SessionBlock.name).order_by(SessionBlock.position)).all()
    assert list(names) == LEARN_BLOCKS


# --- closing ---------------------------------------------------------------------------------


def test_closing_introduces_the_learn_topic_and_enrolls_its_cards(repo: Settings) -> None:
    sessions.today(repo, now=NOON)

    result = close_current(repo, now=NOON + timedelta(hours=1), answer="A shadow on b.")

    assert result.already_closed is False
    assert result.introduced_topic == ALPHA
    assert result.new_cards == 2
    assert mastery_of(repo, ALPHA) == Mastery.INTRODUCED
    assert rows(repo)["cards"] == 2


def test_closing_never_invents_a_solve(repo: Settings) -> None:
    sessions.today(repo, now=NOON)

    close_current(repo, now=NOON + timedelta(hours=1), answer="Anything at all.")

    assert topic_progress(ALPHA, read_snapshot(repo)).solved is False
    assert mastery_of(repo, ALPHA) == Mastery.INTRODUCED


def test_the_close_stores_where_you_stopped_the_next_step_blocker_and_answer(
    repo: Settings,
) -> None:
    sessions.today(repo, now=NOON)
    close_with(
        repo,
        now=NOON + timedelta(hours=1),
        stopped="Practice, example 2",
        step="Write the residual check",
        blocker="numpy shape error",
    )
    with open_session(repo, on_failure=StorageErrorCode.DB_READ_FAILED) as db:
        row = db.scalars(select(StudySession)).one()
        stored = (
            row.status,
            row.closed_at,
            row.stopped_where,
            row.next_step,
            row.blocker,
            row.self_answer,
        )

    assert stored == (
        "closed",
        NOON + timedelta(hours=1),
        "Practice, example 2",
        "Write the residual check",
        "numpy shape error",
        None,
    )


def test_a_second_close_keeps_the_first_note_and_changes_nothing(repo: Settings) -> None:
    sessions.today(repo, now=NOON)
    first = close_current(repo, now=NOON + timedelta(hours=1), answer="First answer.")
    before = rows(repo)

    second = close_with(
        repo,
        now=NOON + timedelta(hours=2),
        stopped="Something else",
        step="A different step",
    )

    assert second.already_closed is True
    assert second.introduced_topic is None and second.new_cards == 0
    assert second.note.next_step == first.note.next_step == "Read example 1"
    assert second.note.closed_at == NOON + timedelta(hours=1)
    assert rows(repo) == before
    with open_session(repo, on_failure=StorageErrorCode.DB_READ_FAILED) as db:
        assert db.scalars(select(StudySession)).one().self_answer == "First answer."


def test_a_closed_consolidate_session_introduces_nothing(repo: Settings) -> None:
    solve(repo, ALPHA, when=NOON - timedelta(days=3))
    sessions.today(repo, now=NOON)
    close_current(repo, now=NOON + timedelta(minutes=10))
    sessions.today(repo, now=NOON + timedelta(hours=1))
    close_current(repo, now=NOON + timedelta(hours=1, minutes=10))
    view = sessions.today(repo, now=NOON + timedelta(hours=2))
    assert view.current is not None and view.current.plan.kind is SessionKind.CONSOLIDATE
    states_before = rows(repo)["topic_states"]

    result = close_current(repo, now=NOON + timedelta(hours=3))

    assert result.introduced_topic is None
    assert rows(repo)["topic_states"] == states_before


def test_closing_without_a_session_today_is_a_structured_error(repo: Settings) -> None:
    with pytest.raises(DogmaError) as caught:
        sessions.close_target(repo, now=NOON)

    assert caught.value.code is ErrorCode.SESSION_NOT_FOUND


def test_closing_an_unknown_session_is_a_structured_error(repo: Settings) -> None:
    sessions.today(repo, now=NOON)

    with pytest.raises(DogmaError) as caught:
        sessions.close_session(
            repo, "no-such-session", CloseNote("here", "next"), answer=None, now=NOON
        )

    assert caught.value.code is ErrorCode.SESSION_NOT_FOUND


@pytest.mark.parametrize(
    ("note", "code"),
    [
        (CloseNote(stopped_where="   ", next_step="Next"), ErrorCode.SESSION_NOTE_INVALID),
        (CloseNote(stopped_where="Here", next_step=""), ErrorCode.SESSION_NOTE_INVALID),
        (CloseNote(stopped_where="x" * 501, next_step="Next"), ErrorCode.SESSION_NOTE_INVALID),
        (
            CloseNote(stopped_where="Here", next_step="Next", blocker="x" * 501),
            ErrorCode.SESSION_NOTE_INVALID,
        ),
    ],
    ids=["blank-stop", "blank-next", "blocker-too-long", "stop-too-long"],
)
def test_invalid_notes_are_refused_and_change_nothing(
    repo: Settings, note: CloseNote, code: ErrorCode
) -> None:
    sessions.today(repo, now=NOON)
    session_id = sessions.close_target(repo, now=NOON).session_id

    with pytest.raises(DogmaError) as caught:
        sessions.close_session(repo, session_id, note, answer=None, now=NOON)

    assert caught.value.code is code
    assert sessions.close_target(repo, now=NOON).stored is None


def test_a_failed_close_leaves_the_session_open_and_nothing_introduced(repo: Settings) -> None:
    sessions.today(repo, now=NOON)
    engine = make_engine(database_path(repo))
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "CREATE TRIGGER reject_intro BEFORE INSERT ON topic_state "
                "BEGIN SELECT RAISE(ABORT, 'injected write failure'); END"
            )
    finally:
        engine.dispose()

    with pytest.raises(StorageError) as caught:
        close_current(repo, now=NOON + timedelta(hours=1))

    assert caught.value.code is StorageErrorCode.DB_WRITE_FAILED
    assert sessions.close_target(repo, now=NOON).stored is None
    assert rows(repo)["topic_states"] == 0


# --- the transition note and the day -----------------------------------------------------------


def test_the_transition_note_leads_the_next_session(repo: Settings) -> None:
    sessions.today(repo, now=NOON)
    close_with(
        repo,
        now=NOON + timedelta(hours=1),
        stopped="Practice, example 2",
        step="Solve example 3",
        blocker="numpy shape error",
    )

    view = sessions.today(repo, now=TOMORROW)

    assert view.last_note is not None
    assert view.last_note.topic_id == ALPHA
    assert view.last_note.stopped_where == "Practice, example 2"
    assert view.last_note.next_step == "Solve example 3"
    assert view.last_note.blocker == "numpy shape error"
    assert view.resumed is False
    assert view.current is not None and view.current.slot == 1
    assert view.current.plan.topic_id == ALPHA  # introduced, not yet solved: continue it


def test_the_second_session_of_a_day_takes_a_different_topic(repo: Settings) -> None:
    sessions.today(repo, now=NOON)
    close_current(repo, now=NOON + timedelta(hours=1))

    view = sessions.today(repo, now=NOON + timedelta(hours=2))

    assert view.current is not None
    assert view.current.slot == 2
    assert view.current.plan.topic_id == DELTA  # BETA waits for ALPHA; GAMMA is a draft


def test_a_prerequisite_unlocks_only_after_a_real_solve(repo: Settings) -> None:
    sessions.today(repo, now=NOON)
    close_current(repo, now=NOON + timedelta(hours=1))
    sessions.today(repo, now=TOMORROW)
    close_current(repo, now=TOMORROW + timedelta(hours=1))
    assert mastery_of(repo, ALPHA) == Mastery.INTRODUCED

    # Introduced ALPHA is planned again on the first slot; a planned or introduced prerequisite
    # never unlocks BETA, so the second slot of that day takes DELTA and BETA never appears.
    second_slot = sessions.today(repo, now=TOMORROW + timedelta(hours=2))
    assert second_slot.current is not None and second_slot.current.plan.topic_id == DELTA

    solve(repo, ALPHA, when=TOMORROW + timedelta(hours=3))
    # Finish the current DELTA task as well; an unfinished introduced topic takes
    # priority over starting a newly unlocked one.
    solve(repo, DELTA, when=TOMORROW + timedelta(hours=3))
    close_current(repo, now=TOMORROW + timedelta(hours=3, minutes=1))
    unlocked = sessions.today(repo, now=TOMORROW + timedelta(days=1))

    assert unlocked.current is not None
    assert unlocked.current.plan.topic_id == BETA


def test_low_energy_plans_minimum_and_introduces_nothing(repo: Settings) -> None:
    view = sessions.today(repo, now=NOON, energy=Energy.LOW)

    assert view.current is not None
    plan = view.current.plan
    assert plan.kind is SessionKind.MINIMUM
    assert plan.topic_id is None and plan.must_do is None
    assert [block.name for block in plan.blocks] == ["Review", "Re-solve"]
    assert plan.minutes == 20
    assert rows(repo)["topic_states"] == 0

    result = close_current(repo, now=NOON + timedelta(hours=1))

    assert result.introduced_topic is None
    assert rows(repo)["topic_states"] == 0


def test_energy_on_an_open_session_replans_it_in_place(repo: Settings) -> None:
    normal = sessions.today(repo, now=NOON)
    low = sessions.today(repo, now=NOON + timedelta(minutes=5), energy=Energy.LOW)

    assert normal.current is not None and low.current is not None
    assert low.replanned is True and low.resumed is False
    assert low.current.session_id == normal.current.session_id
    assert low.current.plan.kind is SessionKind.MINIMUM
    counts = rows(repo)
    assert counts["sessions"] == 1
    assert counts["blocks"] == 2


def test_an_open_session_keeps_its_energy_when_no_flag_is_given(repo: Settings) -> None:
    sessions.today(repo, now=NOON, energy=Energy.LOW)

    again = sessions.today(repo, now=NOON + timedelta(minutes=5))

    assert again.resumed is True
    assert again.current is not None and again.current.plan.kind is SessionKind.MINIMUM


def test_the_day_is_complete_after_its_three_sessions_are_closed(repo: Settings) -> None:
    for minutes in (0, 60, 120):
        sessions.today(repo, now=NOON + timedelta(minutes=minutes))
        close_current(repo, now=NOON + timedelta(minutes=minutes + 10))

    view = sessions.today(repo, now=NOON + timedelta(hours=4))

    assert view.current is None
    assert view.closed_today == 3 and view.sessions_per_day == 3
    assert rows(repo)["sessions"] == 3


def test_an_unclosed_session_is_resumed_across_midnight(repo: Settings) -> None:
    first = sessions.today(repo, now=NOON)

    tomorrow = sessions.today(repo, now=TOMORROW)

    assert first.current is not None and tomorrow.current is not None
    assert tomorrow.resumed is True
    assert tomorrow.current.session_id == first.current.session_id
    assert tomorrow.current.plan_date == sessions.local_day(NOON)
    assert tomorrow.current.slot == 1
    assert rows(repo)["sessions"] == 1


def test_a_learn_session_counts_the_cards_that_are_due_the_next_day(repo: Settings) -> None:
    sessions.today(repo, now=NOON)
    close_current(repo, now=NOON + timedelta(hours=1))

    view = sessions.today(repo, now=TOMORROW)

    assert view.current is not None
    assert "2 due now" in view.current.plan.blocks[0].detail
    assert view.current.plan.first_action == "dogma review --limit 2"


def test_consolidate_re_solves_a_problem_the_learner_has_passed(repo: Settings) -> None:
    solve(repo, ALPHA, when=NOON - timedelta(days=3))
    sessions.today(repo, now=NOON)
    close_current(repo, now=NOON + timedelta(hours=1))
    sessions.today(repo, now=NOON + timedelta(hours=2))
    close_current(repo, now=NOON + timedelta(hours=3))

    view = sessions.today(repo, now=NOON + timedelta(hours=4))

    assert view.current is not None
    assert view.current.plan.kind is SessionKind.CONSOLIDATE
    assert core_problem(ALPHA) in view.current.plan.blocks[1].detail


def test_timed_target_survives_resume_and_is_cleared_by_low_energy_replanning(
    repo: Settings,
) -> None:
    solve(repo, ALPHA, when=NOON - timedelta(days=8))
    for hour in (0, 2):
        sessions.today(repo, now=NOON + timedelta(hours=hour))
        close_current(repo, now=NOON + timedelta(hours=hour + 1))
    planned = sessions.today(repo, now=NOON + timedelta(hours=4))
    assert planned.current is not None
    assert planned.current.plan.timed_problem_id == core_problem(ALPHA)
    resumed = sessions.today(repo, now=NOON + timedelta(hours=4, minutes=1))
    assert resumed.current is not None and resumed.resumed
    assert resumed.current.plan.timed_problem_id == core_problem(ALPHA)
    low = sessions.today(repo, now=NOON + timedelta(hours=4, minutes=2), energy=Energy.LOW)
    assert low.current is not None and low.replanned
    assert low.current.plan.timed_problem_id is None
    resumed_low = sessions.today(repo, now=NOON + timedelta(hours=4, minutes=3))
    assert resumed_low.current is not None
    assert resumed_low.current.plan.timed_problem_id is None


def test_a_missing_core_path_is_a_structured_error(repo: Settings) -> None:
    (repo.curriculum_dir / "paths" / "core.yaml").unlink()

    with pytest.raises(DogmaError) as caught:
        sessions.today(repo, now=NOON)

    assert caught.value.code is ErrorCode.CONTENT_NOT_FOUND


def test_naive_times_are_refused_before_anything_is_written(repo: Settings) -> None:
    naive = NOON.replace(tzinfo=None)

    with pytest.raises(DogmaError) as today_error:
        sessions.today(repo, now=naive)
    assert today_error.value.code is ErrorCode.SESSION_TIME_INVALID
    assert not database_path(repo).exists()

    with pytest.raises(DogmaError) as close_error:
        sessions.close_session(repo, "x", CloseNote("a", "b"), answer=None, now=naive)
    assert close_error.value.code is ErrorCode.SESSION_TIME_INVALID


def test_closing_after_midnight_keeps_the_original_date_and_enrolls_cards(repo: Settings) -> None:
    first = sessions.today(repo, now=NOON)

    target = sessions.close_target(repo, now=TOMORROW)
    result = close_current(repo, now=TOMORROW)

    assert first.current is not None
    assert target.session_id == first.current.session_id
    assert target.plan_date == sessions.local_day(NOON)
    assert result.note.plan_date == sessions.local_day(NOON)
    assert result.new_cards == 2
    assert mastery_of(repo, ALPHA) == Mastery.INTRODUCED
    assert sessions.close_target(repo, now=TOMORROW).stored == result.note
    next_session = sessions.today(repo, now=TOMORROW + timedelta(minutes=1))
    assert next_session.last_note == result.note
    assert next_session.current is not None
    assert next_session.current.plan_date == sessions.local_day(TOMORROW)
    assert next_session.current.slot == 1


def test_failed_card_enrollment_rolls_back_close_and_introduction(repo: Settings) -> None:
    sessions.today(repo, now=NOON)
    engine = make_engine(database_path(repo))
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "CREATE TRIGGER reject_card BEFORE INSERT ON card "
                "BEGIN SELECT RAISE(ABORT, 'injected card failure'); END"
            )
    finally:
        engine.dispose()

    with pytest.raises(StorageError) as caught:
        close_current(repo, now=NOON + timedelta(hours=1))

    assert caught.value.code is StorageErrorCode.DB_WRITE_FAILED
    assert sessions.close_target(repo, now=NOON).stored is None
    assert rows(repo)["topic_states"] == 0
    assert rows(repo)["cards"] == 0
    engine = make_engine(database_path(repo))
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql("DROP TRIGGER reject_card")
    finally:
        engine.dispose()
    retried = close_current(repo, now=NOON + timedelta(hours=2))
    assert retried.already_closed is False
    assert retried.new_cards == 2


@pytest.mark.parametrize("damage", ["missing", "invalid"])
def test_a_path_topic_without_a_usable_package_is_skipped_not_planned(
    repo: Settings, damage: str
) -> None:
    folder = repo.content_dir / DELTA
    if damage == "missing":
        shutil.rmtree(folder)
    else:
        (folder / "topic.yaml").write_text("id: [unclosed\n", encoding="utf-8")
    sessions.today(repo, now=NOON)
    close_current(repo, now=NOON + timedelta(hours=1))

    view = sessions.today(repo, now=NOON + timedelta(hours=2))

    assert view.current is not None
    assert view.current.slot == 2
    assert view.current.plan.kind is SessionKind.CONSOLIDATE
    assert view.current.plan.topic_id is None
