"""Interview answers: the interview-ready rule, its storage, and how drops and re-promotion behave.

The pure rule is tested on its own. The storage tests run the real timed re-solve and card paths,
so a failed window or a card lapse is what drops interview-ready, not a hand-set state.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select
from tests.unit.db.support import PROBLEM, make_report

from dogma.content.rubrics import load_rubric_version
from dogma.core.config import Settings
from dogma.core.errors import DogmaError, ErrorCode
from dogma.db.answer_models import Answer, AnswerKind
from dogma.db.errors import StorageErrorCode
from dogma.db.models import AttemptMode, Mastery, TopicState
from dogma.db.resolve_store import start_window
from dogma.db.session import database_path, open_session
from dogma.db.store import record_attempt
from dogma.grading.models import RunStatus
from dogma.learning.answers import readiness_change, readiness_reason, record_answer
from dogma.learning.cards import due_cards, sync_cards
from dogma.learning.review import rate_card

REPO = Path(__file__).resolve().parents[3]
TOPIC = "cml/vector-operations"
PROMPT = "interview.md#Q1"
NOW = datetime(2026, 10, 27, 9, tzinfo=UTC)
SOLVED_AT = NOW - timedelta(days=12)
RETAINED_AT = NOW - timedelta(days=3)
READY = {"accuracy": 3, "structure": 3, "trade-offs": 3, "communication": 4}
WEAK_TRADE_OFFS = {"accuracy": 4, "structure": 4, "trade-offs": 2, "communication": 4}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """A scratch repo with the example topic and the real rubric files. Never the real var/."""
    root = tmp_path / "repo"
    shutil.copytree(REPO / "content" / TOPIC, root / "content" / TOPIC)
    shutil.copytree(REPO / "curriculum" / "rubrics", root / "curriculum" / "rubrics")
    return Settings(repo_root=root, _env_file=None)


def _set_state(settings: Settings, mastery: Mastery) -> None:
    """Put the topic in a mastery state directly, for tests that are not about how it got there."""
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        session.add(
            TopicState(
                topic_id=TOPIC,
                mastery=mastery.value,
                solved_at=SOLVED_AT,
                retained_at=RETAINED_AT if mastery is not Mastery.CONSOLIDATING else None,
            )
        )


def _state(settings: Settings) -> tuple[str, datetime | None]:
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        state = session.get(TopicState, TOPIC)
        assert state is not None
        return state.mastery, state.ready_at


def _answer_count(settings: Settings) -> int:
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        return session.scalar(select(func.count()).select_from(Answer)) or 0


def _answer(settings: Settings, scores: dict[str, int], when: datetime = NOW) -> None:
    record_answer(
        settings,
        topic_id=TOPIC,
        prompt_ref=PROMPT,
        text="A dot product measures agreement in direction.",
        scores=scores,
        now=when,
    )


def _timed(settings: Settings, when: datetime, *, passed: bool, attempt: str) -> None:
    """A window opened five minutes before ``when`` and decided by a submit at ``when``."""
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        start_window(
            session,
            problem_id=PROBLEM,
            problem_version=2,
            budget_s=1200,
            now=when - timedelta(minutes=5),
        )
    report = (
        make_report(RunStatus.PASSED, public=(4, 4), hidden=(11, 11), attempt_id=attempt)
        if passed
        else make_report(RunStatus.FAILED, public=(2, 4), hidden=(3, 11), attempt_id=attempt)
    )
    record_attempt(
        settings,
        report,
        mode=AttemptMode.SUBMIT,
        core=True,
        started_at=when - timedelta(minutes=6),
        submitted_at=when,
    )


def _lapse(settings: Settings, when: datetime) -> str:
    sync_cards(settings, now=when)
    card = due_cards(settings, now=when)[0]
    result = rate_card(
        settings,
        card.card_id,
        1,
        now=when,
        expected_due=card.due,
        expected_content_hash=card.content_hash,
    )
    return result.mastery.value


# --- the pure rule ----------------------------------------------------------------------------


@pytest.fixture
def rubric_ready() -> int:
    rubric = load_rubric_version(REPO / "curriculum", "topic-answer", 1)
    assert rubric.ready_score is not None
    return rubric.ready_score


def test_a_retained_topic_with_every_dimension_at_three_becomes_interview_ready(
    rubric_ready: int,
) -> None:
    assert readiness_change(Mastery.RETAINED, READY, rubric_ready) is Mastery.INTERVIEW_READY


def test_one_dimension_below_three_keeps_a_retained_topic_retained(rubric_ready: int) -> None:
    after = readiness_change(Mastery.RETAINED, WEAK_TRADE_OFFS, rubric_ready)

    assert after is Mastery.RETAINED


@pytest.mark.parametrize(
    "lower", [Mastery.UNSEEN, Mastery.INTRODUCED, Mastery.SOLVED, Mastery.CONSOLIDATING]
)
def test_no_state_below_retained_is_ever_promoted_by_an_answer(
    lower: Mastery, rubric_ready: int
) -> None:
    perfect = dict.fromkeys(READY, 4)

    assert readiness_change(lower, perfect, rubric_ready) is lower


def test_a_weak_answer_never_lowers_an_interview_ready_topic(rubric_ready: int) -> None:
    assert (
        readiness_change(Mastery.INTERVIEW_READY, dict.fromkeys(READY, 1), rubric_ready)
        is Mastery.INTERVIEW_READY
    )


def test_the_reason_names_the_dimensions_that_held_a_topic_back(rubric_ready: int) -> None:
    reason = readiness_reason(
        Mastery.RETAINED,
        Mastery.RETAINED,
        WEAK_TRADE_OFFS,
        rubric_ready,
    )

    assert "trade-offs" in reason and "below 3" in reason


# --- recording an answer ----------------------------------------------------------------------


def test_an_answer_is_stored_with_its_rubric_id_version_and_scores(settings: Settings) -> None:
    _set_state(settings, Mastery.CONSOLIDATING)

    result = record_answer(
        settings,
        topic_id=TOPIC,
        prompt_ref=PROMPT,
        text="  A dot product measures agreement.  ",
        scores=READY,
        notes="Good shape",
        now=NOW,
    )

    assert result.spoken is False
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        stored = session.scalars(select(Answer)).one()
        row = (
            stored.kind,
            stored.topic_id,
            stored.prompt_ref,
            stored.text,
            stored.rubric_id,
            stored.rubric_version,
            stored.scores,
            stored.notes,
            stored.created_at,
        )
    assert row == (
        AnswerKind.INTERVIEW.value,
        TOPIC,
        PROMPT,
        "A dot product measures agreement.",
        "topic-answer",
        1,
        READY,
        "Good shape",
        NOW,
    )


def test_a_spoken_answer_is_stored_with_no_text(settings: Settings) -> None:
    result = record_answer(
        settings, topic_id=TOPIC, prompt_ref=PROMPT, text=None, scores=READY, now=NOW
    )

    assert result.spoken is True
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        assert session.scalars(select(Answer.text)).one() is None


@pytest.mark.parametrize(
    "scores",
    [
        {"accuracy": 3, "structure": 3, "trade-offs": 3},  # communication missing
        {"accuracy": 5, "structure": 3, "trade-offs": 3, "communication": 3},  # out of range
        {"accuracy": 3, "structure": 3, "trade-offs": 3, "communication": 3, "flair": 4},
    ],
)
def test_scores_that_do_not_fit_the_rubric_are_refused_and_nothing_is_stored(
    settings: Settings, scores: dict[str, int]
) -> None:
    with pytest.raises(DogmaError) as caught:
        record_answer(settings, topic_id=TOPIC, prompt_ref=PROMPT, text="x", scores=scores, now=NOW)

    assert caught.value.code is ErrorCode.ANSWER_SCORE_INVALID
    assert _answer_count(settings) == 0


@pytest.mark.parametrize("text", ["", "   "])
def test_an_empty_written_answer_is_refused(settings: Settings, text: str) -> None:
    with pytest.raises(DogmaError) as caught:
        record_answer(settings, topic_id=TOPIC, prompt_ref=PROMPT, text=text, scores=READY, now=NOW)

    assert caught.value.code is ErrorCode.ANSWER_TEXT_INVALID


def test_recording_switched_off_refuses_and_creates_no_database(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    shutil.copytree(REPO / "curriculum" / "rubrics", root / "curriculum" / "rubrics")
    off = Settings(repo_root=root, record_attempts=False, _env_file=None)

    with pytest.raises(DogmaError) as caught:
        record_answer(off, topic_id=TOPIC, prompt_ref=PROMPT, text="x", scores=READY, now=NOW)

    assert caught.value.code is ErrorCode.ANSWER_RECORDING_OFF
    assert not database_path(off).exists()


def test_a_topic_that_does_not_exist_is_a_content_error(settings: Settings) -> None:
    with pytest.raises(DogmaError) as caught:
        record_answer(
            settings, topic_id="zzz/nothing", prompt_ref=PROMPT, text="x", scores=READY, now=NOW
        )

    assert caught.value.code is ErrorCode.CONTENT_NOT_FOUND


def test_an_unseen_topic_gets_no_state_row_from_an_answer(settings: Settings) -> None:
    result = record_answer(
        settings, topic_id=TOPIC, prompt_ref=PROMPT, text="x", scores=READY, now=NOW
    )

    assert result.before is Mastery.UNSEEN
    assert result.after is Mastery.UNSEEN
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        assert session.get(TopicState, TOPIC) is None


# --- interview-ready on the real paths --------------------------------------------------------


def test_a_qualifying_answer_on_a_retained_topic_sets_interview_ready_and_ready_at(
    settings: Settings,
) -> None:
    _set_state(settings, Mastery.RETAINED)

    result = record_answer(
        settings, topic_id=TOPIC, prompt_ref=PROMPT, text="x", scores=READY, now=NOW
    )

    assert (result.before, result.after) == (Mastery.RETAINED, Mastery.INTERVIEW_READY)
    assert result.ready_at == NOW
    assert _state(settings) == (Mastery.INTERVIEW_READY.value, NOW)
    assert "Interview-ready" in result.reason


def test_a_weak_answer_on_a_retained_topic_leaves_it_retained(settings: Settings) -> None:
    _set_state(settings, Mastery.RETAINED)

    result = record_answer(
        settings, topic_id=TOPIC, prompt_ref=PROMPT, text="x", scores=WEAK_TRADE_OFFS, now=NOW
    )

    assert result.after is Mastery.RETAINED
    assert _state(settings) == (Mastery.RETAINED.value, None)
    assert "trade-offs" in result.reason


def test_a_perfect_answer_while_consolidating_does_not_count_once_retained_later(
    settings: Settings,
) -> None:
    _set_state(settings, Mastery.CONSOLIDATING)
    perfect = dict.fromkeys(READY, 4)
    _answer(settings, perfect)
    assert _state(settings)[0] == Mastery.CONSOLIDATING.value

    _timed(settings, NOW, passed=True, attempt="retaining-pass")

    assert _state(settings)[0] == Mastery.RETAINED.value


def test_a_failed_timed_resolve_drops_interview_ready_to_consolidating(settings: Settings) -> None:
    _set_state(settings, Mastery.RETAINED)
    record_answer(settings, topic_id=TOPIC, prompt_ref=PROMPT, text="x", scores=READY, now=NOW)
    assert _state(settings)[0] == Mastery.INTERVIEW_READY.value

    _timed(settings, NOW + timedelta(days=1), passed=False, attempt="failed-window")

    assert _state(settings)[0] == Mastery.CONSOLIDATING.value


def test_a_card_lapse_drops_interview_ready_to_consolidating(settings: Settings) -> None:
    _set_state(settings, Mastery.RETAINED)
    record_answer(settings, topic_id=TOPIC, prompt_ref=PROMPT, text="x", scores=READY, now=NOW)

    mastery = _lapse(settings, NOW + timedelta(minutes=5))

    assert mastery == Mastery.CONSOLIDATING.value
    assert _state(settings)[0] == Mastery.CONSOLIDATING.value


def test_re_retention_then_a_new_answer_promotes_again_with_a_new_ready_at(
    settings: Settings,
) -> None:
    _set_state(settings, Mastery.RETAINED)
    record_answer(settings, topic_id=TOPIC, prompt_ref=PROMPT, text="x", scores=READY, now=NOW)
    _timed(settings, NOW + timedelta(days=1), passed=False, attempt="failed-window")
    assert _state(settings)[0] == Mastery.CONSOLIDATING.value

    # A passing timed re-solve, a week after the topic was solved, retains it again.
    _timed(settings, NOW + timedelta(days=2), passed=True, attempt="regained")
    assert _state(settings)[0] == Mastery.RETAINED.value
    later = NOW + timedelta(days=4)
    result = record_answer(
        settings, topic_id=TOPIC, prompt_ref=PROMPT, text="again", scores=READY, now=later
    )

    assert result.after is Mastery.INTERVIEW_READY
    assert _state(settings) == (Mastery.INTERVIEW_READY.value, later)


def test_a_weak_answer_while_consolidating_leaves_the_topic_where_it_is(
    settings: Settings,
) -> None:
    _set_state(settings, Mastery.CONSOLIDATING)

    weak = record_answer(
        settings, topic_id=TOPIC, prompt_ref=PROMPT, text="x", scores=WEAK_TRADE_OFFS, now=NOW
    )

    assert weak.after is Mastery.CONSOLIDATING
    assert _answer_count(settings) == 1
