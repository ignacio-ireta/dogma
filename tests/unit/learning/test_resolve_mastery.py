"""Mastery through timed re-solves and card reviews together.

A topic solved before its timed re-solves can be retained by one: the card review that moves it to
consolidating re-evaluates it. A card lapse on a retained topic stands; it is not re-evaluated away.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from tests.unit.db.support import PROBLEM, make_report
from tests.unit.learning.conftest import TOPIC

from dogma.core.config import Settings
from dogma.db.errors import StorageErrorCode
from dogma.db.models import AttemptMode, Mastery, TopicState
from dogma.db.resolve_store import start_window
from dogma.db.session import open_session
from dogma.db.store import record_attempt
from dogma.grading.models import RunStatus
from dogma.learning.cards import due_cards, sync_cards
from dogma.learning.review import rate_card

FIRST_PASS = datetime(2026, 10, 14, 9, tzinfo=UTC)
REWARDED_AT = FIRST_PASS + timedelta(days=8)  # a timed pass, seven days or more after solving
NOW = FIRST_PASS + timedelta(days=9)  # three or more days after solving, so a good rating counts


def _record_pass(settings: Settings, when: datetime, attempt: str) -> None:
    record_attempt(
        settings,
        make_report(RunStatus.PASSED, public=(4, 4), hidden=(11, 11), attempt_id=attempt),
        mode=AttemptMode.SUBMIT,
        core=True,
        started_at=when - timedelta(seconds=1),
        submitted_at=when,
    )


def _timed_pass(settings: Settings, when: datetime) -> None:
    """A window opened at the timed re-solve's start, closed by a passing submit inside it."""
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        start_window(
            session,
            problem_id=PROBLEM,
            problem_version=2,
            budget_s=1200,
            now=when - timedelta(minutes=5),
        )
    record_attempt(
        settings,
        make_report(RunStatus.PASSED, public=(4, 4), hidden=(11, 11), attempt_id="timed-pass"),
        mode=AttemptMode.SUBMIT,
        core=True,
        started_at=when - timedelta(minutes=6),
        submitted_at=when,
    )


def _state(settings: Settings) -> TopicState:
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        state = session.get(TopicState, TOPIC)
        assert state is not None
        return TopicState(
            topic_id=state.topic_id,
            mastery=state.mastery,
            solved_at=state.solved_at,
            retained_at=state.retained_at,
        )


def _rate(settings: Settings, rating: int, when: datetime) -> Mastery:
    """Rate the first card still due. A card rated good is scheduled later, so the next is used."""
    card = due_cards(settings, now=when)[0]
    result = rate_card(
        settings,
        card.card_id,
        rating,
        now=when,
        expected_due=card.due,
        expected_content_hash=card.content_hash,
    )
    return result.mastery


def test_a_good_card_review_retains_a_topic_whose_timed_pass_came_first(
    settings: Settings,
) -> None:
    _record_pass(settings, FIRST_PASS, "first")
    _timed_pass(settings, REWARDED_AT)
    assert _state(settings).mastery == Mastery.SOLVED.value
    sync_cards(settings, now=NOW)

    mastery = _rate(settings, 3, NOW)

    assert mastery is Mastery.RETAINED
    state = _state(settings)
    assert state.mastery == Mastery.RETAINED.value
    assert state.retained_at == REWARDED_AT


def test_a_card_lapse_on_a_retained_topic_is_not_undone_by_re_evaluation(
    settings: Settings,
) -> None:
    _record_pass(settings, FIRST_PASS, "first")
    _timed_pass(settings, REWARDED_AT)
    sync_cards(settings, now=NOW)
    assert _rate(settings, 3, NOW) is Mastery.RETAINED

    lapsed = _rate(settings, 1, NOW + timedelta(seconds=1))

    assert lapsed is Mastery.CONSOLIDATING
    assert _state(settings).mastery == Mastery.CONSOLIDATING.value


def test_a_timed_pass_on_a_consolidating_topic_retains_it_at_once(settings: Settings) -> None:
    _record_pass(settings, FIRST_PASS, "first")
    sync_cards(settings, now=NOW)
    _rate(settings, 3, NOW)
    assert _state(settings).mastery == Mastery.CONSOLIDATING.value

    _timed_pass(settings, REWARDED_AT + timedelta(days=2))

    assert _state(settings).mastery == Mastery.RETAINED.value
