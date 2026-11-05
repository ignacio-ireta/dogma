"""Persistent reviews: enrollment, real scheduling, stale-write protection and milestones."""

from __future__ import annotations

from datetime import timedelta

import pytest
import yaml
from sqlalchemy import func, select
from tests.unit.learning.conftest import NOW, TOPIC

from dogma.core.config import Settings
from dogma.core.errors import DogmaError
from dogma.db.engine import make_engine
from dogma.db.errors import StorageError, StorageErrorCode
from dogma.db.models import Mastery, TopicState
from dogma.db.review_models import Card, CardReview
from dogma.db.session import database_path, open_session
from dogma.learning.cards import due_cards, sync_cards
from dogma.learning.review import rate_card


def _counts(settings: Settings) -> tuple[int, int]:
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        return (
            session.scalar(select(func.count()).select_from(Card)),
            session.scalar(select(func.count()).select_from(CardReview)),
        )


def test_unseen_topics_do_not_create_a_review_backlog(settings: Settings) -> None:
    assert sync_cards(settings, now=NOW) == 0
    assert due_cards(settings, now=NOW) == ()


def test_solved_topics_enroll_once_with_explicit_due_time(solved: Settings) -> None:
    assert sync_cards(solved, now=NOW) == 6
    assert sync_cards(solved, now=NOW + timedelta(days=5)) == 0
    cards = due_cards(solved, now=NOW, limit=2)
    assert [card.card_id for card in cards] == [f"{TOPIC}#1", f"{TOPIC}#2"]
    assert all(card.due == NOW and card.topic_id == TOPIC for card in cards)
    assert len(due_cards(solved, now=NOW)) == 6
    assert due_cards(solved, now=NOW - timedelta(seconds=1)) == ()


def test_introduced_topics_enroll_without_claiming_a_solve(settings: Settings) -> None:
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        session.add(TopicState(topic_id=TOPIC, mastery=Mastery.INTRODUCED, introduced_at=NOW))
    assert sync_cards(settings, now=NOW) == 6
    card = due_cards(settings, now=NOW)[0]
    result = rate_card(
        settings,
        card.card_id,
        4,
        now=NOW,
        expected_due=card.due,
        expected_content_hash=card.content_hash,
    )
    assert result.mastery == Mastery.INTRODUCED


def test_questions_hide_clozes_until_reveal(solved: Settings) -> None:
    sync_cards(solved, now=NOW)
    qa, cloze, *_ = due_cards(solved, now=NOW)
    assert qa.question == "What is the orthogonal projection of a onto b?"
    assert "scalar multiple" in qa.answer
    assert cloze.question == "a·b = ‖a‖ ‖b‖ [...]"
    assert cloze.answer == "a·b = ‖a‖ ‖b‖ cos θ"


@pytest.mark.parametrize("rating,delay", [(1, 60), (2, 330), (3, 600), (4, 8 * 86400)])
def test_each_rating_persists_its_deterministic_fsrs_schedule(
    solved: Settings, rating: int, delay: int
) -> None:
    sync_cards(solved, now=NOW)
    card = due_cards(solved, now=NOW)[0]
    result = rate_card(
        solved,
        card.card_id,
        rating,
        now=NOW,
        expected_due=card.due,
        expected_content_hash=card.content_hash,
    )
    assert result.due == NOW + timedelta(seconds=delay)
    with open_session(solved, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        saved = session.get(Card, card.card_id)
        event = session.scalars(select(CardReview)).one()
        assert saved.due == result.due
        assert saved.last_review == NOW
        assert saved.stability > 0 and saved.difficulty > 0
        assert event.rating == rating and event.reviewed_at == NOW
        assert event.due_before == NOW and event.due_after == result.due
    assert card.card_id not in {item.card_id for item in due_cards(solved, now=NOW)}


def test_good_review_after_three_days_earns_consolidating(old_solve: Settings) -> None:
    sync_cards(old_solve, now=NOW)
    card = due_cards(old_solve, now=NOW)[0]
    assert rate_card(
        old_solve,
        card.card_id,
        3,
        now=NOW,
        expected_due=card.due,
        expected_content_hash=card.content_hash,
    ).mastery == (Mastery.CONSOLIDATING)


@pytest.mark.parametrize("rating", [1, 2])
def test_again_and_hard_cannot_earn_consolidating(old_solve: Settings, rating: int) -> None:
    sync_cards(old_solve, now=NOW)
    card = due_cards(old_solve, now=NOW)[0]
    assert rate_card(
        old_solve,
        card.card_id,
        rating,
        now=NOW,
        expected_due=card.due,
        expected_content_hash=card.content_hash,
    ).mastery == (Mastery.SOLVED)


def test_recent_solve_does_not_earn_consolidating(solved: Settings) -> None:
    sync_cards(solved, now=NOW)
    card = due_cards(solved, now=NOW)[0]
    result = rate_card(
        solved,
        card.card_id,
        3,
        now=NOW,
        expected_due=card.due,
        expected_content_hash=card.content_hash,
    )
    assert result.mastery == Mastery.SOLVED


def test_lapse_drops_readiness_but_preserves_milestones(solved: Settings) -> None:
    with open_session(solved, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        state = session.get(TopicState, TOPIC)
        state.mastery = Mastery.INTERVIEW_READY
        state.retained_at = state.ready_at = NOW - timedelta(days=1)
    sync_cards(solved, now=NOW)
    card = due_cards(solved, now=NOW)[0]
    rate_card(
        solved,
        card.card_id,
        4,
        now=NOW,
        expected_due=card.due,
        expected_content_hash=card.content_hash,
    )
    later = NOW + timedelta(days=8)
    due = next(item for item in due_cards(solved, now=later) if item.card_id == card.card_id)
    result = rate_card(
        solved,
        due.card_id,
        1,
        now=later,
        expected_due=due.due,
        expected_content_hash=due.content_hash,
    )
    assert result.mastery == Mastery.CONSOLIDATING
    with open_session(solved, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        state = session.get(TopicState, TOPIC)
        assert state.solved_at == NOW
        assert state.retained_at == state.ready_at == NOW - timedelta(days=1)
        assert session.get(Card, f"{TOPIC}#1").state == 3  # FSRS relearning


def test_content_edits_keep_schedule_and_history(solved: Settings) -> None:
    sync_cards(solved, now=NOW)
    original = due_cards(solved, now=NOW)[0]
    result = rate_card(
        solved,
        original.card_id,
        3,
        now=NOW,
        expected_due=original.due,
        expected_content_hash=original.content_hash,
    )
    path = solved.content_dir / TOPIC / "cards.yaml"
    path.write_text(path.read_text().replace("orthogonal projection", "projection formula"))
    assert sync_cards(solved, now=NOW + timedelta(seconds=1)) == 0
    card = next(c for c in due_cards(solved, now=result.due) if c.card_id == result.card_id)
    assert "projection formula" in card.question
    assert card.due == result.due
    assert _counts(solved) == (6, 1)


@pytest.mark.parametrize("changed_field", ["front", "back"])
def test_edited_queue_snapshot_is_stale_without_mutating_reviews_or_mastery(
    old_solve: Settings, changed_field: str
) -> None:
    sync_cards(old_solve, now=NOW)
    original = due_cards(old_solve, now=NOW)[0]
    first = rate_card(
        old_solve,
        original.card_id,
        2,
        now=NOW,
        expected_due=original.due,
        expected_content_hash=original.content_hash,
    )
    queued = next(
        card for card in due_cards(old_solve, now=first.due) if card.card_id == original.card_id
    )
    path = old_solve.content_dir / TOPIC / "cards.yaml"
    data = yaml.safe_load(path.read_text())
    data["cards"][0][changed_field] = "Changed card content"
    path.write_text(yaml.safe_dump(data))
    sync_cards(old_solve, now=first.due)
    fresh = next(
        card for card in due_cards(old_solve, now=first.due) if card.card_id == original.card_id
    )
    assert fresh.due == queued.due
    assert fresh.content_hash != queued.content_hash
    card_snapshot = select(Card.__table__).where(Card.id == queued.card_id)
    history_snapshot = select(CardReview.__table__).order_by(CardReview.id)
    with open_session(old_solve, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        before_card = dict(session.execute(card_snapshot).mappings().one())
        before_history = tuple(dict(row) for row in session.execute(history_snapshot).mappings())
        before_mastery = dict(session.execute(select(TopicState.__table__)).mappings().one())

    with pytest.raises(DogmaError, match="REVIEW_STALE"):
        rate_card(
            old_solve,
            queued.card_id,
            3,
            now=first.due,
            expected_due=queued.due,
            expected_content_hash=queued.content_hash,
        )

    with open_session(old_solve, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        assert dict(session.execute(card_snapshot).mappings().one()) == before_card
        assert (
            tuple(dict(row) for row in session.execute(history_snapshot).mappings())
            == before_history
        )
        assert (
            dict(session.execute(select(TopicState.__table__)).mappings().one()) == before_mastery
        )

    accepted = rate_card(
        old_solve,
        fresh.card_id,
        3,
        now=first.due,
        expected_due=fresh.due,
        expected_content_hash=fresh.content_hash,
    )
    assert accepted.mastery == Mastery.CONSOLIDATING
    with open_session(old_solve, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        events = session.scalars(select(CardReview).order_by(CardReview.reviewed_at)).all()
        assert len(events) == 2
        assert events[0].question == original.question
        assert events[0].answer == original.answer
        assert events[1].question == fresh.question
        assert events[1].answer == fresh.answer
        assert events[0].content_hash == original.content_hash
        assert events[1].content_hash == fresh.content_hash


@pytest.mark.parametrize("change", ["retire", "remove-card", "remove-topic", "draft"])
def test_missing_or_unready_content_is_inactive_without_deleting_history(
    solved: Settings, change: str
) -> None:
    sync_cards(solved, now=NOW)
    card = due_cards(solved, now=NOW)[0]
    rate_card(
        solved,
        card.card_id,
        3,
        now=NOW,
        expected_due=card.due,
        expected_content_hash=card.content_hash,
    )
    if change == "remove-card":
        path = solved.content_dir / TOPIC / "cards.yaml"
        data = yaml.safe_load(path.read_text())
        data["cards"] = data["cards"][1:]
        path.write_text(yaml.safe_dump(data))
    else:
        path = solved.content_dir / TOPIC / "topic.yaml"
        if change == "remove-topic":
            path.unlink()
        else:
            data = yaml.safe_load(path.read_text())
            data["status"] = "retired" if change == "retire" else "draft"
            path.write_text(yaml.safe_dump(data))
    sync_cards(solved, now=NOW)
    assert f"{TOPIC}#1" not in {c.card_id for c in due_cards(solved, now=NOW + timedelta(days=9))}
    assert _counts(solved) == (6, 1)


def test_stale_or_repeated_submissions_never_duplicate_events(solved: Settings) -> None:
    sync_cards(solved, now=NOW)
    card = due_cards(solved, now=NOW)[0]
    rate_card(
        solved,
        card.card_id,
        3,
        now=NOW,
        expected_due=card.due,
        expected_content_hash=card.content_hash,
    )
    with pytest.raises(DogmaError, match="REVIEW_STALE"):
        rate_card(
            solved,
            card.card_id,
            3,
            now=NOW + timedelta(days=1),
            expected_due=card.due,
            expected_content_hash=card.content_hash,
        )
    assert _counts(solved) == (6, 1)


def test_early_review_is_rejected(solved: Settings) -> None:
    sync_cards(solved, now=NOW)
    card = due_cards(solved, now=NOW)[0]
    with pytest.raises(DogmaError, match="REVIEW_NOT_DUE"):
        rate_card(
            solved,
            card.card_id,
            3,
            now=NOW - timedelta(seconds=1),
            expected_due=card.due,
            expected_content_hash=card.content_hash,
        )
    assert _counts(solved) == (6, 0)


@pytest.mark.parametrize("rating", [0, 5, True, 1.0])
def test_invalid_ratings_never_change_state(solved: Settings, rating: int) -> None:
    sync_cards(solved, now=NOW)
    card = due_cards(solved, now=NOW)[0]
    with pytest.raises(DogmaError, match="REVIEW_RATING_INVALID"):
        rate_card(
            solved,
            card.card_id,
            rating,
            now=NOW,
            expected_due=card.due,
            expected_content_hash=card.content_hash,
        )
    assert _counts(solved) == (6, 0)


def test_unknown_card_is_a_safe_domain_error(solved: Settings) -> None:
    with pytest.raises(DogmaError, match="REVIEW_NOT_FOUND"):
        rate_card(
            solved, "missing#1", 3, now=NOW, expected_due=NOW, expected_content_hash="missing"
        )


@pytest.mark.parametrize("operation", ["sync", "due", "rate", "expected_due"])
def test_naive_timestamps_are_rejected_before_any_mutation(
    solved: Settings, operation: str
) -> None:
    sync_cards(solved, now=NOW)
    card = due_cards(solved, now=NOW)[0]
    naive = NOW.replace(tzinfo=None)
    with pytest.raises(DogmaError, match="REVIEW_TIME_INVALID"):
        if operation == "sync":
            sync_cards(solved, now=naive)
        elif operation == "due":
            due_cards(solved, now=naive)
        else:
            rate_card(
                solved,
                card.card_id,
                3,
                now=naive if operation == "rate" else NOW,
                expected_due=naive if operation == "expected_due" else card.due,
                expected_content_hash=card.content_hash,
            )
    assert _counts(solved) == (6, 0)


@pytest.mark.parametrize("limit", [0, -1])
def test_due_queue_requires_a_positive_bound(solved: Settings, limit: int) -> None:
    with pytest.raises(DogmaError, match="REVIEW_LIMIT_INVALID"):
        due_cards(solved, now=NOW, limit=limit)


def test_failed_event_insert_rolls_back_schedule_and_mastery(old_solve: Settings) -> None:
    sync_cards(old_solve, now=NOW)
    card = due_cards(old_solve, now=NOW)[0]
    engine = make_engine(database_path(old_solve))
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "CREATE TRIGGER reject_review BEFORE INSERT ON card_review "
                "BEGIN SELECT RAISE(ABORT, 'injected write failure'); END"
            )
    finally:
        engine.dispose()
    with pytest.raises(StorageError):
        rate_card(
            old_solve,
            card.card_id,
            3,
            now=NOW,
            expected_due=card.due,
            expected_content_hash=card.content_hash,
        )
    assert _counts(old_solve) == (6, 0)
    assert due_cards(old_solve, now=NOW)[0].due == NOW
    with open_session(old_solve, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        assert session.get(TopicState, TOPIC).mastery == Mastery.SOLVED
