"""Atomic review ratings and the limited mastery changes earned by retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import uuid4

from fsrs import Card as FsrsCard
from fsrs import State

from dogma.core.config import Settings
from dogma.core.errors import DogmaError, ErrorCode
from dogma.db.errors import StorageErrorCode
from dogma.db.models import Mastery, TopicState, mastery_rank
from dogma.db.resolve_store import apply_mastery
from dogma.db.review_models import Card, CardReview
from dogma.db.session import open_session
from dogma.learning.scheduler import schedule, utc_time


@dataclass(frozen=True)
class ReviewResult:
    """A persisted rating's new due time and topic mastery, detached from the DB session."""

    card_id: str
    topic_id: str
    due: datetime
    rating: int
    mastery: Mastery


def rate_card(
    settings: Settings,
    card_id: str,
    rating: int,
    *,
    now: datetime,
    expected_due: datetime,
    expected_content_hash: str,
) -> ReviewResult:
    """Persist one rating atomically only if its displayed due time and content still match."""
    now, expected_due = utc_time(now), utc_time(expected_due)
    if type(rating) is not int or rating not in {1, 2, 3, 4}:
        raise DogmaError(
            ErrorCode.REVIEW_RATING_INVALID, "Choose 1 Again, 2 Hard, 3 Good or 4 Easy."
        )
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        card = session.get(Card, card_id)
        _validate_card(card, expected_due, expected_content_hash, now)
        if card is None:  # _validate_card rejects this; narrow the ORM's optional return type.
            raise DogmaError(ErrorCode.REVIEW_NOT_FOUND, "This card is no longer available.")
        updated = schedule(_fsrs_card(card), rating, now=now)
        session.add(
            CardReview(
                id=uuid4().hex,
                card_id=card.id,
                rating=rating,
                reviewed_at=now,
                due_before=card.due,
                due_after=updated.due,
                content_hash=card.content_hash,
                question=card.question,
                answer=card.answer,
            )
        )
        card.state, card.step = int(updated.state), updated.step
        card.stability, card.difficulty = updated.stability, updated.difficulty
        card.due, card.last_review = updated.due, updated.last_review
        topic = session.get(TopicState, card.topic_id)
        previous = None if topic is None else Mastery(topic.mastery)
        mastery = _review_mastery(topic, rating, now)
        if topic is not None and previous is Mastery.SOLVED and mastery is Mastery.CONSOLIDATING:
            # A topic solved before its timed re-solves can be retained by one now: the card
            # review is what moves it to consolidating, so re-evaluate it here. A lapse on an
            # already retained topic is not re-evaluated, so the lapse stands.
            apply_mastery(session, topic)
            mastery = Mastery(topic.mastery)
        return ReviewResult(card.id, card.topic_id, card.due, rating, mastery)


def _validate_card(
    card: Card | None, expected_due: datetime, expected_content_hash: str, now: datetime
) -> None:
    if card is None or not card.active:
        raise DogmaError(ErrorCode.REVIEW_NOT_FOUND, "This card is no longer available.")
    if card.due != expected_due or card.content_hash != expected_content_hash:
        raise DogmaError(
            ErrorCode.REVIEW_STALE, "This card's schedule or content changed; refresh the queue."
        )
    if card.due > now:
        raise DogmaError(ErrorCode.REVIEW_NOT_DUE, "This card is not due yet.")


def _fsrs_card(card: Card) -> FsrsCard:
    # Authored ids are strings. FSRS's numeric id is unused by interval calculations;
    # passing one explicitly also avoids its default wall clock and sleep.
    return FsrsCard(
        card_id=0,
        state=State(card.state),
        step=card.step,
        stability=card.stability,
        difficulty=card.difficulty,
        due=card.due,
        last_review=card.last_review,
    )


def _review_mastery(topic: TopicState | None, rating: int, now: datetime) -> Mastery:
    if topic is None:
        return Mastery.UNSEEN
    if (rating == 1 and mastery_rank(topic.mastery) > mastery_rank(Mastery.CONSOLIDATING)) or (
        rating >= 3
        and topic.mastery == Mastery.SOLVED
        and topic.solved_at is not None
        and now - topic.solved_at >= timedelta(days=3)
    ):
        topic.mastery = Mastery.CONSOLIDATING
    return Mastery(topic.mastery)
