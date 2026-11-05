"""The FSRS boundary: reproducible intervals and explicit UTC timestamps."""

from __future__ import annotations

from datetime import UTC, datetime

from fsrs import Card, Rating, Scheduler

from dogma.core.errors import DogmaError, ErrorCode


def utc_time(value: datetime) -> datetime:
    """Normalize an aware timestamp to UTC; never guess a naive timestamp's zone."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise DogmaError(ErrorCode.REVIEW_TIME_INVALID, "Review timestamps must be timezone-aware.")
    return value.astimezone(UTC)


def schedule(card: Card, rating: int, *, now: datetime) -> Card:
    """Compute the next FSRS card with 90% retention and no random interval fuzzing."""
    scheduler = Scheduler(desired_retention=0.90, enable_fuzzing=False)
    updated, _ = scheduler.review_card(card, Rating(rating), review_datetime=utc_time(now))
    return updated
