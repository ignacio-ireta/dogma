"""Synchronize authored cards for studied topics and read a bounded due queue."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from dogma.content.loader import load_topic
from dogma.content.schemas import Card as AuthoredCard
from dogma.core.config import Settings
from dogma.core.errors import ContentError, DogmaError, ErrorCode
from dogma.db.errors import StorageErrorCode
from dogma.db.models import TopicState
from dogma.db.review_models import Card
from dogma.db.session import database_path, open_session
from dogma.learning.scheduler import utc_time

_CLOZE = re.compile(r"\{\{c\d+::(.*?)\}\}", re.DOTALL)


@dataclass(frozen=True)
class ReviewCard:
    """A detached retrieval snapshot; ratings bind its UTC-aware due time and content hash."""

    card_id: str
    topic_id: str
    question: str
    answer: str
    due: datetime
    content_hash: str


def sync_cards(settings: Settings, *, now: datetime) -> int:
    """Enroll studied, ready topics, retain edited schedules, and deactivate absent content.

    Return the number of newly enrolled cards. Existing ids are never deleted or reset.
    """
    now = utc_time(now)
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        return sync_cards_in_session(settings, session, now=now)


def sync_cards_in_session(settings: Settings, session: Session, *, now: datetime) -> int:
    """Synchronize cards inside a caller's transaction, including pending introductions."""
    now = utc_time(now)
    session.flush()
    existing = {card.id: card for card in session.scalars(select(Card))}
    active: set[str] = set()
    created = 0
    for state in session.scalars(select(TopicState).order_by(TopicState.topic_id)):
        for authored in _topic_cards(settings, state.topic_id):
            if authored.id in active:
                raise ContentError(ErrorCode.CONTENT_SCHEMA_INVALID, "Duplicate review card id.")
            active.add(authored.id)
            stored = existing.get(authored.id)
            if stored is None:
                stored = Card(id=authored.id, topic_id=state.topic_id, state=1, step=0, due=now)
                session.add(stored)
                created += 1
            stored.content_hash = hashlib.sha256(authored.model_dump_json().encode()).hexdigest()
            stored.question, stored.answer = _render(authored)
            stored.active = True
    for card in existing.values():
        if card.id not in active:
            card.active = False
    return created


def due_cards(settings: Settings, *, now: datetime, limit: int = 20) -> tuple[ReviewCard, ...]:
    """Return active due cards in stable due/id order without creating a database on first read."""
    now = utc_time(now)
    if type(limit) is not int or limit < 1:
        raise DogmaError(
            ErrorCode.REVIEW_LIMIT_INVALID, "The review limit must be a positive integer."
        )
    if not database_path(settings).is_file():
        return ()
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        rows = session.scalars(
            select(Card)
            .where(Card.active.is_(True), Card.due <= now)
            .order_by(Card.due, Card.id)
            .limit(limit)
        )
        return tuple(
            ReviewCard(row.id, row.topic_id, row.question, row.answer, row.due, row.content_hash)
            for row in rows
        )


def _topic_cards(settings: Settings, topic_id: str) -> tuple[AuthoredCard, ...]:
    path = settings.content_dir / topic_id
    if not (path / "topic.yaml").is_file():
        return ()
    topic = load_topic(path)
    if topic.meta.status not in {"verified", "reviewed"}:
        return ()
    for card in topic.cards or ():
        suffix = card.id.removeprefix(f"{topic_id}#")
        if not card.id.startswith(f"{topic_id}#") or not suffix.isdigit():
            raise ContentError(
                ErrorCode.CONTENT_SCHEMA_INVALID, f"{topic_id}: invalid stable card id {card.id}."
            )
    return topic.cards or ()


def _render(card: AuthoredCard) -> tuple[str, str]:
    if card.type == "qa":
        return card.front, card.back
    return _CLOZE.sub("[...]", card.text), _CLOZE.sub(r"\1", card.text)
