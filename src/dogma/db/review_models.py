"""Persistent FSRS cards and append-only review events."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, CheckConstraint, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from dogma.db.base import Base, UtcDateTime


class Card(Base):
    """One stable authored card, including its current FSRS schedule."""

    __tablename__ = "card"
    __table_args__ = (CheckConstraint("state IN (1, 2, 3)", name="state_valid"),)

    id: Mapped[str] = mapped_column(String(200), primary_key=True)
    topic_id: Mapped[str] = mapped_column(String(200), index=True)
    content_hash: Mapped[str] = mapped_column(String(64))
    question: Mapped[str] = mapped_column(String)
    answer: Mapped[str] = mapped_column(String)
    active: Mapped[bool] = mapped_column(Boolean)
    state: Mapped[int] = mapped_column(Integer)
    step: Mapped[int | None] = mapped_column(Integer)
    stability: Mapped[float | None] = mapped_column(Float)
    difficulty: Mapped[float | None] = mapped_column(Float)
    due: Mapped[datetime] = mapped_column(UtcDateTime(), index=True)
    last_review: Mapped[datetime | None] = mapped_column(UtcDateTime())


class CardReview(Base):
    """One rating with its before/after schedule; content snapshots preserve historical meaning."""

    __tablename__ = "card_review"
    __table_args__ = (CheckConstraint("rating IN (1, 2, 3, 4)", name="rating_valid"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    card_id: Mapped[str] = mapped_column(ForeignKey("card.id"), index=True)
    rating: Mapped[int] = mapped_column(Integer)
    reviewed_at: Mapped[datetime] = mapped_column(UtcDateTime())
    due_before: Mapped[datetime] = mapped_column(UtcDateTime())
    due_after: Mapped[datetime] = mapped_column(UtcDateTime())
    content_hash: Mapped[str] = mapped_column(String(64))
    question: Mapped[str] = mapped_column(String)
    answer: Mapped[str] = mapped_column(String)
