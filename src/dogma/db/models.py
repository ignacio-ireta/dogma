"""ORM models for the first two tables (ARCHITECTURE §4.4, §4.9).

``attempt`` and ``topic_state`` only. The other §4.9 tables arrive with the increments that use
them, each with its own migration.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from sqlalchemy import CheckConstraint, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from dogma.db.base import Base, UtcDateTime


class AttemptMode(StrEnum):
    RUN = "run"  # public tests only
    SUBMIT = "submit"  # public and hidden tests


class Mastery(StrEnum):
    """Topic mastery in the order a topic moves through it (ARCHITECTURE §4.4).

    A topic with no ``topic_state`` row is ``unseen``. ``unseen`` itself is never stored.
    """

    UNSEEN = "unseen"
    INTRODUCED = "introduced"
    SOLVED = "solved"
    CONSOLIDATING = "consolidating"
    RETAINED = "retained"
    INTERVIEW_READY = "interview-ready"


_MASTERY_ORDER = tuple(Mastery)


def mastery_rank(state: str) -> int:
    """Position on the mastery path. A state never moves to a lower rank."""
    return _MASTERY_ORDER.index(Mastery(state))


class Attempt(Base):
    """One graded run of one problem. ``problem_version`` says which edition of it was graded."""

    __tablename__ = "attempt"
    __table_args__ = (CheckConstraint("mode IN ('run', 'submit')", name="mode_valid"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    problem_id: Mapped[str] = mapped_column(String(200), index=True)
    problem_version: Mapped[int] = mapped_column(Integer)
    mode: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16))
    public_passed: Mapped[int] = mapped_column(Integer)
    public_total: Mapped[int] = mapped_column(Integer)
    hidden_passed: Mapped[int] = mapped_column(Integer)
    hidden_total: Mapped[int] = mapped_column(Integer)
    hints_used: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    run_dir: Mapped[str | None] = mapped_column(String(255))
    started_at: Mapped[datetime] = mapped_column(UtcDateTime())
    submitted_at: Mapped[datetime] = mapped_column(UtcDateTime())
    duration_ms: Mapped[int] = mapped_column(Integer)


class TopicState(Base):
    """A topic's furthest mastery state so far, plus when it got there."""

    __tablename__ = "topic_state"
    __table_args__ = (
        CheckConstraint(
            "mastery IN ('introduced', 'solved', 'consolidating', 'retained', 'interview-ready')",
            name="mastery_valid",
        ),
    )

    topic_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    mastery: Mapped[str] = mapped_column(String(32))
    introduced_at: Mapped[datetime | None] = mapped_column(UtcDateTime())
    solved_at: Mapped[datetime | None] = mapped_column(UtcDateTime())
    retained_at: Mapped[datetime | None] = mapped_column(UtcDateTime())
    ready_at: Mapped[datetime | None] = mapped_column(UtcDateTime())
