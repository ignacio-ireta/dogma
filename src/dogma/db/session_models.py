"""Study sessions: one row per planned session, its blocks, and the note it closed with (I-10).

A session belongs to a local date and a slot in that day (1 to ``sessions_per_day``). The plan is
stored when it is made, so a resumed session shows exactly what was promised, even if content has
changed since. Closing fills in the transition note; a closed row never changes again.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import CheckConstraint, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from dogma.db.base import Base, UtcDateTime


class StudySession(Base):
    """One planned session. ``plan_date`` is the learner's local date, ``YYYY-MM-DD``."""

    __tablename__ = "study_session"
    __table_args__ = (
        UniqueConstraint("plan_date", "slot"),
        CheckConstraint("slot >= 1", name="slot_valid"),
        CheckConstraint(
            "kind IN ('learn', 'consolidate', 'minimum', 'interview')", name="kind_valid"
        ),
        CheckConstraint("energy IN ('normal', 'low')", name="energy_valid"),
        CheckConstraint("status IN ('open', 'closed')", name="status_valid"),
        CheckConstraint(
            "status = 'open' OR (closed_at IS NOT NULL AND stopped_where IS NOT NULL "
            "AND next_step IS NOT NULL)",
            name="closed_has_note",
        ),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    plan_date: Mapped[str] = mapped_column(String(10), index=True)
    slot: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(16))
    energy: Mapped[str] = mapped_column(String(8))
    status: Mapped[str] = mapped_column(String(8))
    topic_id: Mapped[str | None] = mapped_column(String(200))
    must_do: Mapped[str | None] = mapped_column(String(200))
    first_action: Mapped[str] = mapped_column(String(500))
    plan_notes: Mapped[str | None] = mapped_column(String(1000))
    created_at: Mapped[datetime] = mapped_column(UtcDateTime())
    closed_at: Mapped[datetime | None] = mapped_column(UtcDateTime())
    stopped_where: Mapped[str | None] = mapped_column(String(500))
    next_step: Mapped[str | None] = mapped_column(String(500))
    blocker: Mapped[str | None] = mapped_column(String(500))
    self_answer: Mapped[str | None] = mapped_column(String(2000))
    timed_problem_id: Mapped[str | None] = mapped_column(String(200))


class SessionBlock(Base):
    """One timed block of a planned session, in the order the learner works through them."""

    __tablename__ = "session_block"
    __table_args__ = (
        UniqueConstraint("session_id", "position"),
        CheckConstraint("minutes > 0", name="minutes_positive"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("study_session.id"), index=True)
    position: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(16))
    minutes: Mapped[int] = mapped_column(Integer)
    detail: Mapped[str] = mapped_column(String(500))
