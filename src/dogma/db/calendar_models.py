"""The personal course calendar's log: one row per calendar change (ADR-0019, I-30a).

Append-only. The newest row holds the offset in force, in whole weeks, that every date-dependent
reader adds to the authored core path. A course reset writes a row too, so the log keeps the
history of moves while the progress tables are cleared.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import CheckConstraint, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from dogma.db.base import Base, UtcDateTime


class CalendarWarp(Base):
    """One calendar change. ``week_before``/``week_after`` are the calendar week today, if known."""

    __tablename__ = "calendar_warp"
    __table_args__ = (
        CheckConstraint(
            "kind IN ('sync', 'to_week', 'ahead', 'back', 'original', 'course_reset')",
            name="kind_valid",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime())
    kind: Mapped[str] = mapped_column(String(16))
    offset_weeks: Mapped[int] = mapped_column(Integer)
    previous_offset_weeks: Mapped[int] = mapped_column(Integer)
    week_before: Mapped[int | None] = mapped_column(Integer)
    week_after: Mapped[int | None] = mapped_column(Integer)
