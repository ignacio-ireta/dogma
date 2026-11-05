"""Timed re-solve windows: one timed, hint-counted attempt at a core problem (ARCHITECTURE §4.4).

A window is a row in ``resolve_window``. It opens when the learner starts a re-solve and closes
on the first recorded submit of that problem, or when it is abandoned. The attempt that decided
it is linked by ``attempt_id``. At most one window is open at a time, enforced by a partial unique
index, so the CHECK constraints and the index mirror the rules in ``dogma.learning.resolves``.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from sqlalchemy import CheckConstraint, ForeignKey, Index, Integer, String, text
from sqlalchemy.orm import Mapped, mapped_column

from dogma.db.base import Base, UtcDateTime


class WindowStatus(StrEnum):
    OPEN = "open"
    PASSED = "passed"  # all tests passed, within the budget, with no hints
    PARTIAL = "partial"  # all tests passed, but late or hinted
    FAILED = "failed"  # anything else, including a rejected or timed-out run
    ABANDONED = "abandoned"  # closed by the learner, or left open for over 12 hours; no penalty


CLOSED_VERDICTS = frozenset({WindowStatus.PASSED, WindowStatus.PARTIAL, WindowStatus.FAILED})


class ResolveWindow(Base):
    """One timed re-solve. ``budget_s`` is the problem's ``meta.minutes`` in seconds."""

    __tablename__ = "resolve_window"
    __table_args__ = (
        CheckConstraint("budget_s > 0", name="budget_positive"),
        CheckConstraint("hints_used >= 0", name="hints_nonnegative"),
        CheckConstraint(
            "status IN ('open', 'passed', 'partial', 'failed', 'abandoned')",
            name="status_valid",
        ),
        CheckConstraint(
            "(status = 'open' AND closed_at IS NULL) "
            "OR (status <> 'open' AND closed_at IS NOT NULL)",
            name="closed_matches_status",
        ),
        CheckConstraint(
            "(status IN ('passed', 'partial', 'failed') AND attempt_id IS NOT NULL) "
            "OR (status IN ('open', 'abandoned') AND attempt_id IS NULL)",
            name="attempt_matches_verdict",
        ),
        Index(
            "ix_resolve_window_one_open",
            "status",
            unique=True,
            sqlite_where=text("status = 'open'"),
        ),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    problem_id: Mapped[str] = mapped_column(String(200), index=True)
    problem_version: Mapped[int] = mapped_column(Integer)
    topic_id: Mapped[str] = mapped_column(String(200), index=True)
    started_at: Mapped[datetime] = mapped_column(UtcDateTime())
    budget_s: Mapped[int] = mapped_column(Integer)
    hints_used: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    status: Mapped[str] = mapped_column(String(16))
    attempt_id: Mapped[str | None] = mapped_column(ForeignKey("attempt.id"))
    closed_at: Mapped[datetime | None] = mapped_column(UtcDateTime())
