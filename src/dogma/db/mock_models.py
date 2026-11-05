"""Mock interview sessions: one row per mock, from start to score or abandon (ARCHITECTURE §4.4).

A mock picks a case, a passed core coding problem and a theory question when it starts. Its
scores live in an ``answer`` row (kind ``mock``) that points back here through
``answer.mock_session_id``. At most one mock is open at a time, enforced by a partial unique index.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from sqlalchemy import JSON, CheckConstraint, Index, Integer, String, text
from sqlalchemy.orm import Mapped, mapped_column

from dogma.db.base import Base, UtcDateTime


class MockStatus(StrEnum):
    OPEN = "open"
    SCORED = "scored"  # all 30 items scored and stored
    ABANDONED = "abandoned"  # closed by the learner without scores


class MockSession(Base):
    """One mock. ``part_minutes`` is the timing each rubric part was planned for."""

    __tablename__ = "mock_session"
    __table_args__ = (
        CheckConstraint("status IN ('open', 'scored', 'abandoned')", name="status_valid"),
        CheckConstraint(
            "(status = 'open' AND ended_at IS NULL) OR (status <> 'open' AND ended_at IS NOT NULL)",
            name="ended_matches_status",
        ),
        CheckConstraint(
            "theory_question IS NULL OR theory_question >= 1", name="theory_question_valid"
        ),
        Index(
            "ix_mock_session_one_open",
            "status",
            unique=True,
            sqlite_where=text("status = 'open'"),
        ),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    started_at: Mapped[datetime] = mapped_column(UtcDateTime())
    ended_at: Mapped[datetime | None] = mapped_column(UtcDateTime())
    case_id: Mapped[str | None] = mapped_column(String(200))
    coding_problem_id: Mapped[str | None] = mapped_column(String(200))
    theory_topic_id: Mapped[str | None] = mapped_column(String(200))
    theory_question: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16))
    part_minutes: Mapped[dict[str, str]] = mapped_column(JSON)
