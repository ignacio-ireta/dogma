"""Interview answers and their rubric scores: one row per answer (ARCHITECTURE §4.4, §4.9).

``answer`` holds what the learner wrote or said about a prompt, and the scores they gave it. The
rubric id and version are stored with the scores, so a later rubric edit never rewrites them.
``text`` is NULL for an answer given aloud (``--spoken``): the score still counts, the words do
not exist. ``scores`` maps each criterion id to its 1-4 score, as JSON.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from sqlalchemy import JSON, CheckConstraint, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from dogma.db.base import Base, UtcDateTime


class AnswerKind(StrEnum):
    INTERVIEW = "interview"  # a topic's interview question, answered by the learner
    CLOSE = "close"  # the 60-second close of a session (reserved)
    MOCK = "mock"  # a criterion of a mock interview (I-18c)


class Answer(Base):
    """One answer to one prompt, scored against one version of one rubric."""

    __tablename__ = "answer"
    __table_args__ = (
        CheckConstraint("kind IN ('interview', 'close', 'mock')", name="kind_valid"),
        CheckConstraint("rubric_version >= 1", name="rubric_version_valid"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    topic_id: Mapped[str | None] = mapped_column(String(200), index=True)
    kind: Mapped[str] = mapped_column(String(16))
    prompt_ref: Mapped[str] = mapped_column(String(200))
    text: Mapped[str | None] = mapped_column(String(8000))
    rubric_id: Mapped[str] = mapped_column(String(64))
    rubric_version: Mapped[int] = mapped_column(Integer)
    scores: Mapped[dict[str, int]] = mapped_column(JSON)
    notes: Mapped[str | None] = mapped_column(String(1000))
    created_at: Mapped[datetime] = mapped_column(UtcDateTime())
    # Set on mock scores only (I-18c): the mock whose 30 items these are.
    mock_session_id: Mapped[str | None] = mapped_column(ForeignKey("mock_session.id"), index=True)
    # Advisory model scores (I-18d): the model's numbers and quotes, beside the self-scores above.
    # Never read as a score: mastery and the self-score ignore this column.
    model_score: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
