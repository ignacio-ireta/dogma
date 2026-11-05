"""Add interview answers and their rubric scores (ARCHITECTURE §4.4, §4.9, I-18a).

Forward-only. Existing attempt, topic_state, card, card_review, study_session, session_block and
resolve_window rows are not touched. ``scores`` is JSON text; timestamps are UTC ISO-8601 text.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e5f6a7b8c9d0"
down_revision: str | None = "d4e5f6a7b8c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the answer table and its topic index. Nothing that exists changes."""
    op.create_table(
        "answer",
        sa.Column("id", sa.String(32), nullable=False),
        sa.Column("topic_id", sa.String(200), nullable=True),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("prompt_ref", sa.String(200), nullable=False),
        sa.Column("text", sa.String(8000), nullable=True),
        sa.Column("rubric_id", sa.String(64), nullable=False),
        sa.Column("rubric_version", sa.Integer(), nullable=False),
        sa.Column("scores", sa.JSON(), nullable=False),
        sa.Column("notes", sa.String(1000), nullable=True),
        sa.Column("created_at", sa.String(32), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_answer"),
        sa.CheckConstraint("kind IN ('interview', 'close', 'mock')", name="kind_valid"),
        sa.CheckConstraint("rubric_version >= 1", name="rubric_version_valid"),
    )
    op.create_index("ix_answer_topic_id", "answer", ["topic_id"], unique=False)


def downgrade() -> None:
    """Refuse destructive migrations consistently with the existing schema."""
    raise RuntimeError("dogma migrations are forward-only; restore from var/backups instead")
