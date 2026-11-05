"""Add planned study sessions and their blocks (ARCHITECTURE §4.4, I-10).

Forward-only. Existing attempt, topic_state, card and card_review rows are not touched. Timestamps
are UTC ISO-8601 text, matching dogma.db.base.UtcDateTime.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c3d4e5f6a7b8"
down_revision: str | None = "b2c3d4e5f6a7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the session tables. Nothing that already exists changes."""
    op.create_table(
        "study_session",
        sa.Column("id", sa.String(32), nullable=False),
        sa.Column("plan_date", sa.String(10), nullable=False),
        sa.Column("slot", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("energy", sa.String(8), nullable=False),
        sa.Column("status", sa.String(8), nullable=False),
        sa.Column("topic_id", sa.String(200), nullable=True),
        sa.Column("must_do", sa.String(200), nullable=True),
        sa.Column("first_action", sa.String(500), nullable=False),
        sa.Column("plan_notes", sa.String(1000), nullable=True),
        sa.Column("created_at", sa.String(32), nullable=False),
        sa.Column("closed_at", sa.String(32), nullable=True),
        sa.Column("stopped_where", sa.String(500), nullable=True),
        sa.Column("next_step", sa.String(500), nullable=True),
        sa.Column("blocker", sa.String(500), nullable=True),
        sa.Column("self_answer", sa.String(2000), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_study_session"),
        sa.UniqueConstraint("plan_date", "slot"),
        sa.CheckConstraint("slot >= 1", name="slot_valid"),
        sa.CheckConstraint("kind IN ('learn', 'consolidate', 'minimum')", name="kind_valid"),
        sa.CheckConstraint("energy IN ('normal', 'low')", name="energy_valid"),
        sa.CheckConstraint("status IN ('open', 'closed')", name="status_valid"),
        sa.CheckConstraint(
            "status = 'open' OR (closed_at IS NOT NULL AND stopped_where IS NOT NULL "
            "AND next_step IS NOT NULL)",
            name="closed_has_note",
        ),
    )
    op.create_index("ix_study_session_plan_date", "study_session", ["plan_date"], unique=False)

    op.create_table(
        "session_block",
        sa.Column("id", sa.Integer(), nullable=False, autoincrement=True),
        sa.Column("session_id", sa.String(32), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(16), nullable=False),
        sa.Column("minutes", sa.Integer(), nullable=False),
        sa.Column("detail", sa.String(500), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_session_block"),
        sa.UniqueConstraint("session_id", "position"),
        sa.CheckConstraint("minutes > 0", name="minutes_positive"),
        sa.ForeignKeyConstraint(["session_id"], ["study_session.id"]),
    )
    op.create_index("ix_session_block_session_id", "session_block", ["session_id"], unique=False)


def downgrade() -> None:
    """Refuse destructive migrations consistently with the existing schema."""
    raise RuntimeError("dogma migrations are forward-only; restore from var/backups instead")
