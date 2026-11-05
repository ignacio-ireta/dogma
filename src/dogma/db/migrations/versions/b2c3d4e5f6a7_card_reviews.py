"""Add FSRS cards and review history without changing existing progress."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b2c3d4e5f6a7"
down_revision: str | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create review storage; learner attempt and mastery rows remain untouched."""
    op.create_table(
        "card",
        sa.Column("id", sa.String(200), nullable=False),
        sa.Column("topic_id", sa.String(200), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("question", sa.String(), nullable=False),
        sa.Column("answer", sa.String(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("state", sa.Integer(), nullable=False),
        sa.Column("step", sa.Integer(), nullable=True),
        sa.Column("stability", sa.Float(), nullable=True),
        sa.Column("difficulty", sa.Float(), nullable=True),
        sa.Column("due", sa.String(32), nullable=False),
        sa.Column("last_review", sa.String(32), nullable=True),
        sa.CheckConstraint("state IN (1, 2, 3)", name="state_valid"),
        sa.PrimaryKeyConstraint("id", name="pk_card"),
    )
    op.create_index("ix_card_topic_id", "card", ["topic_id"])
    op.create_index("ix_card_due", "card", ["due"])
    op.create_table(
        "card_review",
        sa.Column("id", sa.String(32), nullable=False),
        sa.Column("card_id", sa.String(200), nullable=False),
        sa.Column("rating", sa.Integer(), nullable=False),
        sa.Column("reviewed_at", sa.String(32), nullable=False),
        sa.Column("due_before", sa.String(32), nullable=False),
        sa.Column("due_after", sa.String(32), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("question", sa.String(), nullable=False),
        sa.Column("answer", sa.String(), nullable=False),
        sa.CheckConstraint("rating IN (1, 2, 3, 4)", name="rating_valid"),
        sa.ForeignKeyConstraint(["card_id"], ["card.id"]),
        sa.PrimaryKeyConstraint("id", name="pk_card_review"),
    )
    op.create_index("ix_card_review_card_id", "card_review", ["card_id"])


def downgrade() -> None:
    """Refuse destructive migrations consistently with the existing schema."""
    raise RuntimeError("dogma migrations are forward-only; restore from var/backups instead")
