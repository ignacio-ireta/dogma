"""Initial schema: attempt and topic_state (ARCHITECTURE §4.9).

Forward-only. Timestamps are UTC ISO-8601 text, matching dogma.db.base.UtcDateTime.
Constraint names are bare: the naming convention in dogma.db.base adds the
``ck_<table>_`` prefix, as it does for the models.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "attempt",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("problem_id", sa.String(length=200), nullable=False),
        sa.Column("problem_version", sa.Integer(), nullable=False),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("public_passed", sa.Integer(), nullable=False),
        sa.Column("public_total", sa.Integer(), nullable=False),
        sa.Column("hidden_passed", sa.Integer(), nullable=False),
        sa.Column("hidden_total", sa.Integer(), nullable=False),
        sa.Column("hints_used", sa.Integer(), server_default="0", nullable=False),
        sa.Column("run_dir", sa.String(length=255), nullable=True),
        sa.Column("started_at", sa.String(length=32), nullable=False),
        sa.Column("submitted_at", sa.String(length=32), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.CheckConstraint("mode IN ('run', 'submit')", name="mode_valid"),
        sa.PrimaryKeyConstraint("id", name="pk_attempt"),
    )
    op.create_index("ix_attempt_problem_id", "attempt", ["problem_id"], unique=False)

    op.create_table(
        "topic_state",
        sa.Column("topic_id", sa.String(length=200), nullable=False),
        sa.Column("mastery", sa.String(length=32), nullable=False),
        sa.Column("introduced_at", sa.String(length=32), nullable=True),
        sa.Column("solved_at", sa.String(length=32), nullable=True),
        sa.Column("retained_at", sa.String(length=32), nullable=True),
        sa.Column("ready_at", sa.String(length=32), nullable=True),
        sa.CheckConstraint(
            "mastery IN ('introduced', 'solved', 'consolidating', 'retained', 'interview-ready')",
            name="mastery_valid",
        ),
        sa.PrimaryKeyConstraint("topic_id", name="pk_topic_state"),
    )


def downgrade() -> None:
    raise RuntimeError("dogma migrations are forward-only; restore from var/backups instead")
