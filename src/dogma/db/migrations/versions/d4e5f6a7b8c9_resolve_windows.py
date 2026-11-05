"""Add timed re-solve windows (ARCHITECTURE §4.4, §4.9, I-13).

Forward-only. Existing attempt, topic_state, card, card_review, study_session and session_block
rows are not touched. The partial unique index allows at most one open window.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d4e5f6a7b8c9"
down_revision: str | None = "c3d4e5f6a7b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the window table and its one-open-window index. Nothing that exists changes."""
    op.create_table(
        "resolve_window",
        sa.Column("id", sa.String(32), nullable=False),
        sa.Column("problem_id", sa.String(200), nullable=False),
        sa.Column("problem_version", sa.Integer(), nullable=False),
        sa.Column("topic_id", sa.String(200), nullable=False),
        sa.Column("started_at", sa.String(32), nullable=False),
        sa.Column("budget_s", sa.Integer(), nullable=False),
        sa.Column("hints_used", sa.Integer(), server_default="0", nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("attempt_id", sa.String(64), nullable=True),
        sa.Column("closed_at", sa.String(32), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_resolve_window"),
        sa.CheckConstraint("budget_s > 0", name="budget_positive"),
        sa.CheckConstraint("hints_used >= 0", name="hints_nonnegative"),
        sa.CheckConstraint(
            "status IN ('open', 'passed', 'partial', 'failed', 'abandoned')",
            name="status_valid",
        ),
        sa.CheckConstraint(
            "(status = 'open' AND closed_at IS NULL) "
            "OR (status <> 'open' AND closed_at IS NOT NULL)",
            name="closed_matches_status",
        ),
        sa.CheckConstraint(
            "(status IN ('passed', 'partial', 'failed') AND attempt_id IS NOT NULL) "
            "OR (status IN ('open', 'abandoned') AND attempt_id IS NULL)",
            name="attempt_matches_verdict",
        ),
        sa.ForeignKeyConstraint(["attempt_id"], ["attempt.id"]),
    )
    op.create_index("ix_resolve_window_problem_id", "resolve_window", ["problem_id"], unique=False)
    op.create_index("ix_resolve_window_topic_id", "resolve_window", ["topic_id"], unique=False)
    op.create_index(
        "ix_resolve_window_one_open",
        "resolve_window",
        ["status"],
        unique=True,
        sqlite_where=sa.text("status = 'open'"),
    )


def downgrade() -> None:
    """Refuse destructive migrations consistently with the existing schema."""
    raise RuntimeError("dogma migrations are forward-only; restore from var/backups instead")
