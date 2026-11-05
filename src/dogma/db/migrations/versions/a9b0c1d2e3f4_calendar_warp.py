"""Add the calendar_warp log for the personal course calendar (I-30a, ADR-0019).

Forward-only. One new table and nothing else changes, so every existing row keeps its place.
The offset in force is the newest row's ``offset_weeks``; with no rows the calendar is the
authored one.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a9b0c1d2e3f4"
down_revision: str | None = "b8c9d0e1f2a3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create calendar_warp, the append-only log of calendar moves."""
    op.create_table(
        "calendar_warp",
        sa.Column("id", sa.Integer(), nullable=False, autoincrement=True),
        sa.Column("created_at", sa.String(32), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("offset_weeks", sa.Integer(), nullable=False),
        sa.Column("previous_offset_weeks", sa.Integer(), nullable=False),
        sa.Column("week_before", sa.Integer(), nullable=True),
        sa.Column("week_after", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_calendar_warp"),
        sa.CheckConstraint(
            "kind IN ('sync', 'to_week', 'ahead', 'back', 'original', 'course_reset')",
            name="kind_valid",
        ),
    )


def downgrade() -> None:
    """Refuse destructive migrations consistently with the existing schema."""
    raise RuntimeError("dogma migrations are forward-only; restore from var/backups instead")
