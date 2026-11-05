"""Persist timed re-solve targets independently of learner-facing instructions.

Backfill only the two known legacy timed-action templates. All existing rows and instructions
are retained. New plans write the target directly; prose parsing ends at this migration.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b8c9d0e1f2a3"
down_revision: str | None = "a7b8c9d0e1f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add a nullable target and preserve existing timed plans across the upgrade."""
    op.add_column("study_session", sa.Column("timed_problem_id", sa.String(200), nullable=True))
    target = re.compile(
        r"^(?:Start the timed re-solve: dogma resolve |"
        r"Finish the open timed re-solve: dogma submit )([\w-]+/[\w-]+/[\w-]+)(?=[ ,(]|$)"
    )
    connection = op.get_bind()
    for session_id, action in connection.execute(
        sa.text("SELECT id, first_action FROM study_session")
    ):
        match = target.match(action)
        if match is not None:
            connection.execute(
                sa.text("UPDATE study_session SET timed_problem_id = :target WHERE id = :id"),
                {"target": match.group(1), "id": session_id},
            )


def downgrade() -> None:
    """Refuse destructive migrations consistently with the existing schema."""
    raise RuntimeError("dogma migrations are forward-only; restore from var/backups instead")
