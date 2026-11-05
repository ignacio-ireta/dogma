"""Store advisory model scores next to an answer (I-18d).

Forward-only. ``answer`` gains one nullable JSON column, ``model_score``. It holds the model's
advisory scores and their quotes. No existing row changes, and no self-score or mastery reads it.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a7b8c9d0e1f2"
down_revision: str | None = "f6a7b8c9d0e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the nullable advisory column. Existing answers keep NULL."""
    op.add_column("answer", sa.Column("model_score", sa.JSON(), nullable=True))


def downgrade() -> None:
    """Refuse destructive migrations consistently with the existing schema."""
    raise RuntimeError("dogma migrations are forward-only; restore from var/backups instead")
