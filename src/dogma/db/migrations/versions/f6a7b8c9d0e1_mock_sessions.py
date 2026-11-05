"""Add mock sessions, link mock scores to them, and let planned sessions be interviews (I-18c).

Forward-only. Three things change:

- ``mock_session`` is a new table, with at most one open mock at a time.
- ``answer`` gains ``mock_session_id``, set on the 30 item scores of a mock.
- ``study_session`` and ``session_block`` are rebuilt, because SQLite cannot change a CHECK
  constraint in place. ``study_session.kind`` gains ``interview``. Every row is copied with its
  id, so every block still points at its session.

Tables that other tables point at are renamed aside, copied, and dropped last, in an order that
keeps every foreign key valid. Timestamps stay UTC ISO-8601 text.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f6a7b8c9d0e1"
down_revision: str | None = "e5f6a7b8c9d0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SESSION_COLUMNS = (
    "id, plan_date, slot, kind, energy, status, topic_id, must_do, first_action, plan_notes, "
    "created_at, closed_at, stopped_where, next_step, blocker, self_answer"
)
_BLOCK_COLUMNS = "id, session_id, position, name, minutes, detail"
_ANSWER_COLUMNS = (
    "id, topic_id, kind, prompt_ref, text, rubric_id, rubric_version, scores, notes, created_at"
)


def upgrade() -> None:
    """Create mock_session, link answers to it, and rebuild the two session tables."""
    op.create_table(
        "mock_session",
        sa.Column("id", sa.String(32), nullable=False),
        sa.Column("started_at", sa.String(32), nullable=False),
        sa.Column("ended_at", sa.String(32), nullable=True),
        sa.Column("case_id", sa.String(200), nullable=True),
        sa.Column("coding_problem_id", sa.String(200), nullable=True),
        sa.Column("theory_topic_id", sa.String(200), nullable=True),
        sa.Column("theory_question", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("part_minutes", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_mock_session"),
        sa.CheckConstraint("status IN ('open', 'scored', 'abandoned')", name="status_valid"),
        sa.CheckConstraint(
            "(status = 'open' AND ended_at IS NULL) OR (status <> 'open' AND ended_at IS NOT NULL)",
            name="ended_matches_status",
        ),
        sa.CheckConstraint(
            "theory_question IS NULL OR theory_question >= 1", name="theory_question_valid"
        ),
    )
    op.create_index(
        "ix_mock_session_one_open",
        "mock_session",
        ["status"],
        unique=True,
        sqlite_where=sa.text("status = 'open'"),
    )
    _rebuild_answer()
    _rebuild_sessions()


def _rebuild_answer() -> None:
    op.execute("ALTER TABLE answer RENAME TO _answer_old")
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
        sa.Column("mock_session_id", sa.String(32), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_answer"),
        sa.CheckConstraint("kind IN ('interview', 'close', 'mock')", name="kind_valid"),
        sa.CheckConstraint("rubric_version >= 1", name="rubric_version_valid"),
        sa.ForeignKeyConstraint(["mock_session_id"], ["mock_session.id"]),
    )
    op.execute(
        f"INSERT INTO answer ({_ANSWER_COLUMNS}) "  # noqa: S608  constant column lists
        f"SELECT {_ANSWER_COLUMNS} FROM _answer_old"
    )
    op.execute("DROP TABLE _answer_old")
    op.create_index("ix_answer_topic_id", "answer", ["topic_id"], unique=False)
    op.create_index("ix_answer_mock_session_id", "answer", ["mock_session_id"], unique=False)


def _rebuild_sessions() -> None:
    op.execute("ALTER TABLE study_session RENAME TO _study_session_old")
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
        sa.CheckConstraint(
            "kind IN ('learn', 'consolidate', 'minimum', 'interview')", name="kind_valid"
        ),
        sa.CheckConstraint("energy IN ('normal', 'low')", name="energy_valid"),
        sa.CheckConstraint("status IN ('open', 'closed')", name="status_valid"),
        sa.CheckConstraint(
            "status = 'open' OR (closed_at IS NOT NULL AND stopped_where IS NOT NULL "
            "AND next_step IS NOT NULL)",
            name="closed_has_note",
        ),
    )
    op.execute(
        f"INSERT INTO study_session ({_SESSION_COLUMNS}) "  # noqa: S608  constant column lists
        f"SELECT {_SESSION_COLUMNS} FROM _study_session_old"
    )
    op.execute("ALTER TABLE session_block RENAME TO _session_block_old")
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
    op.execute(
        f"INSERT INTO session_block ({_BLOCK_COLUMNS}) "  # noqa: S608  constant column lists
        f"SELECT {_BLOCK_COLUMNS} FROM _session_block_old"
    )
    op.execute("DROP TABLE _session_block_old")
    op.execute("DROP TABLE _study_session_old")
    op.create_index("ix_study_session_plan_date", "study_session", ["plan_date"], unique=False)
    op.create_index("ix_session_block_session_id", "session_block", ["session_id"], unique=False)


def downgrade() -> None:
    """Refuse destructive migrations consistently with the existing schema."""
    raise RuntimeError("dogma migrations are forward-only; restore from var/backups instead")
