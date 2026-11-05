"""Alembic environment: runs migrations on the connection that ``dogma.db.migrate`` provides.

Offline mode (``alembic upgrade --sql``) is not supported; the app only ever migrates a live file.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import Connection

from dogma.db import (  # noqa: F401  registers tables
    answer_models,
    calendar_models,
    models,
    resolve_models,
    review_models,
    session_models,
)
from dogma.db.base import Base


def run_migrations() -> None:
    connection = context.config.attributes.get("connection")
    if not isinstance(connection, Connection):
        raise RuntimeError("no connection: run migrations through dogma.db.migrate")
    context.configure(
        connection=connection,
        target_metadata=Base.metadata,
        render_as_batch=True,  # SQLite cannot ALTER most things; batch mode rebuilds the table
    )
    with context.begin_transaction():
        context.run_migrations()


run_migrations()
