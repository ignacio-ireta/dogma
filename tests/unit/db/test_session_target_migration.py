"""Legacy plans keep their data and acquire structured timed targets once, at upgrade."""

from __future__ import annotations

import pytest
from alembic import command
from sqlalchemy import text

from dogma.core.config import Settings
from dogma.db.engine import make_engine
from dogma.db.migrate import alembic_config
from dogma.db.session import database_path, ensure_schema


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (
            "Start the timed re-solve: dogma resolve tst/a/p01-core (budget 20 min, no hints).",
            "tst/a/p01-core",
        ),
        (
            "Finish the open timed re-solve: dogma submit tst/a/p01-core, or abandon.",
            "tst/a/p01-core",
        ),
        ("Open tst/a/p01-core for untimed practice.", None),
        ("Start the mock: dogma mock start", None),
    ],
)
def test_migration_preserves_legacy_plan_and_backfills_only_timed_targets(
    settings: Settings, action: str, expected: str | None
) -> None:
    engine = make_engine(database_path(settings))
    try:
        with engine.begin() as connection:
            config = alembic_config()
            config.attributes["connection"] = connection
            command.upgrade(config, "a7b8c9d0e1f2")
            connection.execute(
                text(
                    "INSERT INTO study_session (id, plan_date, slot, kind, energy, status, "
                    "first_action, created_at) VALUES ('legacy', '2026-10-28', 3, 'consolidate', "
                    "'normal', 'open', :action, '2026-10-28T12:00:00.000000+00:00')"
                ),
                {"action": action},
            )
            before = tuple(connection.exec_driver_sql("SELECT * FROM study_session").one())
        ensure_schema(engine)
        with engine.connect() as connection:
            after = tuple(connection.exec_driver_sql("SELECT * FROM study_session").one())
        assert after == (*before, expected)
    finally:
        engine.dispose()
