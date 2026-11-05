"""Alembic, configured in code. The repo has no alembic.ini: migrations live in this package.

``upgrade_to_head`` runs every pending migration on a connection that the caller owns, in one
transaction. Migrations are forward-only (ENGINEERING_STANDARDS §9): downgrading refuses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Connection, Engine

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


@dataclass(frozen=True)
class UpgradeResult:
    """The schema revision before and after an upgrade. ``before`` is None for a new file."""

    before: str | None
    after: str

    @property
    def changed(self) -> bool:
        return self.before != self.after


def alembic_config() -> Config:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    return config


def head_revision() -> str:
    head = ScriptDirectory.from_config(alembic_config()).get_current_head()
    if head is None:
        raise RuntimeError(f"no migrations found in {MIGRATIONS_DIR}")
    return head


def current_revision(connection: Connection) -> str | None:
    return MigrationContext.configure(connection).get_current_revision()


def upgrade_to_head(engine: Engine) -> UpgradeResult:
    """Apply every pending migration in a single transaction. Re-running it does nothing."""
    config = alembic_config()
    with engine.begin() as connection:
        before = current_revision(connection)
        config.attributes["connection"] = connection
        command.upgrade(config, "head")
        after = current_revision(connection)
    if after is None:
        raise RuntimeError("the upgrade finished without a schema revision")
    return UpgradeResult(before=before, after=after)
