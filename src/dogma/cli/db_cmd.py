"""``dogma db upgrade``: create or migrate var/dogma.db (ARCHITECTURE §4.9)."""

from __future__ import annotations

import typer

from dogma.cli.context import out, settings_of
from dogma.db.session import database_path, upgrade_database


def upgrade_command(ctx: typer.Context) -> None:
    """Create the database if needed and apply every pending migration. Safe to re-run."""
    settings = settings_of(ctx)
    result = upgrade_database(settings)
    path = database_path(settings)
    console = out()
    if result.before is None:
        console.print(f"Created {path} at schema revision {result.after}.", soft_wrap=True)
    elif result.changed:
        console.print(
            f"Upgraded {path} from revision {result.before} to {result.after}.", soft_wrap=True
        )
    else:
        console.print(f"{path} is already at schema revision {result.after}.", soft_wrap=True)
