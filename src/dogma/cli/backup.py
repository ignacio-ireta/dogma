"""``dogma backup``: export every table to one JSON file under var/backups (ENGINEERING §9)."""

from __future__ import annotations

from datetime import UTC, datetime

import typer

from dogma.cli.context import out, settings_of
from dogma.db.backup import write_backup


def backup_command(ctx: typer.Context) -> None:
    """Export every table to var/backups/dogma-<UTC time>.json and print where it went."""
    settings = settings_of(ctx)
    result = write_backup(settings, now=datetime.now(UTC))
    counts = ", ".join(f"{name}: {rows}" for name, rows in result.rows.items())
    console = out()
    if result.snapshot_path is not None:
        console.print(f"Pre-migration SQLite backup: {result.snapshot_path}", soft_wrap=True)
    console.print(f"Backup written: {result.path}", soft_wrap=True)
    console.print(f"Rows exported: {counts}", soft_wrap=True)
