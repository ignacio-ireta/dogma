"""``dogma catalog build``: compile valid content into var/catalog.json (ARCHITECTURE §4.1).

Every content issue is printed, not just the first. When there are any, nothing is written and
the command exits 1. Re-running on unchanged content writes nothing and says so.
"""

from __future__ import annotations

import typer
from rich.markup import escape

from dogma.cli.context import out, settings_of
from dogma.content.catalog import build_catalog, write_catalog
from dogma.content.validate import validate_repo


def build_command(ctx: typer.Context) -> None:
    """Validate all content, then write var/catalog.json. Refuses to build on any issue."""
    settings = settings_of(ctx)
    console = out()
    issues = validate_repo(settings)
    if issues:
        console.print(f"[bold red]Catalog not built:[/] {len(issues)} content issue(s).")
        for issue in issues:
            console.print(
                f"  [red]{escape(issue.code)}[/] {escape(issue.path)}: {escape(issue.message)}"
            )
        raise typer.Exit(1)
    catalog = build_catalog(settings)
    path = settings.var_dir / "catalog.json"
    count = len(catalog.topics)
    if write_catalog(catalog, path):
        console.print(f"Wrote {escape(str(path))} ({count} topics).", soft_wrap=True)
    else:
        console.print(
            f"{escape(str(path))} is up to date ({count} topics); nothing written.", soft_wrap=True
        )
