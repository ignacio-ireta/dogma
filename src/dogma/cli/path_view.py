"""``dogma path``: the core path week by week, with each topic's content status (ROADMAP I-05).

Without ``--week`` it shows the study week that contains today (week 1 before the path starts).
The calendar and the topic statuses come from ``dogma.learning.path``; this module renders them.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated

import typer
from rich.markup import escape
from rich.table import Table

from dogma.cli.context import CliError, out, settings_of
from dogma.content.schemas import PathWeek
from dogma.core.config import Settings
from dogma.learning.calendar import current_offset, load_calendar, offset_note
from dogma.learning.path import READY, TopicRow, current_week, topic_row

_STYLE = {
    "reviewed": "bold green",
    "verified": "green",
    "draft": "yellow",
    "retired": "dim",
    "missing": "dim",
    "invalid": "bold red",
}


def path_command(
    ctx: typer.Context,
    week: Annotated[
        int | None,
        typer.Option("--week", min=1, help="Show this study week (1-based). Default: this week."),
    ] = None,
) -> None:
    """Show the core path: each week's theme, dates, topics, their status and problems."""
    settings = settings_of(ctx)
    core = load_calendar(settings)
    if not core.weeks:
        raise CliError("the core path has no weeks")
    if week is None:
        chosen = [current_week(core.weeks, date.today())]
    else:
        matches = [item for item in core.weeks if item.week == week]
        if not matches:
            raise CliError(f"the core path has weeks 1 to {len(core.weeks)}, not {week}")
        chosen = matches
    console = out()
    console.print(f"[bold]{escape(core.title)}[/] · {core.start:%d %b %Y} to {core.end:%d %b %Y}")
    note = offset_note(current_offset(settings))
    if note is not None:
        console.print(note, soft_wrap=True)
    for item in chosen:
        heading, table = _week_block(settings, item)
        console.print(f"\n{heading}")
        console.print(table)
    console.print("Open a problem with: dogma open <topic id>/<problem name>")


def _week_block(settings: Settings, week: PathWeek) -> tuple[str, Table]:
    rows = [topic_row(settings, topic_id) for topic_id in week.topics]
    ready = sum(row.status in READY for row in rows)
    heading = (
        f"[bold]Week {week.week}[/] · {escape(week.theme)} · starts {week.starts:%a %d %b %Y}"
        f" · {ready}/{len(rows)} topics ready"
    )
    table = Table(show_edge=False)
    table.add_column("Status", no_wrap=True)
    table.add_column("Topic", overflow="fold")
    table.add_column("Problems", overflow="fold")
    for row in rows:
        table.add_row(
            f"[{_STYLE.get(row.status, '')}]{row.status}[/]",
            _topic_cell(row),
            "\n".join(escape(name) for name in row.problems) or "[dim]none yet[/]",
        )
    return heading, table


def _topic_cell(row: TopicRow) -> str:
    if not row.title:
        return escape(row.topic_id)
    return f"{escape(row.topic_id)}\n[dim]{escape(row.title)}[/]"
