"""``dogma today``: the last transition note, then the session to work through (ARCHITECTURE §4.4).

The note from the last closed session comes first, so the learner never starts by reconstructing
where they were. The plan follows with a time estimate for each block and exactly one first action.
Running it again resumes the open session. ``--energy low`` plans a minimum-viable session: review
and one familiar re-solve, with no new material.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from dogma.cli.context import out, settings_of
from dogma.learning import sessions
from dogma.learning.planner import Energy, SessionKind
from dogma.learning.sessions import StoredSession, TodayView, TransitionNote

KIND_LABEL = {
    SessionKind.LEARN: "Learn",
    SessionKind.CONSOLIDATE: "Consolidate",
    SessionKind.MINIMUM: "Minimum viable",
    SessionKind.INTERVIEW: "Interview",
}


def today_command(
    ctx: typer.Context,
    energy: Annotated[
        Energy | None,
        typer.Option(
            help="low: minimum-viable session (review and one familiar re-solve, no new material)."
            " Default: resume the open session, or plan a normal one."
        ),
    ] = None,
) -> None:
    """Show your last transition note, then today's session with one first action."""
    settings = settings_of(ctx)
    view = sessions.today(settings, now=datetime.now(UTC), energy=energy)
    console = out()
    if view.last_note is None:
        console.print("No earlier session yet. Start with the first action below.")
    else:
        _print_note(console, view.last_note)
    for day in view.stale_open:
        if view.current is not None and view.current.plan_date == day:
            continue
        console.print(
            f"[dim]The session from {day:%a %d %b %Y} was never closed, so it has no note.[/]",
            soft_wrap=True,
        )
    if view.current is None:
        console.print(
            f"All {view.sessions_per_day} sessions for {view.local_date:%a %d %b %Y} are closed. "
            "Nothing else is planned today; the next session starts tomorrow."
        )
        return
    _print_plan(console, view, view.current)


def _print_note(console: Console, note: TransitionNote) -> None:
    where = f" · {escape(note.topic_id)}" if note.topic_id is not None else ""
    console.print(
        f"[bold]Last session[/] · {KIND_LABEL[note.kind]} · {note.plan_date:%a %d %b %Y}{where}"
    )
    console.print(f"  Stopped at: {escape(note.stopped_where)}", soft_wrap=True)
    console.print(f"  Next step:  {escape(note.next_step)}", soft_wrap=True)
    console.print(f"  Blocker:    {escape(note.blocker or 'none')}", soft_wrap=True)


def _print_plan(console: Console, view: TodayView, current: StoredSession) -> None:
    plan = current.plan
    style = "normal energy" if plan.energy is Energy.NORMAL else "low energy"
    console.print(
        f"\n[bold]Today[/] · {view.local_date:%a %d %b %Y} · session {current.slot} of "
        f"{current.sessions_per_day} · {KIND_LABEL[plan.kind]} · {plan.minutes} min · {style}",
        soft_wrap=True,
    )
    if view.replanned:
        console.print(f"[dim]Re-planned at {plan.energy.value} energy.[/]")
    elif view.resumed:
        console.print("[dim]Resuming the session you planned earlier.[/]")
    if current.plan_date != view.local_date:
        console.print(
            f"[dim]Originally planned for {current.plan_date:%a %d %b %Y}. "
            "Close this session before starting a new one.[/]"
        )
    table = Table(show_edge=False)
    table.add_column("Block", no_wrap=True)
    table.add_column("Min", justify="right", no_wrap=True)
    table.add_column("What", overflow="fold")
    for block in plan.blocks:
        table.add_row(block.name, str(block.minutes), escape(block.detail))
    console.print(table)
    for note in plan.notes:
        console.print(f"[yellow]{escape(note)}[/]", soft_wrap=True)
    console.print(f"[bold]First action:[/] {escape(plan.first_action)}", soft_wrap=True)
    console.print("[dim]When you stop: dogma close[/]")
