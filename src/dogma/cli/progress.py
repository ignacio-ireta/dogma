"""``dogma progress``: how far the core path has got, week by week (ARCHITECTURE §4.4, §4.8).

Read-only. Each topic shows its mastery, its attempts, its best result and its last attempt. The
summary counts topics solved across the whole path and attempts in the current study week. Without
a database yet, everything reads as not started, and no database is created.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import UTC, date, datetime
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from dogma.cli.context import CliError, out, settings_of
from dogma.cli.resolve_cmd import clock
from dogma.content.schemas import CorePath, PathWeek
from dogma.core.config import Settings
from dogma.db.models import Mastery
from dogma.db.progress import (
    Snapshot,
    TopicProgress,
    attempts_between,
    read_snapshot,
    topic_progress,
)
from dogma.learning import resolve_service
from dogma.learning.calendar import load_calendar
from dogma.learning.path import current_week, topic_row

_MASTERY_STYLE = {Mastery.UNSEEN: "dim", Mastery.INTRODUCED: "yellow", Mastery.SOLVED: "bold green"}


def progress_command(
    ctx: typer.Context,
    week: Annotated[
        int | None,
        typer.Option("--week", min=1, help="Show this study week only (1-based). Default: all."),
    ] = None,
) -> None:
    """Show mastery, attempts and best result for each topic on the core path."""
    settings = settings_of(ctx)
    core = load_calendar(settings)
    chosen = _chosen_weeks(core.weeks, week)
    now = datetime.now(UTC)
    snapshot = read_snapshot(settings, now=now)
    console = out()
    console.print(f"[bold]{escape(core.title)}[/]")
    for item in chosen:
        heading, table = _week_block(settings, item, snapshot)
        console.print(f"\n{heading}", soft_wrap=True)
        console.print(table)
    _print_readiness(console, chosen, snapshot)
    console.print(f"\n{summary_line(core.weeks, snapshot, date.today())}", soft_wrap=True)
    _print_resolves(console, settings, core, snapshot, now)


def _print_readiness(console: Console, weeks: Sequence[PathWeek], snapshot: Snapshot) -> None:
    """Per study week, how many topics are interview-ready. A count, never a streak."""
    console.print("\n[bold]Readiness[/]")
    for item in weeks:
        ready = sum(
            topic_progress(topic_id, snapshot).mastery is Mastery.INTERVIEW_READY
            for topic_id in item.topics
        )
        console.print(
            f"Week {item.week} · interview-ready {ready} of {len(item.topics)} topics",
            soft_wrap=True,
        )


def _chosen_weeks(weeks: Sequence[PathWeek], week: int | None) -> Sequence[PathWeek]:
    if not weeks:
        raise CliError("the core path has no weeks")
    if week is None:
        return weeks
    matches = [item for item in weeks if item.week == week]
    if not matches:
        raise CliError(f"the core path has weeks 1 to {len(weeks)}, not {week}")
    return matches


def _week_block(settings: Settings, week: PathWeek, snapshot: Snapshot) -> tuple[str, Table]:
    rows = [
        (topic_row(settings, topic_id).title, topic_progress(topic_id, snapshot))
        for topic_id in week.topics
    ]
    solved = sum(progress.solved for _, progress in rows)
    heading = (
        f"[bold]Week {week.week}[/] · {escape(week.theme)} · starts {week.starts:%a %d %b %Y}"
        f" · {solved}/{len(rows)} topics solved\n"
        f"[dim]Mastery:[/] {_mastery_counts([progress for _, progress in rows])}"
    )
    table = Table(show_edge=False)
    table.add_column("Topic", overflow="fold")
    table.add_column("Mastery", no_wrap=True)
    table.add_column("Attempts", justify="right", no_wrap=True)
    table.add_column("Best result", no_wrap=True)
    table.add_column("Last attempt", no_wrap=True)
    for title, progress in rows:
        table.add_row(
            _topic_cell(progress.topic_id, title),
            _mastery_cell(progress),
            str(progress.attempts),
            _best_cell(progress),
            _last_cell(progress),
        )
    return heading, table


def summary_line(weeks: Sequence[PathWeek], snapshot: Snapshot, today: date) -> str:
    """Topics solved across the whole path, and attempts in the study week that holds ``today``."""
    topics = {topic for item in weeks for topic in item.topics}
    solved = sum(topic_progress(topic, snapshot).solved for topic in topics)
    current = current_week(tuple(weeks), today)
    # Before the path starts, the current week counts from today, so practice done now shows up.
    start = min(current.starts, today)
    later = [item.starts for item in weeks if item.starts > current.starts]
    attempts = attempts_between(snapshot, start, min(later) if later else None)
    return (
        f"Solved {solved} of {len(topics)} topics in the core path · "
        f"{attempts} {'attempt' if attempts == 1 else 'attempts'} in week {current.week}"
        f" (from {start:%d %b})"
    )


def _mastery_counts(progresses: Sequence[TopicProgress]) -> str:
    """Topics per mastery state, in path order of the states, zero counts left out."""
    counts = Counter(progress.mastery for progress in progresses)
    parts = [
        f"{'not started' if state is Mastery.UNSEEN else state.value} {counts[state]}"
        for state in Mastery
        if counts[state]
    ]
    return " · ".join(parts)


def _print_resolves(
    console: Console, settings: Settings, core: CorePath, snapshot: Snapshot, now: datetime
) -> None:
    """The next three due timed re-solves and the last five results, on core problems only."""
    view = resolve_service.readiness(
        settings, (topic for week in core.weeks for topic in week.topics), snapshot, now
    )
    recent = view.recent
    console.print("\n[bold]Re-solves[/]")
    soonest = view.upcoming
    if not soonest:
        console.print(
            "Next due: none yet. A timed re-solve is scheduled once you pass a core problem.",
            soft_wrap=True,
        )
    else:
        console.print("Next due:", soft_wrap=True)
        for item in soonest:
            when = "due now" if item.is_due(now) else f"due {item.due_at.astimezone():%d %b %Y}"
            console.print(
                f"  {escape(item.problem_id)} · stage {item.stage} of 3 · {when}",
                soft_wrap=True,
            )
    if not recent:
        console.print("Last timed results: none yet.", soft_wrap=True)
    else:
        console.print("Last timed results:", soft_wrap=True)
        for result in recent:
            console.print(
                f"  {escape(result.problem_id)} · {result.status.value} · "
                f"{clock(result.used_s)} of {result.budget_s // 60} min",
                soft_wrap=True,
            )


def _topic_cell(topic_id: str, title: str) -> str:
    if not title:
        return escape(topic_id)
    return f"{escape(topic_id)}\n[dim]{escape(title)}[/]"


def _mastery_cell(progress: TopicProgress) -> str:
    style = _MASTERY_STYLE.get(progress.mastery, "green")
    return f"[{style}]{progress.mastery.value}[/]"


def _best_cell(progress: TopicProgress) -> str:
    best = progress.best
    if best is None:
        return "[dim]none yet[/]"
    style = "green" if best.passed == best.total else "yellow"
    return f"[{style}]{best.passed}/{best.total}[/] · {best.mode}"


def _last_cell(progress: TopicProgress) -> str:
    if progress.last_attempt_at is None:
        return "[dim]never[/]"
    return f"{progress.last_attempt_at.astimezone():%d %b %Y}"
