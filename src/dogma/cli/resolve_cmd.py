"""``dogma resolve``: a timed, hint-free re-solve of a core problem (ARCHITECTURE §4.4, §4.7).

``dogma resolve <problem>`` starts the clock and restores the starter (your old file is kept under
``var/workspace-history/``). The first ``dogma submit`` of that problem decides the re-solve.
``--status`` shows the open one, and ``--abandon`` stops it with no penalty. The verdict
printer lives here too, so ``dogma submit`` and ``dogma resolve`` describe a result the same way.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape

from dogma.cli.context import CliError, out, settings_of
from dogma.db.resolve_models import WindowStatus
from dogma.learning import resolve_service
from dogma.learning.resolve_service import StartedWindow
from dogma.learning.resolves import ClosedWindow, ReSolveSchedule, WindowRecord

ProblemArg = Annotated[
    str | None,
    typer.Argument(help="Core problem id, e.g. cml/vector-operations/p01-vector-projection."),
]
StatusOpt = Annotated[
    bool, typer.Option("--status", help="Show the open timed re-solve, if there is one.")
]
AbandonOpt = Annotated[
    bool, typer.Option("--abandon", help="Stop the open timed re-solve. No penalty.")
]


def resolve_command(
    ctx: typer.Context,
    problem: ProblemArg = None,
    status: StatusOpt = False,
    abandon: AbandonOpt = False,
) -> None:
    """Start a timed re-solve: the budget runs, no hints, and the first submit decides."""
    settings = settings_of(ctx)
    now = datetime.now(UTC)
    console = out()
    if status and abandon:
        raise CliError("choose --status or --abandon, not both.")
    if abandon:
        window = resolve_service.abandon(settings, problem, now=now)
        console.print(
            f"Abandoned the timed re-solve of {escape(window.problem_id)}. No penalty: the "
            "schedule did not change.",
            soft_wrap=True,
        )
        return
    if status:
        _print_status(console, resolve_service.status(settings, now=now), now)
        return
    if problem is None:
        raise CliError("name a problem, for example: dogma resolve <problem>, or pass --status.")
    _print_started(console, resolve_service.start(settings, problem, now=now))


def render_closed(console: Console, closed: ClosedWindow, now: datetime) -> None:
    """Say what a submit did to the open timed re-solve, and when the next one is due."""
    console.print(_verdict_line(closed), soft_wrap=True)
    if closed.next_due is not None:
        console.print(_next_line(closed.next_due, now), soft_wrap=True)


def clock(seconds: float) -> str:
    """Seconds as ``14 min 32 s``, the form every timed result uses."""
    whole = round(seconds)
    return f"{whole // 60} min {whole % 60:02d} s"


def _print_started(console: Console, started: StartedWindow) -> None:
    window = started.window
    minutes = window.budget_s // 60
    ends = (window.started_at + timedelta(seconds=window.budget_s)).astimezone()
    console.print(f"Timed re-solve open: {escape(window.problem_id)}")
    console.print(
        f"Time budget {minutes} min, ends {ends:%H:%M} local. No hints: a hint counts as "
        "practice, not a timed pass.",
        soft_wrap=True,
    )
    console.print(
        "Rules: run is unlimited. The first submit decides: passed when every test passes "
        "within the budget with no hints; partial when it is late or hinted; failed otherwise.",
        soft_wrap=True,
    )
    console.print(f"Starter copied to: {escape(str(started.workspace))}", soft_wrap=True)
    if started.kept is not None:
        console.print(f"Previous file kept: {escape(str(started.kept))}", soft_wrap=True)
    console.print(f"Submit with: dogma submit {escape(window.problem_id)}", soft_wrap=True)


def _print_status(console: Console, window: WindowRecord | None, now: datetime) -> None:
    if window is None:
        console.print("No timed re-solve is open.")
        return
    used = clock((now - window.started_at).total_seconds())
    console.print(
        f"Open timed re-solve: {escape(window.problem_id)} · started "
        f"{window.started_at.astimezone():%H:%M} · {used} of {window.budget_s // 60} min used · "
        f"hints used: {window.hints_used}",
        soft_wrap=True,
    )
    console.print(
        f"Submit with: dogma submit {escape(window.problem_id)}, or stop it with: "
        f"dogma resolve {escape(window.problem_id)} --abandon",
        soft_wrap=True,
    )


def _verdict_line(closed: ClosedWindow) -> str:
    budget = closed.budget_s // 60
    used = clock(closed.used_s)
    if closed.status is WindowStatus.PASSED:
        return f"Timed re-solve passed: {used} of {budget} min, no hints."
    if closed.status is WindowStatus.FAILED:
        return "Timed re-solve failed: not every test passed."
    reasons = []
    if closed.used_s > closed.budget_s:
        reasons.append(f"over budget ({used} of {budget} min)")
    if closed.hints_used:
        reasons.append(f"hints used ({closed.hints_used})")
    tail = ", so it counts as practice." if closed.hints_used else "."
    return f"Timed re-solve partial: {' and '.join(reasons)}{tail}"


def _next_line(schedule: ReSolveSchedule, now: datetime) -> str:
    due = schedule.due_at.astimezone()
    when = "due now" if schedule.is_due(now) else f"due {due:%a %d %b %Y}"
    return (
        f"Next timed re-solve: {escape(schedule.problem_id)} {when}, stage "
        f"{schedule.stage} of 3. {escape(schedule.reason)}"
    )
