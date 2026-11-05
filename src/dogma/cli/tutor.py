"""``dogma hint`` and ``dogma debug``: the tutor's terminal adapters (ARCHITECTURE §4.5, §4.7).

``hint`` at levels 1 to 3 prints the problem's reviewed hint (hints.md) before any model call, so it
shows at once and works when the model is down. Then the model's nudge on the learner's code
follows, or one note says the nudge needs the local model. Every model reply is printed as plain
text. A
failure that is not a down model is one ``error [CODE]`` line from ``friendly_errors``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

import typer
from rich.markup import escape

from dogma.cli.context import out, settings_of
from dogma.content.resolve import resolve_problem
from dogma.core.config import Settings
from dogma.learning import resolve_service
from dogma.learning.resolves import WindowRecord
from dogma.tutor import service
from dogma.tutor.gateway import Gateway
from dogma.tutor.models import ReviewedHint, TutorAnswer

ProblemArg = Annotated[
    str,
    typer.Argument(
        help="Problem id, e.g. cml/vector-operations/p01-vector-projection, or its path."
    ),
]
LevelOpt = Annotated[
    int,
    typer.Option(
        "--level",
        min=1,
        max=4,
        help="1 concept, 2 approach, 3 code shape: the reviewed hint, then a nudge on your code "
        "when the local model answers. 4 walkthrough (locked until two recorded submits fail, "
        "or confirm it).",
    ),
]
YesOpt = Annotated[
    bool,
    typer.Option(
        "--yes",
        help="Open the walkthrough (L4), and confirm a hint in a timed re-solve, without asking.",
    ),
]
_LEVEL_NAMES = {1: "concept", 2: "approach", 3: "code shape", 4: "walkthrough"}


def make_gateway(settings: Settings) -> Gateway:
    """The gateway the commands use. Tests replace this to put a fake provider behind it."""
    return Gateway(settings)


def hint_command(
    ctx: typer.Context,
    problem: ProblemArg,
    level: LevelOpt = 1,
    yes: YesOpt = False,
) -> None:
    """Show the reviewed hint for a level at once, then a nudge on your code.

    L4, the walkthrough, needs two failed submits or your confirmation.
    """
    settings = settings_of(ctx)
    ref = resolve_problem(settings, problem)
    now = datetime.now(UTC)
    window = resolve_service.open_window_for(settings, ref.id, now=now)
    if window is not None and not _confirm_during_resolve(window, yes, what="Hint"):
        return
    target = service.Target(problem_dir=ref.dir, problem_id=ref.id, meta=ref.meta)
    prepared = service.prepare_hint(settings, target, level=level, yes=yes, ask=_ask)
    if window is not None:
        resolve_service.record_hint(settings, ref.id, now=now)
    if prepared.reviewed is not None:
        _print_reviewed(prepared.reviewed)
    elif prepared.hints_note is not None:
        _print_note(prepared.hints_note)
    answer = prepared.complete(make_gateway(settings))
    if answer.nudge is not None:
        heading = (
            f"Personalised nudge · L{level} · model"
            if answer.reviewed is not None
            else f"Hint · L{level} · {_LEVEL_NAMES[level]}"
        )
        _print(answer.nudge, heading=heading)
    if answer.nudge_note is not None:
        _print_note(answer.nudge_note)


def debug_command(ctx: typer.Context, problem: ProblemArg, yes: YesOpt = False) -> None:
    """Name the first wrong concept or line after a failing submit. It never rewrites your code."""
    settings = settings_of(ctx)
    ref = resolve_problem(settings, problem)
    now = datetime.now(UTC)
    window = resolve_service.open_window_for(settings, ref.id, now=now)
    if window is not None and not _confirm_during_resolve(window, yes, what="Debug"):
        return
    target = service.Target(problem_dir=ref.dir, problem_id=ref.id, meta=ref.meta)
    answer = service.debug(settings, target, gateway=make_gateway(settings))
    if window is not None:
        resolve_service.record_hint(settings, ref.id, now=now)
    _print(answer, heading="Debug · latest recorded attempt")


def _confirm_during_resolve(window: WindowRecord, yes: bool, *, what: str) -> bool:
    """A hint or debug during a timed re-solve counts against it. Ask first, unless told yes."""
    if yes:
        return True
    if _ask(
        f"A {what.lower()} during the timed re-solve of {window.problem_id} makes it count as "
        "practice, not a timed pass. Continue?"
    ):
        return True
    _print_note(f"{what} not shown. The timed re-solve continues, with no hint used.")
    return False


def _ask(question: str) -> bool:
    """A learner's yes opens the walkthrough. No input at all (a pipe, a closed stdin) is a no."""
    try:
        return typer.confirm(question, default=False)
    except typer.Abort:
        return False


def _print_reviewed(hint: ReviewedHint) -> None:
    """The authored hint for the level, labelled as reviewed. Lower levels are named, not shown."""
    console = out()
    heading = f"Reviewed hint · L{hint.level} · {hint.label} · from hints.md, no model call"
    console.print(f"[bold]{escape(heading)}[/]")
    console.print(escape(hint.text), soft_wrap=True)
    if hint.level > 1:
        earlier = ", ".join(f"--level {number}" for number in range(1, hint.level))
        _print_note(f"Earlier hints: {earlier}.")
    console.print()


def _print_note(text: str) -> None:
    out().print(f"[dim]{escape(text)}[/]", soft_wrap=True)


def _print(answer: TutorAnswer, *, heading: str) -> None:
    console = out()
    console.print(f"[bold]{escape(heading)}[/]")
    console.print(escape(answer.text), soft_wrap=True)
    if answer.sources:
        console.print()
        console.print("[bold]Sources[/]")
        for source in answer.sources:
            console.print(escape(f"  [{source.tag}] {source.citation}"), soft_wrap=True)
    for note in answer.notes:
        console.print(f"[dim]{escape(note)}[/]", soft_wrap=True)
    seconds = answer.latency_ms / 1000
    footer = (
        f"{answer.model} · prompt {answer.mode.value} v{answer.prompt_version} · {seconds:.1f} s"
    )
    console.print(f"[dim]{escape(footer)}[/]", soft_wrap=True)
