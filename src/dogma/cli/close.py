"""``dogma close``: where you stopped, the next step, a blocker and a 60-second answer.

Closing happens once. A second ``dogma close`` says the session is already closed and keeps the
stored note. The note is shown first the next time you run ``dogma today``. Missing fields are
asked for interactively; flags answer them for scripts. Nothing you write here is graded, and a
close never marks a topic solved.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape

from dogma.cli.context import CliError, out, settings_of
from dogma.cli.today import KIND_LABEL
from dogma.learning import sessions
from dogma.learning.sessions import CloseNote, CloseTarget, TransitionNote


def close_command(
    ctx: typer.Context,
    stopped_where: Annotated[
        str | None,
        typer.Option("--stopped-where", help="Where you stopped, e.g. 'Practice, example 2'."),
    ] = None,
    next_step: Annotated[
        str | None,
        typer.Option("--next-step", help="The exact next step, for when you start again."),
    ] = None,
    blocker: Annotated[
        str | None,
        typer.Option("--blocker", help="An open blocker, if there is one."),
    ] = None,
    answer: Annotated[
        str | None,
        typer.Option(
            "--answer",
            help="Your 60-second answer, in your own words. Stored as written; never graded.",
        ),
    ] = None,
) -> None:
    """Close today's session with where you stopped, the next step and your answer."""
    settings = settings_of(ctx)
    now = datetime.now(UTC)
    target = sessions.close_target(settings, now=now)
    console = out()
    if target.stored is not None:
        _print_already_closed(console, target.stored)
        return
    stopped = (
        _ask(console, "Where did you stop?", "--stopped-where")
        if stopped_where is None
        else stopped_where
    )
    step = _ask(console, "The exact next step", "--next-step") if next_step is None else next_step
    blocker_text = (
        _ask_optional(console, "Any blocker (Enter if none)") if blocker is None else blocker
    )
    answer_text = _ask_optional(console, _answer_prompt(target)) if answer is None else answer
    result = sessions.close_session(
        settings,
        target.session_id,
        CloseNote(stopped_where=stopped, next_step=step, blocker=blocker_text or None),
        answer=answer_text or None,
        now=now,
    )
    console.print(
        f"Closed {KIND_LABEL[result.note.kind]} session {result.note.slot} of "
        f"{result.note.plan_date:%a %d %b %Y}.",
        soft_wrap=True,
    )
    console.print(f"Next time you start: {escape(result.note.next_step)}", soft_wrap=True)
    if result.introduced_topic is not None:
        console.print(f"Introduced {escape(result.introduced_topic)}.", soft_wrap=True)
    if result.new_cards:
        console.print(f"Added {result.new_cards} new review cards.")


def _answer_prompt(target: CloseTarget) -> str:
    if target.topic_id is None:
        return "In your own words, what did you review or re-solve? (Enter to skip)"
    return f"Explain {target.topic_id} in 60 seconds, in your own words (Enter to skip)"


def _print_already_closed(console: Console, note: TransitionNote) -> None:
    when = note.closed_at.astimezone() if note.closed_at is not None else None
    stamp = f" at {when:%H:%M}" if when is not None else ""
    console.print(
        f"Already closed{stamp}. The stored note was kept, nothing was changed.",
        soft_wrap=True,
    )
    console.print(f"Next time you start: {escape(note.next_step)}", soft_wrap=True)


def _ask(console: Console, message: str, flag: str) -> str:
    """A required answer: asked interactively, or it is a usage error without a terminal."""
    try:
        text = console.input(f"{message}: ").strip()
    except EOFError:
        raise CliError(f"{message} is required; pass {flag} when there is no terminal.") from None
    except KeyboardInterrupt:
        raise CliError("cancelled; nothing was saved.") from None
    if not text:
        raise CliError(f"{message} cannot be empty.")
    return text


def _ask_optional(console: Console, message: str) -> str:
    """An optional answer: Enter skips it, and so does no terminal."""
    try:
        return console.input(f"{message}: ").strip()
    except EOFError:
        return ""
    except KeyboardInterrupt:
        raise CliError("cancelled; nothing was saved.") from None
