"""``dogma warp`` and ``dogma course reset``: move the course calendar, or start the course over.

Every move prints what changes before anything is written, then asks, unless ``--yes``. A reset
explains what it keeps and clears, then needs the word ``reset`` typed out. The domain rules live
in ``dogma.learning.calendar``; this module only parses options and prints.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Annotated, Final

import typer
from rich.console import Console
from rich.markup import escape

from dogma.cli.context import CliError, out, settings_of
from dogma.learning import calendar
from dogma.learning.calendar import Anchor, WarpKind, WarpPreview, WarpRequest

RESET_WORD: Final = "reset"
CONFLICT: Final = "choose one of --to-week, --ahead, --back or --original, or give none to sync"


def warp_command(
    ctx: typer.Context,
    *,
    to_week: Annotated[
        int | None,
        typer.Option("--to-week", min=1, help="Make this week week N (clamped to the course)."),
    ] = None,
    ahead: Annotated[
        int | None,
        typer.Option(
            "--ahead",
            min=1,
            help="Jump N weeks ahead in the course: the week number goes up, dates move earlier.",
        ),
    ] = None,
    back: Annotated[
        int | None,
        typer.Option(
            "--back",
            min=1,
            help="Go N weeks back in the course: the week number goes down, dates move later.",
        ),
    ] = None,
    original: Annotated[
        bool,
        typer.Option("--original", help="Go back to the authored course dates."),
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would change and change nothing.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Move without asking.")] = False,
) -> None:
    """Move the course calendar. With no option, this week becomes your furthest exercise's week."""
    settings = settings_of(ctx)
    request = _request(to_week, ahead, back, original)
    now = datetime.now(UTC)
    console = out()
    result = calendar.preview(settings, request, now=now)
    _print_anchor(console, result)
    if not result.changed:
        console.print(f"This week is already week {result.before.current_week}. Nothing changed.")
        return
    _print_change(console, result)
    if dry_run:
        console.print("Dry run: nothing changed.")
        return
    calendar.require_recording(settings)
    if not yes and not typer.confirm(
        "Move the calendar? [y/N]", default=False, show_default=False, prompt_suffix=" "
    ):
        console.print("Nothing changed.")
        return
    applied = calendar.apply(settings, request, now=now)
    console.print(f"Calendar moved. This week is week {applied.after.current_week}.")


def reset_command(
    ctx: typer.Context,
    yes: Annotated[
        bool, typer.Option("--yes", help="Reset without typing the confirmation.")
    ] = False,
) -> None:
    """Start the course over: back up, clear all progress, and make this week week 1."""
    settings = settings_of(ctx)
    calendar.require_recording(settings)
    console = out()
    _print_reset_explanation(console)
    if not yes:
        answer = typer.prompt(
            f"Type {RESET_WORD} to start over (anything else cancels)",
            default="",
            show_default=False,
        )
        if answer.strip() != RESET_WORD:
            console.print("Cancelled. Nothing was cleared.")
            return
    result = calendar.reset_course(settings, now=datetime.now(UTC))
    console.print(f"Backup: {escape(str(result.backup_path))}")
    if result.snapshot_path is not None:
        console.print(f"Database snapshot: {escape(str(result.snapshot_path))}")
    cleared = sum(result.rows_cleared.values())
    console.print(f"Cleared {cleared} rows from {len(result.rows_cleared)} tables.")
    _print_set_aside(console, result)
    console.print(
        f"This week is week 1. Week 1 starts {result.week_one_starts:%a %d %b %Y}, "
        f"and the course ends {_day(result.course_end, year=True)}."
    )
    if result.snapshot_path is not None:
        console.print(
            "To restore the old progress, stop dogma serve and copy the snapshot over "
            f"{escape(str(settings.var_dir / 'dogma.db'))}."
        )


def _request(
    to_week: int | None, ahead: int | None, back: int | None, original: bool
) -> WarpRequest:
    chosen = [value is not None for value in (to_week, ahead, back)] + [original]
    if sum(chosen) > 1:
        raise CliError(CONFLICT)
    if to_week is not None:
        return WarpRequest(WarpKind.TO_WEEK, week=to_week)
    if ahead is not None:
        return WarpRequest(WarpKind.AHEAD, weeks=ahead)
    if back is not None:
        return WarpRequest(WarpKind.BACK, weeks=back)
    if original:
        return WarpRequest(WarpKind.ORIGINAL)
    return WarpRequest(WarpKind.SYNC)


def _print_anchor(console: Console, result: WarpPreview) -> None:
    """For a sync: say what set the target week, in the learner's terms."""
    if result.request.kind is not WarpKind.SYNC or result.anchor is None:
        return
    found: Anchor = result.anchor
    if found.problem_id is None:
        console.print("You haven't run an exercise yet, so this week becomes week 1.")
    else:
        console.print(
            f"Your furthest exercise is in week {found.week} ({escape(found.problem_id)})."
        )


def _print_change(console: Console, result: WarpPreview) -> None:
    before, after = result.before, result.after
    console.print(f"This week: week {before.current_week} → week {after.current_week}")
    console.print(f"Weekly mocks from: {_day(before.mocks_from)} → {_day(after.mocks_from)}")
    console.print(f"Interview week: {_day(before.interview_week)} → {_day(after.interview_week)}")
    console.print(f"Course ends: {_day(before.end, year=True)} → {_day(after.end, year=True)}")


def _print_reset_explanation(console: Console) -> None:
    console.print("Starting over clears all of your progress:")
    console.print(
        "  cleared: attempts, topic mastery, study sessions and their notes, review cards and "
        "reviews, timed re-solves, mock and interview answers"
    )
    console.print("  kept: the course content, the Library (corpus) and your run folders")
    console.print("  workspace files are moved to var/workspace-history/, not deleted")
    console.print("  this week becomes week 1, and a backup is written first")


def _print_set_aside(console: Console, result: calendar.ResetResult) -> None:
    if result.history_dir is None:
        console.print("No workspace files to set aside.")
    elif result.workspace_error is not None:
        console.print(
            f"Set aside {result.files_set_aside} workspace files to "
            f"{escape(str(result.history_dir))}, then stopped: {escape(result.workspace_error)}"
        )
    else:
        console.print(
            f"Set aside {result.files_set_aside} workspace files to "
            f"{escape(str(result.history_dir))}."
        )


def _day(value: date, *, year: bool = False) -> str:
    """A date as the learner reads it in the plan: ``Mon 09 Nov``, with the year when asked."""
    return value.strftime("%a %d %b %Y" if year else "%a %d %b")
