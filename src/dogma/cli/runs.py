"""``dogma runs``: explicit maintenance of graded run directories (ARCHITECTURE §4.2, §4.9).

Dry run by default. Recorded attempts and every run the database references are always kept.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Annotated

import typer
from rich.markup import escape

from dogma.cli.context import friendly_errors, out, settings_of
from dogma.core.config import Settings
from dogma.db.run_artifacts import protected_runs
from dogma.runner.retention import prune_runs

DEFAULT_AGE_DAYS = 30

app = typer.Typer(
    help="Maintain graded run directories: list or remove old, completed, unrecorded ones.",
    no_args_is_help=True,
    add_completion=False,
)


def _protection(settings: Settings) -> tuple[frozenset[str], frozenset[Path]]:
    """Ids and paths the database still references. Read before any directory is listed."""
    ids, stored = protected_runs(settings)
    # A malformed stored path still protects its attempt id and its final name.
    names = {PurePosixPath(text.replace("\\", "/")).name for text in stored}
    paths: set[Path] = set()
    for text in stored:
        if "\x00" in text:
            continue
        candidate = Path(text)
        paths.add(candidate if candidate.is_absolute() else settings.repo_root / candidate)
    return frozenset(ids | names), frozenset(paths)


def _shown(path: Path, settings: Settings) -> str:
    try:
        return path.relative_to(settings.repo_root).as_posix()
    except ValueError:
        return path.as_posix()


@app.command("prune", epilog="Example: dogma runs prune --older-than-days 30  (dry run)")
@friendly_errors
def prune_command(
    ctx: typer.Context,
    older_than_days: Annotated[
        int,
        typer.Option(
            "--older-than-days",
            min=1,
            help="Only runs that finished at least this many days ago (minimum 1).",
        ),
    ] = DEFAULT_AGE_DAYS,
    apply: Annotated[
        bool,
        typer.Option("--apply", help="Delete the candidates. Without it, nothing is deleted."),
    ] = False,
) -> None:
    """List, and with --apply delete, completed run directories older than the given age.

    Recorded attempts are never removed. Unmarked directories (active, interrupted or legacy) are
    kept. Without --apply this is a dry run.
    """
    settings = settings_of(ctx)
    before = datetime.now(UTC) - timedelta(days=older_than_days)
    protected_ids, protected_paths = _protection(settings)
    paths = prune_runs(
        settings.runs_dir,
        protected_ids=protected_ids,
        protected_paths=protected_paths,
        before=before,
        apply=apply,
    )
    _report(settings, paths, older_than_days=older_than_days, apply=apply)


def _report(
    settings: Settings, paths: tuple[Path, ...], *, older_than_days: int, apply: bool
) -> None:
    console = out()
    noun = "run directory" if len(paths) == 1 else "run directories"
    if not paths:
        console.print(
            f"No run directories are eligible (finished more than {older_than_days} days ago,"
            " not recorded).",
            soft_wrap=True,
        )
        return
    if apply:
        console.print(f"Removed {len(paths)} {noun}:", soft_wrap=True)
    else:
        console.print(
            f"Dry run: {len(paths)} {noun} would be removed. Nothing was deleted."
            " Re-run with --apply to delete them.",
            soft_wrap=True,
        )
    for path in paths:
        console.print(f"  {escape(_shown(path, settings))}", soft_wrap=True)
