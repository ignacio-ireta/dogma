"""`docmd repos` subcommands: inspect tracked repos, their state, and history.

Read-only views over what sync already persists — the registry, per-repo
snapshots, and each real run's ``changes.json``. History answers "what went
where": every add/update/move/delete with the archive location of any displaced
conversion. Dry-run plans are computed in memory and are never persisted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from docmd.errors import DocmdError

repos_app = typer.Typer(
    name="repos",
    help="Inspect tracked document repos: registry, snapshots, change history.",
    no_args_is_help=True,
)

_OUT_OPTION = typer.Option("--out", help="Output root of the tracked knowledge base.")


def _fail(message: str, exit_code: int = 1) -> typer.Exit:
    typer.secho(f"ERROR: {message}", err=True, fg="red")
    return typer.Exit(exit_code)


@repos_app.command("list")
def list_repos(out: Annotated[Path, _OUT_OPTION]) -> None:
    """List every repo tracked under the output root."""
    from docmd.tracking.registry import load_registry

    try:
        registry = load_registry(out)
    except DocmdError as err:
        raise _fail(f"[{err.error_code}] {err.message}", err.exit_code) from err
    if not registry.repos:
        typer.echo(f"No repos tracked under {out}.")
        return
    typer.echo(f"{'REPO':<24} {'FILES':>6} {'RUNS':>5} {'LAST SYNCED':<21} SOURCE")
    for name in sorted(registry.repos):
        entry = registry.repos[name]
        last = entry.last_synced.strftime("%Y-%m-%d %H:%M:%SZ")
        typer.echo(
            f"{entry.name:<24} {entry.file_count:>6} {entry.run_count:>5} "
            f"{last:<21} {entry.source_root}"
        )


@repos_app.command()
def show(
    name: Annotated[str, typer.Argument(help="Repo name (source root folder name).")],
    out: Annotated[Path, _OUT_OPTION],
    files: Annotated[
        bool, typer.Option("--files", help="List every tracked file with hash and output.")
    ] = False,
) -> None:
    """Show one repo's registry entry and tracked-file state."""
    from docmd.tracking.registry import load_registry
    from docmd.tracking.snapshot import load_snapshot

    try:
        registry = load_registry(out)
        entry = registry.repos.get(name)
        if entry is None:
            known = ", ".join(sorted(registry.repos)) or "none"
            raise _fail(f"unknown repo {name!r}; tracked repos: {known}")
        snapshot = load_snapshot(out, name)
    except DocmdError as err:
        raise _fail(f"[{err.error_code}] {err.message}", err.exit_code) from err

    typer.echo(f"repo:         {entry.name}")
    typer.echo(f"source root:  {entry.source_root}")
    typer.echo(f"output root:  {out / entry.name}")
    typer.echo(f"first synced: {entry.first_synced.isoformat(timespec='seconds')}")
    typer.echo(f"last synced:  {entry.last_synced.isoformat(timespec='seconds')}")
    typer.echo(f"last run:     {entry.last_run_id}")
    typer.echo(f"runs:         {entry.run_count}")
    typer.echo(f"files:        {entry.file_count}")
    if entry.last_counts is not None:
        counts = entry.last_counts
        typer.echo(
            f"last changes: added {counts.added}, updated {counts.updated}, "
            f"moved {counts.moved}, deleted {counts.deleted}, "
            f"unchanged {counts.unchanged}, failed {counts.failed}"
        )
    if files:
        if snapshot is None or not snapshot.files:
            typer.echo("\nNo tracked files (no completed sync yet).")
            return
        typer.echo(f"\n{'SHA256':<12} {'SOURCE':<48} OUTPUT")
        for path in sorted(snapshot.files):
            state = snapshot.files[path]
            typer.echo(f"{state.sha256[:10]:<12} {path:<48} {state.output}")


@repos_app.command()
def history(
    name: Annotated[str, typer.Argument(help="Repo name (source root folder name).")],
    out: Annotated[Path, _OUT_OPTION],
    path: Annotated[
        str | None,
        typer.Option("--path", help="Only actions touching this source relpath (old or new)."),
    ] = None,
    run: Annotated[str | None, typer.Option("--run", help="Only actions from this run id.")] = None,
    limit: Annotated[
        int, typer.Option("--limit", min=1, help="Maximum number of actions to print.")
    ] = 50,
) -> None:
    """Show what went where: per-run actions, newest first, with archive paths."""
    from docmd.tracking.registry import load_registry
    from docmd.tracking.report import CHANGES_NAME, ChangeStatus, SyncReport

    try:
        registry = load_registry(out)
    except DocmdError as err:
        raise _fail(f"[{err.error_code}] {err.message}", err.exit_code) from err
    if name not in registry.repos:
        known = ", ".join(sorted(registry.repos)) or "none"
        raise _fail(f"unknown repo {name!r}; tracked repos: {known}")

    runs_dir = out / "runs"
    run_dirs = (
        sorted((p for p in runs_dir.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True)
        if runs_dir.is_dir()
        else []
    )
    printed = 0
    for run_dir in run_dirs:
        if run is not None and run_dir.name != run:
            continue
        changes_file = run_dir / CHANGES_NAME
        if not changes_file.is_file():
            continue
        try:
            report = SyncReport.model_validate_json(changes_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            typer.secho(f"note: skipping unreadable report {changes_file}", err=True, fg="yellow")
            continue
        if report.repo != name or report.dry_run:
            continue
        for change in report.changes:
            if path is not None and path not in (change.source, change.previous_source):
                continue
            parts = [f"{report.run_id}  {change.action.value:<8} {change.source}"]
            if change.previous_source is not None:
                parts.append(f"(from {change.previous_source})")
            if change.archived_to is not None:
                label = f"archived_to={change.archived_to}"
                if not (out / change.archived_to).exists():
                    label += " (pruned)"
                parts.append(label)
            if change.status is not ChangeStatus.COMPLETED:
                parts.append(f"[{change.status.value}]")
            typer.echo("  ".join(parts))
            printed += 1
            if printed >= limit:
                typer.echo(f"... limited to {limit} action(s); raise with --limit")
                return
    if printed == 0:
        typer.echo("No recorded actions match.")
