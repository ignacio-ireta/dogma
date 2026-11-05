"""Typer CLI entry points."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from pydantic import ValidationError

from docmd import __version__
from docmd.cli_repos import repos_app
from docmd.config import AppConfig, OcrMode
from docmd.errors import DocmdError
from docmd.logging_config import get_logger, setup_logging

if TYPE_CHECKING:
    from docmd.pipeline.state import FileRecord, RunSummary
    from docmd.tracking.sync import SyncOutcome

app = typer.Typer(
    name="docmd",
    help="Extract DOCX/PPTX/XLSX/PDF content into normalized Markdown.",
    no_args_is_help=True,
)
app.add_typer(repos_app, name="repos")

EXIT_NOTHING_TO_DO = 4


@app.callback()
def _main() -> None:
    """Extract DOCX/PPTX/XLSX/PDF content into normalized Markdown."""


@app.command()
def version() -> None:
    """Print the docmd version."""
    typer.echo(f"docmd {__version__}")


def _build_config(**overrides: object) -> AppConfig:
    try:
        return AppConfig.from_cli(**overrides)
    except ValidationError as exc:
        typer.secho(f"invalid configuration: {exc}", err=True, fg="red")
        raise typer.Exit(2) from exc


def _same_input_root(recorded: str, source: Path) -> bool:
    """Whether a manifest's input_root names the same directory as ``source``.

    Plain string equality misses same-directory spellings that differ only in
    case on case-insensitive filesystems; ``samefile`` resolves those without
    ever equating genuinely different paths.
    """
    if recorded == str(source):
        return True
    try:
        return Path(recorded).samefile(source)
    except OSError:
        return False


def _check_ocr_available(config: AppConfig) -> None:
    """Fail fast (once, not per file) when OCR is requested but cannot run."""
    if config.ocr is OcrMode.OFF:
        return
    from docmd.ocr import is_ocr_available

    if not is_ocr_available():
        typer.secho(
            "OCR was requested but no backend is available; "
            "OCR requires macOS and the ocr extra (uv sync --extra ocr).",
            err=True,
            fg="red",
        )
        raise typer.Exit(2)


@app.command()
def scan(
    source: Annotated[Path, typer.Argument(exists=True, readable=True, resolve_path=True)],
    out: Annotated[Path, typer.Option("--out", help="Output directory for Markdown.")],
    recursive: Annotated[
        bool, typer.Option("--recursive", help="Descend into subdirectories.")
    ] = False,
    resume: Annotated[
        bool, typer.Option("--resume", help="Re-attempt only failed/interrupted files.")
    ] = False,
    fail_fast: Annotated[
        bool, typer.Option("--fail-fast", help="Stop the run on the first failure.")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Reprocess files even when output is current.")
    ] = False,
    workers: Annotated[
        int | None, typer.Option("--workers", min=1, max=32, help="Worker pool size.")
    ] = None,
    deterministic: Annotated[
        bool, typer.Option("--deterministic", help="Omit timestamps from outputs.")
    ] = False,
    ocr: Annotated[
        OcrMode | None,
        typer.Option(
            "--ocr",
            help="OCR scanned PDF pages: auto = pages without a text layer, force = all pages.",
        ),
    ] = None,
    ocr_lang: Annotated[
        str | None,
        typer.Option("--ocr-lang", help="Comma-separated OCR language codes (e.g. en-US,de-DE)."),
    ] = None,
    ocr_dpi: Annotated[
        int | None,
        typer.Option("--ocr-dpi", min=72, max=600, help="Page render resolution for OCR."),
    ] = None,
    log_level: Annotated[str | None, typer.Option("--log-level")] = None,
    progress: Annotated[
        bool, typer.Option("--progress/--no-progress", help="Show a progress bar.")
    ] = True,
) -> None:
    """Convert every supported file under SOURCE to Markdown."""
    from docmd.discovery.scanner import scan_with_skips
    from docmd.errors import DiscoveryError
    from docmd.pipeline.progress import NullProgress, RichProgress
    from docmd.pipeline.runner import CancellationToken, PipelineRunner, install_sigint_handler
    from docmd.pipeline.state import (
        FileStatus,
        Manifest,
        ManifestStore,
        allocate_run_dir,
        write_errors_json,
    )
    from docmd.pipeline.steps import steps_from_names

    config = _build_config(
        recursive=recursive or None,
        resume=resume or None,
        fail_fast=fail_fast or None,
        force=force or None,
        workers=workers,
        deterministic=deterministic or None,
        ocr=ocr,
        ocr_lang=ocr_lang,
        ocr_dpi=ocr_dpi,
        log_level=log_level,
    )
    _check_ocr_available(config)
    from datetime import UTC, datetime

    run_id, run_dir = allocate_run_dir(out / "runs")
    setup_logging(config.log_level, run_dir / "run.log")
    log = get_logger("cli")
    log.info("starting run run_id=%s input=%s output=%s", run_id, source, out)

    try:
        files, skipped = scan_with_skips(source, recursive=config.recursive, out_root=out)
    except DiscoveryError as err:
        typer.secho(f"ERROR: [{err.error_code}] {err.message}", err=True, fg="red")
        raise typer.Exit(err.exit_code) from err
    log.info(
        "discovered files total=%d supported=%d unsupported=%d",
        len(files) + len(skipped),
        len(files),
        len(skipped),
    )
    if not files:
        typer.secho("No supported files found.", err=True, fg="yellow")
        raise typer.Exit(EXIT_NOTHING_TO_DO)

    # Load the previous run BEFORE creating this run's manifest, or load_latest
    # would find the new (empty) manifest instead.
    previous = ManifestStore.load_latest(out / "runs") if config.resume else None
    if previous is not None and (
        not _same_input_root(previous.input_root, source)
        or (previous.command and previous.command[0] != "scan")
    ):
        # The runs dir is shared with `sync`; never resume against another
        # command's manifest or another source tree.
        log.info(
            "resume: ignoring latest manifest run=%s (different command or input root)",
            previous.run_id,
        )
        previous = None

    store = ManifestStore(
        Manifest(
            run_id=run_id,
            tool_version=__version__,
            command=["scan", str(source), "--out", str(out)],
            input_root=str(source),
            output_root=str(out),
            started_at=datetime.now(UTC),
        ),
        run_dir / "manifest.json",
    )

    if config.resume and previous is not None and previous.run_id != run_id:
        done = {
            rec.source
            for rec in previous.files.values()
            if rec.status in (FileStatus.SUCCESS, FileStatus.WARNING, FileStatus.SKIPPED)
        }
        resumed_skips = [f for f in files if f.relative_path in done]
        files = [f for f in files if f.relative_path not in done]
        from docmd.pipeline.state import FileRecord

        store.update_many(
            [
                FileRecord(
                    source=f.relative_path,
                    size=f.size,
                    status=FileStatus.SKIPPED,
                    skip_reason=f"resume: completed in run {previous.run_id}",
                )
                for f in resumed_skips
            ]
        )
        log.info(
            "resume: skipping %d completed file(s) from run %s",
            len(resumed_skips),
            previous.run_id,
        )

    token = CancellationToken()
    install_sigint_handler(token)
    reporter = RichProgress() if progress and sys.stderr.isatty() else NullProgress()
    runner = PipelineRunner(
        config=config,
        steps=steps_from_names(),
        store=store,
        out_root=out,
        progress=reporter,
        token=token,
    )
    summary = runner.run(files, skipped)
    write_errors_json(run_dir, store.records)
    store.finalize(summary)

    _print_summary(summary, store.records, run_dir)
    if summary.interrupted:
        typer.echo(
            f"\nResume with: docmd scan {source} --out {out}"
            f"{' --recursive' if config.recursive else ''} --resume"
        )
        raise typer.Exit(130)
    if summary.failed:
        raise typer.Exit(1)
    raise typer.Exit(0)


def _print_summary(summary: RunSummary, records: list[FileRecord], run_dir: Path) -> None:
    from docmd.pipeline.state import FileStatus

    typer.echo("")
    typer.secho(f"SUCCESS: {summary.success} files converted", fg="green")
    if summary.warning:
        typer.secho(f"WARNING: {summary.warning} files converted with warnings", fg="yellow")
    if summary.skipped:
        typer.echo(f"SKIPPED: {summary.skipped} files")
    if summary.failed:
        typer.secho(f"FAILED: {summary.failed} files failed", fg="red")
    if summary.interrupted:
        typer.secho(f"INTERRUPTED: {summary.interrupted} files not completed", fg="yellow")

    failed = [r for r in records if r.status is FileStatus.FAILED]
    if failed:
        typer.echo("\nFailed files:")
        for record in failed:
            reason = record.error.message if record.error else "unknown error"
            typer.echo(f"- {record.source}: {reason}")
    typer.echo(f"\nRun report: {run_dir}")


@app.command()
def sync(
    source: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, readable=True, resolve_path=True)
    ],
    out: Annotated[Path, typer.Option("--out", help="Output root of the tracked knowledge base.")],
    recursive: Annotated[
        bool, typer.Option("--recursive/--no-recursive", help="Descend into subdirectories.")
    ] = True,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Classify and report; change nothing.")
    ] = False,
    confirm_deletes: Annotated[
        bool,
        typer.Option(
            "--confirm-deletes",
            help="Archive deletions even when they exceed the mass-deletion safety threshold.",
        ),
    ] = False,
    archive_keep_runs: Annotated[
        int | None,
        typer.Option(
            "--archive-keep-runs",
            min=0,
            help="Keep only the newest N archived runs per repo (0 = keep forever).",
        ),
    ] = None,
    fail_fast: Annotated[
        bool, typer.Option("--fail-fast", help="Stop the run on the first failure.")
    ] = False,
    workers: Annotated[
        int | None, typer.Option("--workers", min=1, max=32, help="Worker pool size.")
    ] = None,
    deterministic: Annotated[
        bool, typer.Option("--deterministic", help="Omit timestamps from outputs.")
    ] = False,
    ocr: Annotated[
        OcrMode | None,
        typer.Option(
            "--ocr",
            help="OCR scanned PDF pages: auto = pages without a text layer, force = all pages.",
        ),
    ] = None,
    ocr_lang: Annotated[
        str | None,
        typer.Option("--ocr-lang", help="Comma-separated OCR language codes (e.g. en-US,de-DE)."),
    ] = None,
    ocr_dpi: Annotated[
        int | None,
        typer.Option("--ocr-dpi", min=72, max=600, help="Page render resolution for OCR."),
    ] = None,
    log_level: Annotated[str | None, typer.Option("--log-level")] = None,
    progress: Annotated[
        bool, typer.Option("--progress/--no-progress", help="Show a progress bar.")
    ] = True,
) -> None:
    """Convert a document repo with tracking: diff, archive, and registry.

    The repo is identified by its root folder name. Outputs land under
    <out>/<repo>/; superseded conversions are archived, moves are relocated
    without re-extraction, and every action is recorded in changes.json.
    Re-running the same sync resumes an interrupted one.
    """
    from docmd.tracking.sync import run_sync

    config = _build_config(
        # Sync defaults to recursive; pass the flag through explicitly instead
        # of the drop-if-unset idiom so --no-recursive works.
        recursive=recursive,
        confirm_deletes=confirm_deletes or None,
        archive_keep_runs=archive_keep_runs,
        fail_fast=fail_fast or None,
        workers=workers,
        deterministic=deterministic or None,
        ocr=ocr,
        ocr_lang=ocr_lang,
        ocr_dpi=ocr_dpi,
        log_level=log_level,
    )
    _check_ocr_available(config)

    try:
        outcome = run_sync(
            source=source,
            out_root=out,
            config=config,
            command=["sync", str(source), "--out", str(out)],
            dry_run=dry_run,
            show_progress=progress,
        )
    except DocmdError as err:
        typer.secho(f"ERROR: [{err.error_code}] {err.message}", err=True, fg="red")
        raise typer.Exit(err.exit_code) from err

    _print_sync_summary(outcome)
    raise typer.Exit(outcome.exit_code)


def _print_sync_summary(outcome: SyncOutcome) -> None:
    from docmd.tracking.report import ChangeStatus

    report = outcome.report
    if report is None:
        typer.secho("No supported files found.", err=True, fg="yellow")
        if outcome.known:
            typer.secho(
                "Known repo with an empty source tree: deletions were NOT processed.",
                err=True,
                fg="yellow",
            )
        if outcome.run_dir.is_dir():
            typer.echo(f"\nRun report: {outcome.run_dir}")
        return

    counts = report.counts
    typer.echo("")
    verb = "planned (dry run)" if report.dry_run else "applied"
    typer.echo(f"Sync {verb} for repo '{report.repo}':")
    typer.echo(
        f"  added: {counts.added}  updated: {counts.updated}  moved: {counts.moved}  "
        f"deleted: {counts.deleted}  unchanged: {counts.unchanged}"
    )
    if counts.failed:
        typer.secho(f"  FAILED: {counts.failed} action(s)", fg="red")
    if counts.interrupted:
        typer.secho(f"  INTERRUPTED: {counts.interrupted} action(s)", fg="yellow")
    for note in report.notes:
        typer.secho(f"  note: {note}", fg="yellow")

    failed = [c for c in report.changes if c.status is ChangeStatus.FAILED]
    if failed:
        typer.echo("\nFailed actions:")
        for change in failed:
            reason = change.error.message if change.error else "unknown error"
            typer.echo(f"- {change.action.value} {change.source}: {reason}")

    if outcome.summary is not None:
        _print_summary(outcome.summary, outcome.records, outcome.run_dir)
    elif report.dry_run:
        typer.echo("\nDry-run plan was not persisted.")
    else:
        typer.echo(f"\nRun report: {outcome.run_dir}")
    if outcome.exit_code == 130:
        typer.echo("\nInterrupted: re-run the same docmd sync command to continue.")


@app.command()
def process(
    file: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True, resolve_path=True)
    ],
    out: Annotated[Path, typer.Option("--out", help="Output directory for Markdown.")],
    steps: Annotated[
        list[str] | None,
        typer.Option("--steps", help="Subset of pipeline steps: extract normalize write."),
    ] = None,
    deterministic: Annotated[
        bool, typer.Option("--deterministic", help="Omit timestamps from outputs.")
    ] = False,
    ocr: Annotated[
        OcrMode | None,
        typer.Option(
            "--ocr",
            help="OCR scanned PDF pages: auto = pages without a text layer, force = all pages.",
        ),
    ] = None,
    ocr_lang: Annotated[
        str | None,
        typer.Option("--ocr-lang", help="Comma-separated OCR language codes (e.g. en-US,de-DE)."),
    ] = None,
    ocr_dpi: Annotated[
        int | None,
        typer.Option("--ocr-dpi", min=72, max=600, help="Page render resolution for OCR."),
    ] = None,
    log_level: Annotated[str | None, typer.Option("--log-level")] = None,
) -> None:
    """Convert a single file to Markdown."""
    from docmd.discovery.scanner import scan_with_skips
    from docmd.pipeline.steps import FileContext, run_steps, steps_from_names

    config = _build_config(
        deterministic=deterministic or None,
        ocr=ocr,
        ocr_lang=ocr_lang,
        ocr_dpi=ocr_dpi,
        log_level=log_level,
    )
    _check_ocr_available(config)
    setup_logging(config.log_level)
    log = get_logger("cli")

    try:
        step_list = steps_from_names(tuple(steps) if steps else ("extract", "normalize", "write"))
        discovered, skipped = scan_with_skips(file)
        if not discovered:
            reason = skipped[0].reason if skipped else "not a supported file"
            typer.secho(f"SKIPPED: {file}: {reason}", err=True, fg="yellow")
            raise typer.Exit(EXIT_NOTHING_TO_DO)
        ctx = FileContext(source=discovered[0], out_root=out, config=config)
        run_steps(ctx, step_list)
    except DocmdError as err:
        log.error("failed file path=%s code=%s reason=%s", file, err.error_code, err.message)
        typer.secho(f"FAILED: {file}: [{err.error_code}] {err.message}", err=True, fg="red")
        raise typer.Exit(1 if err.exit_code == 3 else err.exit_code) from err

    if ctx.result is not None:
        for warning in ctx.result.warnings:
            log.warning("extraction warning code=%s location=%s", warning.code, warning.location)
    if ctx.output_path is not None:
        typer.echo(f"OK: {file} -> {ctx.output_path}")
    else:
        typer.echo(f"OK: {file} (no write step; ran: {', '.join(ctx.steps_run)})")


if __name__ == "__main__":  # pragma: no cover
    app()
