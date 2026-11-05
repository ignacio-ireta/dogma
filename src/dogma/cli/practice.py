"""open, run and submit: the learner's practice loop (ARCHITECTURE §4.7, CONTENT_SPEC §7).

``open`` copies the starter into ``workspace/`` and never overwrites the learner's file; ``open
--fresh`` first keeps a copy under ``var/workspace-history/``. ``run`` grades that file against the
public tests; ``submit`` adds the hidden tests. Both exit 0 only when every test they ran passed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from dogma.cli import attempts
from dogma.cli.context import CliError, out, settings_of
from dogma.cli.render import render_report
from dogma.cli.resolve_cmd import render_closed
from dogma.content.resolve import resolve_problem, set_aside_workspace, workspace_file
from dogma.core.errors import ContentError, ErrorCode
from dogma.core.fsio import atomic_write
from dogma.db.models import AttemptMode
from dogma.runner import attempt as runner_attempt

ProblemArg = Annotated[
    str,
    typer.Argument(
        help="Problem id, e.g. cml/vector-operations/p01-vector-projection, or its path."
    ),
]
SubmissionOpt = Annotated[
    Path | None,
    typer.Option("--submission", help="Grade this file instead of your workspace copy."),
]
FreshOpt = Annotated[
    bool,
    typer.Option(
        "--fresh",
        help="Start over from the starter. Your current file is kept under "
        "var/workspace-history/, never deleted.",
    ),
]
RecordOpt = Annotated[
    bool | None,
    typer.Option(
        "--record/--no-record",
        help="Count this attempt in your progress. Default: only your workspace file counts, "
        "so grading another file with --submission (e.g. while authoring) is not recorded.",
    ),
]


def open_command(ctx: typer.Context, problem: ProblemArg, fresh: FreshOpt = False) -> None:
    """Copy the starter into workspace/ so you can edit it. Never overwrites your file."""
    settings = settings_of(ctx)
    ref = resolve_problem(settings, problem)
    dest = workspace_file(settings, ref)
    console = out()
    if dest.exists() and not fresh:
        console.print(f"Already open, not overwritten: {dest}", soft_wrap=True)
        console.print(f"To start over:     dogma open {ref.id} --fresh", soft_wrap=True)
    else:
        if not ref.starter.is_file():
            raise ContentError(ErrorCode.CONTENT_NOT_FOUND, f"{ref.starter} is missing")
        kept = set_aside_workspace(settings, ref, at=datetime.now(UTC))
        if kept is not None:
            console.print(f"Previous file kept: {kept}", soft_wrap=True)
        atomic_write(dest, ref.starter.read_text(encoding="utf-8"))
        console.print(f"Starter copied to: {dest}", soft_wrap=True)
    console.print(f"Read the task:     {ref.problem_md}", soft_wrap=True)
    console.print(f"{ref.meta.title} · {ref.meta.difficulty} · about {ref.meta.minutes} min")
    console.print(f"When you are ready: dogma run {ref.id}")


def run_command(
    ctx: typer.Context,
    problem: ProblemArg,
    submission: SubmissionOpt = None,
    record: RecordOpt = None,
) -> None:
    """Run your file against the public tests. No hidden tests yet."""
    _grade(ctx, problem, submission, include_hidden=False, record=record)


def submit_command(
    ctx: typer.Context,
    problem: ProblemArg,
    submission: SubmissionOpt = None,
    record: RecordOpt = None,
) -> None:
    """Run your file against the public and the hidden tests. Exits 0 only when all pass."""
    _grade(ctx, problem, submission, include_hidden=True, record=record)


def _grade(
    ctx: typer.Context,
    problem: str,
    submission: Path | None,
    *,
    include_hidden: bool,
    record: bool | None = None,
) -> None:
    settings = settings_of(ctx)
    ref = resolve_problem(settings, problem)
    source = submission if submission is not None else workspace_file(settings, ref)
    if not source.is_file():
        if submission is not None:
            raise CliError(f"no file at {source}")
        raise CliError(f"no submission at {source}; run `dogma open {ref.id}` first")
    started_at = datetime.now(UTC)
    report = runner_attempt.grade_attempt(
        ref.dir, source, include_hidden=include_hidden, settings=settings
    )
    submitted_at = datetime.now(UTC)
    render_report(out(), report, hidden=include_hidden)
    # Progress tracks the learner's own workspace attempts. Grading some other file (authors
    # mutation-testing, reference solutions) must never mark topics solved unless asked.
    should_record = (submission is None) if record is None else record
    if should_record and settings.record_attempts:
        # After the report: a database problem is a warning and never changes the grade.
        closed = attempts.record(
            settings,
            report,
            mode=AttemptMode.SUBMIT if include_hidden else AttemptMode.RUN,
            core=ref.meta.core,
            started_at=started_at,
            submitted_at=submitted_at,
        )
        if closed is not None:
            render_closed(out(), closed, submitted_at)
    if not report.passed:
        raise typer.Exit(1)
