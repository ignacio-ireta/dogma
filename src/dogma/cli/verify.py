"""``dogma content verify``: the author's gate (CONTENT_SPEC §7, ENGINEERING_STANDARDS §5.2).

For every topic: the CONTENT_SPEC rules (all issues are printed). For every problem: the solution
passes every test, twice with identical outcomes; the starter fails at least one test; there are at
least 3 public and 5 hidden tests; the solution clears the static checks. Then ``ruff check`` on the
topic directory. Grading runs in a scratch directory, never under var/runs.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Protocol

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from dogma.cli.context import CliError, out, settings_of
from dogma.cli.render import describe_finding
from dogma.content.loader import ProblemPackage, discover_topic_dirs, load_syllabus, load_topic
from dogma.content.resolve import Target, resolve_target
from dogma.content.schemas import Syllabus
from dogma.content.validate import (
    HIDDEN_TESTS_MIN,
    PUBLIC_TESTS_MIN,
    Issue,
    validate_repo,
    validate_topic,
)
from dogma.core.config import Settings
from dogma.core.errors import ContentError, DogmaError
from dogma.grading.models import GradeReport, RunStatus
from dogma.runner import attempt as runner_attempt
from dogma.runner import load_meta


class Grader(Protocol):
    """The signature of ``runner.attempt.grade_attempt``; tests pass a fake with the same shape."""

    def __call__(
        self,
        problem_dir: Path,
        submission: Path,
        *,
        include_hidden: bool,
        settings: Settings,
    ) -> GradeReport: ...


@dataclass(frozen=True)
class ProblemVerdict:
    name: str
    reasons: tuple[str, ...]
    public: int = 0
    hidden: int = 0

    @property
    def ok(self) -> bool:
        return not self.reasons


@dataclass(frozen=True)
class TopicVerdict:
    topic_id: str
    issues: tuple[Issue, ...]
    problems: tuple[ProblemVerdict, ...]
    ruff_ok: bool
    ruff_output: str

    @property
    def ok(self) -> bool:
        return not self.issues and self.ruff_ok and all(p.ok for p in self.problems)


def verify_command(
    ctx: typer.Context,
    paths: Annotated[
        list[str] | None,
        typer.Argument(help="Topic or problem ids or directories, e.g. cml/vector-operations."),
    ] = None,
    all_topics: Annotated[
        bool, typer.Option("--all", help="Verify every topic, plus the repository-level checks.")
    ] = False,
) -> None:
    """Check topics and problems the way CI does. Exits 1 on any failure."""
    settings = settings_of(ctx)
    if all_topics and paths:
        raise CliError("give either --all or topic/problem paths, not both")
    if not all_topics and not paths:
        raise CliError("give topic or problem paths to verify, or --all")
    syllabus = load_syllabus(settings.syllabus_path)
    plan: list[tuple[Path, frozenset[Path] | None]]
    if all_topics:
        plan = [(topic_dir, None) for topic_dir in discover_topic_dirs(settings.content_dir)]
        repo_issues = validate_repo(settings)
    else:
        plan = _plan(settings, paths or [])
        repo_issues = []
    console = out()
    verdicts: list[TopicVerdict] = []
    for topic_dir, only in plan:
        verdict = verify_topic(topic_dir, settings, syllabus=syllabus, only=only)
        _print_topic(console, verdict)
        verdicts.append(verdict)
    # validate_repo already reports every topic's issues; show only what the topics did not.
    seen = {issue for verdict in verdicts for issue in verdict.issues}
    extra = [issue for issue in repo_issues if issue not in seen]
    _print_summary(console, verdicts, extra)
    if extra or not all(verdict.ok for verdict in verdicts):
        raise typer.Exit(1)


def verify_topic(
    topic_dir: Path,
    settings: Settings,
    *,
    syllabus: Syllabus,
    only: frozenset[Path] | None = None,
    grade: Grader | None = None,
) -> TopicVerdict:
    """Verify one topic. ``only`` limits the problem checks to those problem directories."""
    grader: Grader = grade if grade is not None else runner_attempt.grade_attempt
    try:
        package = load_topic(topic_dir)
    except ContentError as exc:
        issues: tuple[Issue, ...] = (Issue(exc.code.value, str(topic_dir), exc.message),)
        topic_id = "/".join(topic_dir.parts[-2:])
        problems: tuple[ProblemVerdict, ...] = ()
    else:
        issues = tuple(validate_topic(package, syllabus))
        topic_id = package.meta.id
        problems = _verify_problems(package.problems, settings, grader, only)
    ruff_ok, ruff_output = run_ruff(topic_dir, settings.repo_root)
    return TopicVerdict(topic_id, issues, problems, ruff_ok, ruff_output)


def verify_problem(problem_dir: Path, settings: Settings, grade: Grader) -> ProblemVerdict:
    """Run the problem's solution and starter in ``settings.runs_dir`` and apply the rules."""
    name = problem_dir.name
    try:
        meta = load_meta(problem_dir)
    except DogmaError as exc:
        return ProblemVerdict(name, (f"meta.yaml cannot be read: {exc.message}",))
    suffix = ".sql" if meta.language == "sql" else ".py"
    missing = [
        f"{stem}{suffix}"
        for stem in ("solution", "starter")
        if not (problem_dir / f"{stem}{suffix}").is_file()
    ]
    if missing:
        return ProblemVerdict(name, (f"missing {', '.join(missing)}",))
    try:
        return _check_problem(problem_dir, settings, grade, suffix)
    except DogmaError as exc:
        return ProblemVerdict(name, (f"the runner failed: [{exc.code.value}] {exc.message}",))


def run_ruff(topic_dir: Path, repo_root: Path) -> tuple[bool, str]:
    """``ruff check`` on one topic directory, with the repository's configuration.

    The repository's pyproject.toml is passed explicitly, so a user-level ruff config cannot change
    the verdict. Without one (a scratch copy), ruff runs isolated: its defaults only.
    """
    pyproject = repo_root / "pyproject.toml"
    scope = ["--config", str(pyproject)] if pyproject.is_file() else ["--isolated"]
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "ruff", "check", *scope, str(topic_dir)],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def _verify_problems(
    problems: tuple[ProblemPackage, ...],
    settings: Settings,
    grade: Grader,
    only: frozenset[Path] | None,
) -> tuple[ProblemVerdict, ...]:
    chosen = [problem for problem in problems if only is None or problem.dir in only]
    if not chosen:
        return ()
    # One scratch directory per topic, so nothing is written under the learner's var/runs.
    with tempfile.TemporaryDirectory(prefix="dogma-verify-") as scratch:
        scratch_settings = settings.model_copy(update={"repo_root": Path(scratch)})
        return tuple(verify_problem(problem.dir, scratch_settings, grade) for problem in chosen)


def _check_problem(
    problem_dir: Path, settings: Settings, grade: Grader, suffix: str
) -> ProblemVerdict:
    solution = problem_dir / f"solution{suffix}"
    starter = problem_dir / f"starter{suffix}"
    first = grade(problem_dir, solution, include_hidden=True, settings=settings)
    second = grade(problem_dir, solution, include_hidden=True, settings=settings)
    reasons: list[str] = []
    if first.status is RunStatus.REJECTED:
        findings = "; ".join(describe_finding(finding) for finding in first.static)
        reasons.append(f"the solution is rejected by the static checks: {findings}")
    elif first.status is not RunStatus.PASSED:
        reasons.append(_solution_failure(first))
    if _outcomes(first) != _outcomes(second):
        reasons.append("outcomes differ between two runs (non-deterministic tests)")
    public, hidden = first.counts("public")[1], first.counts("hidden")[1]
    if first.tests:
        if public < PUBLIC_TESTS_MIN:
            reasons.append(f"needs >= {PUBLIC_TESTS_MIN} public tests, has {public}")
        if hidden < HIDDEN_TESTS_MIN:
            reasons.append(f"needs >= {HIDDEN_TESTS_MIN} hidden tests, has {hidden}")
    starter_report = grade(problem_dir, starter, include_hidden=True, settings=settings)
    if not any(test.outcome in ("fail", "error") for test in starter_report.tests):
        reasons.append(
            f"the starter must fail at least one test, but it {_starter_state(starter_report)}"
        )
    return ProblemVerdict(problem_dir.name, tuple(reasons), public, hidden)


def _outcomes(report: GradeReport) -> tuple[RunStatus, tuple[tuple[str, str, str], ...]]:
    return report.status, tuple((test.file, test.name, test.outcome) for test in report.tests)


def _solution_failure(report: GradeReport) -> str:
    failing = [
        f"{test.file} {test.name}" for test in report.tests if test.outcome in ("fail", "error")
    ]
    if failing:
        return "the solution does not pass every test: " + ", ".join(failing)
    first_line = next((line for line in report.message.splitlines() if line.strip()), "")
    detail = f" ({first_line.strip()})" if first_line else ""
    return f"the solution ends with status {report.status.value}{detail}"


def _starter_state(report: GradeReport) -> str:
    if report.status is RunStatus.REJECTED:
        return "was rejected by the static checks"
    if report.status is RunStatus.PASSED:
        return "passes every test, so the tests are too weak"
    return f"ends with status {report.status.value} and no failing test"


def _plan(settings: Settings, refs: list[str]) -> list[tuple[Path, frozenset[Path] | None]]:
    """Group the requested targets by topic. A whole-topic request wins over problem requests."""
    chosen: dict[Path, set[Path] | None] = {}
    for ref in refs:
        target: Target = resolve_target(settings, ref)
        if target.problem_dir is None:
            chosen[target.topic_dir] = None
            continue
        existing = chosen.get(target.topic_dir, set())
        if existing is not None:
            existing.add(target.problem_dir)
            chosen[target.topic_dir] = existing
    return [
        (topic, None if problems is None else frozenset(problems))
        for topic, problems in chosen.items()
    ]


def _print_topic(console: Console, verdict: TopicVerdict) -> None:
    mark = "[bold green]✓[/]" if verdict.ok else "[bold red]✗[/]"
    console.print(f"{mark} [bold]{escape(verdict.topic_id)}[/]")
    for issue in verdict.issues:
        console.print(
            f"    [red]{escape(issue.code)}[/] {escape(issue.path)}: {escape(issue.message)}"
        )
    for problem in verdict.problems:
        if problem.ok:
            console.print(
                f"    [green]✓[/] {escape(problem.name)}  public {problem.public} · "
                f"hidden {problem.hidden} · repeatable · starter fails"
            )
            continue
        console.print(f"    [red]✗[/] {escape(problem.name)}")
        for reason in problem.reasons:
            console.print(f"        {escape(reason)}")
    if verdict.ruff_ok:
        console.print("    [green]✓[/] ruff check")
    else:
        console.print("    [red]✗ ruff check[/]")
        console.print(escape(verdict.ruff_output[:3000]))


def _print_summary(console: Console, verdicts: list[TopicVerdict], extra: list[Issue]) -> None:
    if extra:
        console.print("[bold red]Repository issues[/] (syllabus, learning paths, prerequisites):")
        for issue in extra:
            console.print(
                f"    [red]{escape(issue.code)}[/] {escape(issue.path)}: {escape(issue.message)}"
            )
    table = Table(title="Verification summary", title_justify="left")
    for name in ("Topic", "Issues", "Problems", "Ruff", "Result"):
        table.add_column(name, justify="right" if name in ("Issues", "Problems") else "left")
    for verdict in verdicts:
        passed = sum(problem.ok for problem in verdict.problems)
        table.add_row(
            escape(verdict.topic_id),
            str(len(verdict.issues)),
            f"{passed}/{len(verdict.problems)}",
            "ok" if verdict.ruff_ok else "fail",
            "[green]verified[/]" if verdict.ok else "[red]failed[/]",
        )
    console.print(table)
    failed = sum(not verdict.ok for verdict in verdicts)
    if failed or extra:
        console.print(
            f"[bold red]{failed} of {len(verdicts)} topic(s) failed"
            + (f"; {len(extra)} repository issue(s)" if extra else "")
            + "[/]"
        )
    else:
        console.print(f"[bold green]All {len(verdicts)} topic(s) verified.[/]")
