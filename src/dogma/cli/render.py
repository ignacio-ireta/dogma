"""Plain-language terminal output for a GradeReport (ARCHITECTURE §4.3, ENGINEERING §4.5).

Public failures show the test's full message. Hidden failures show only ``hidden_summary``: the
first line that names the mistake, with no values, so the learner cannot read the hidden inputs
back out of the output. Every message is escaped, so Rich markup in a learner's output stays text.

The wording rules live in ``dogma.grading.display`` so the API applies the same ones.
"""

from __future__ import annotations

from rich.console import Console
from rich.markup import escape

from dogma.grading.display import (
    NO_DETAIL,
    describe_finding,
    hidden_summary,
    learner_message,
    status_headline,
)
from dogma.grading.models import GradeReport, RunStatus, TestResult

__all__ = ["NO_DETAIL", "describe_finding", "hidden_summary", "render_report"]


def render_report(console: Console, report: GradeReport, *, hidden: bool) -> None:
    """Print one attempt: a line per test, what stopped the run (if anything), and a verdict."""
    for test in report.tests:
        # A long test name is one unbreakable word: let the terminal wrap it, so Rich never moves
        # it below an otherwise empty "✗ public" line.
        console.print(_test_line(test), soft_wrap=True)
        detail = _failure_detail(test)
        if detail:
            console.print(detail)
    if report.status is RunStatus.REJECTED:
        _rejected_block(console, report)
    elif (headline := status_headline(report.status)) is not None:
        console.print(f"[bold red]{headline}.[/]")
        if report.message:
            console.print(f"  {escape(report.message)}")
    console.print(_verdict(report, hidden=hidden))


def _test_line(test: TestResult) -> str:
    mark = {"pass": "[green]✓[/]", "skip": "[yellow]-[/]"}.get(test.outcome, "[red]✗[/]")
    return f"  {mark} {test.file:<7} {escape(test.name)}"


def _failure_detail(test: TestResult) -> str:
    if test.outcome not in ("fail", "error"):
        return ""
    text = learner_message(test).strip() or NO_DETAIL
    return "\n".join(f"      [red]{escape(line)}[/]" for line in text.splitlines())


def _rejected_block(console: Console, report: GradeReport) -> None:
    console.print("[bold red]Rejected before running.[/] Fix these, then submit again:")
    for finding in report.static:
        console.print(f"  - {escape(describe_finding(finding))}")
    if report.message:
        console.print(f"  [dim]{escape(report.message)}[/]")


def _verdict(report: GradeReport, *, hidden: bool) -> str:
    if report.status is RunStatus.REJECTED:
        return "[bold yellow]not run[/]"
    public_passed, public_total = report.counts("public")
    parts = [f"public {public_passed}/{public_total}"]
    if hidden:
        hidden_passed, hidden_total = report.counts("hidden")
        parts.append(f"hidden {hidden_passed}/{hidden_total}")
    if report.passed:
        note = "" if hidden else " (run submit to include the hidden tests)"
        verdict = f"[bold green]passed[/]{note}"
    else:
        verdict = "[bold yellow]not passed yet[/]"
    return " · ".join([*parts, verdict])
