"""Plain-language rendering: hidden summaries leak no values; findings read as sentences."""

from __future__ import annotations

import io

from rich.console import Console

from dogma.cli.render import NO_DETAIL, describe_finding, hidden_summary, render_report
from dogma.grading.models import GradeReport, RunStatus, StaticFinding, TestResult


def test_hidden_summary_keeps_the_first_line_that_names_the_mistake():
    message = (
        "AssertionError: wrong sign convention\n"
        "Arrays are not equal\n"
        "Mismatched elements: 2 / 3\n"
        "x: array([1., 2., 3.])\n"
    )
    assert hidden_summary(message) == "wrong sign convention"


def test_hidden_summary_drops_hypothesis_inputs_after_the_falsifying_marker():
    message = "Falsifying example: test_property(\n    a=array([9.5, -2.25]),\n)\n"
    assert hidden_summary(message) == NO_DETAIL


def test_hidden_summary_never_returns_a_value_line():
    message = (
        "Arrays are not almost equal\nx: array([5.5])\n  y: array([1.25])\n"
        "Max absolute difference: 4.0"
    )
    assert hidden_summary(message) == NO_DETAIL


def test_hidden_summary_of_an_empty_message_is_a_fixed_sentence():
    assert hidden_summary("") == NO_DETAIL


def test_banned_import_finding_names_the_line_and_module():
    finding = StaticFinding(
        rule="banned-import",
        message="`numpy.linalg` is not allowed in this problem (line 3)",
        line=3,
    )
    assert describe_finding(finding) == "line 3 uses numpy.linalg, which this problem bans"


def test_syntax_and_entry_findings_read_as_sentences():
    syntax = StaticFinding(rule="syntax", message="invalid syntax (line 2)", line=2)
    entry = StaticFinding(
        rule="missing-entry", message="define `vector_projection` as a top-level function or class"
    )
    assert describe_finding(syntax) == "syntax error: invalid syntax (line 2)"
    assert describe_finding(entry) == (
        "the file must define `vector_projection` as a top-level function or class"
    )


def test_a_long_test_name_stays_on_its_result_line_in_a_narrow_terminal():
    name = "test_" + "a_deliberately_long_descriptive_behaviour_name_" * 2
    report = GradeReport(
        attempt_id="a1",
        problem_id="cml/x/p01-y",
        problem_version=1,
        runtime="py-stdlib",
        status=RunStatus.FAILED,
        tests=(TestResult(name=name, file="public", outcome="fail", message="wrong"),),
    )
    buffer = io.StringIO()
    render_report(Console(file=buffer, width=40, color_system=None), report, hidden=False)
    result_line = next(line for line in buffer.getvalue().splitlines() if "✗" in line)
    assert name[:20] in result_line, "the name was pushed off the line that shows the result"
