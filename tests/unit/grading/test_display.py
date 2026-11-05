"""Learner-safe wording: hidden tests show one line with no values, public tests show all."""

from __future__ import annotations

import pytest

from dogma.grading.display import (
    NO_DETAIL,
    describe_finding,
    findings_in_words,
    hidden_summary,
    learner_message,
    status_headline,
)
from dogma.grading.models import GradeReport, RunStatus, StaticFinding
from dogma.grading.models import TestResult as Result

SENTINEL = "SECRET_VALUE_4242"
HIDDEN_FAILURE = (
    "AssertionError: wrong sign convention\n"
    "Arrays are not almost equal to 7 decimals\n"
    f"x: array([{SENTINEL}, 2.0])\n"
    "ACTUAL: array([9.0])\n"
    "Max absolute difference: 4.0\n"
    f"Falsifying example: test_hidden(\n    a=array([{SENTINEL}]),\n)\n"
)


def _report(*tests: Result, status: RunStatus = RunStatus.FAILED) -> GradeReport:
    return GradeReport(
        attempt_id="t1",
        problem_id="cml/vector-operations/p01-vector-projection",
        problem_version=1,
        runtime="py-numpy",
        status=status,
        tests=tuple(tests),
    )


def test_hidden_summary_names_the_mistake_and_drops_every_value() -> None:
    summary = hidden_summary(HIDDEN_FAILURE)
    assert summary == "wrong sign convention"
    assert SENTINEL not in summary


@pytest.mark.parametrize(
    "message",
    [
        f"AssertionError: assert 42 == {SENTINEL}",
        f"AssertionError: ACTUAL: array([{SENTINEL}])",
        f"Failed: AssertionError: x: array([{SENTINEL}])",
        f"AssertionError: Falsifying example: {SENTINEL}\n{SENTINEL}",
    ],
    ids=["wrapped-assert", "wrapped-actual", "nested-wrappers", "wrapped-falsifier"],
)
def test_wrapped_hidden_values_are_filtered_after_normalizing(message: str) -> None:
    assert hidden_summary(message) == NO_DETAIL


def test_learner_message_for_a_failing_hidden_test_is_the_summary_only() -> None:
    test = Result(name="test_h", file="hidden", outcome="fail", message=HIDDEN_FAILURE)
    message = learner_message(test)
    assert message == "wrong sign convention"
    assert SENTINEL not in message


def test_learner_message_for_an_errored_hidden_test_is_also_summarised() -> None:
    test = Result(name="test_h", file="hidden", outcome="error", message=HIDDEN_FAILURE)
    assert learner_message(test) == "wrong sign convention"


def test_learner_message_for_a_passing_hidden_test_is_empty() -> None:
    test = Result(name="test_h", file="hidden", outcome="pass", message=HIDDEN_FAILURE)
    assert learner_message(test) == ""


def test_learner_message_for_a_public_test_keeps_the_full_text() -> None:
    message = "AssertionError: shape\nx: array([1.0, 2.0])"
    test = Result(name="test_p", file="public", outcome="fail", message=message)
    assert learner_message(test) == message


def test_a_hidden_message_with_no_detail_is_the_fixed_sentence() -> None:
    test = Result(name="test_h", file="hidden", outcome="fail", message="")
    assert learner_message(test) == NO_DETAIL


def test_status_headlines_cover_the_kills_and_nothing_else() -> None:
    assert status_headline(RunStatus.TIMEOUT) is not None
    assert status_headline(RunStatus.OOM) is not None
    assert status_headline(RunStatus.ERROR) is not None
    assert status_headline(RunStatus.PASSED) is None
    assert status_headline(RunStatus.FAILED) is None
    assert status_headline(RunStatus.REJECTED) is None


def test_findings_are_plain_words_in_report_order() -> None:
    report = GradeReport(
        attempt_id="t2",
        problem_id="cml/vector-operations/p01-vector-projection",
        problem_version=1,
        runtime="py-numpy",
        status=RunStatus.REJECTED,
        static=(
            StaticFinding(rule="banned-import", message="`numpy.linalg` is not allowed", line=3),
            StaticFinding(rule="syntax", message="invalid syntax (line 9)", line=9),
        ),
    )
    assert findings_in_words(report) == [
        "line 3 uses numpy.linalg, which this problem bans",
        "syntax error: invalid syntax (line 9)",
    ]


def test_describe_finding_falls_back_to_the_message_for_unknown_rules() -> None:
    finding = StaticFinding(rule="other", message="something plain")
    assert describe_finding(finding) == "something plain"
