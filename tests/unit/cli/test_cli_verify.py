"""``dogma content verify``: every rule, with a fake grader; ruff runs for real on the copy."""

from __future__ import annotations

import pytest
from tests.unit.cli.support import HIDDEN_PASS, PUBLIC_PASS, make_report, result

from dogma.grading.models import RunStatus, StaticFinding

TOPIC = "cml/vector-operations"


@pytest.fixture
def passing(fake_grader):
    """The solution passes everything and the starter fails one test: a verified problem."""
    fake_grader.script["solution.py"] = make_report(
        tests=PUBLIC_PASS + HIDDEN_PASS, status=RunStatus.PASSED
    )
    fake_grader.script["starter.py"] = make_report(
        status=RunStatus.FAILED,
        tests=(result("test_example", "public", "fail", "not implemented"),),
    )
    return fake_grader


def test_a_verified_topic_exits_zero(invoke, passing):
    outcome = invoke("content", "verify", TOPIC)

    assert outcome.exit_code == 0, outcome.output
    assert "verified" in outcome.output
    assert "All 1 topic(s) verified" in outcome.output


def test_the_solution_runs_twice_and_the_starter_once(invoke, passing):
    invoke("content", "verify", TOPIC)

    names = [call[0] for call in passing.calls]
    assert names == ["solution.py", "solution.py", "starter.py"]
    assert all(call[2] is True for call in passing.calls), "verify always includes hidden tests"


def test_a_starter_that_passes_fails_verification(invoke, fake_grader):
    fake_grader.script["solution.py"] = make_report(
        tests=PUBLIC_PASS + HIDDEN_PASS, status=RunStatus.PASSED
    )
    fake_grader.script["starter.py"] = make_report(
        tests=PUBLIC_PASS + HIDDEN_PASS, status=RunStatus.PASSED
    )

    outcome = invoke("content", "verify", TOPIC)

    assert outcome.exit_code == 1
    assert "must fail at least one test" in outcome.output
    assert "tests are too weak" in outcome.output


def test_a_solution_that_fails_a_test_is_named(invoke, fake_grader):
    fake_grader.script["solution.py"] = make_report(
        status=RunStatus.FAILED,
        tests=(*PUBLIC_PASS[:3], result("test_edge", "public", "fail", "boom"), *HIDDEN_PASS),
    )
    fake_grader.script["starter.py"] = make_report(status=RunStatus.FAILED, tests=PUBLIC_PASS)

    outcome = invoke("content", "verify", TOPIC)

    assert outcome.exit_code == 1
    assert "does not pass every test: public test_edge" in outcome.output


def test_non_deterministic_outcomes_fail(invoke, fake_grader):
    flaky_second = make_report(
        status=RunStatus.FAILED,
        tests=(*PUBLIC_PASS, result("test_hid_0", "hidden", "fail", "flaky"), *HIDDEN_PASS[1:]),
    )
    fake_grader.queue["solution.py"] = [
        make_report(tests=PUBLIC_PASS + HIDDEN_PASS, status=RunStatus.PASSED),
        flaky_second,
    ]
    fake_grader.script["starter.py"] = make_report(status=RunStatus.FAILED, tests=PUBLIC_PASS)

    outcome = invoke("content", "verify", TOPIC)

    assert outcome.exit_code == 1
    assert "outcomes differ between two runs" in outcome.output


def test_too_few_public_and_hidden_tests_are_reported(invoke, fake_grader):
    fake_grader.script["solution.py"] = make_report(
        tests=PUBLIC_PASS[:2] + HIDDEN_PASS[:4], status=RunStatus.PASSED
    )
    fake_grader.script["starter.py"] = make_report(
        status=RunStatus.FAILED, tests=(result("t", "public", "fail"),)
    )

    outcome = invoke("content", "verify", TOPIC)

    assert outcome.exit_code == 1
    assert "needs >= 3 public tests, has 2" in outcome.output
    assert "needs >= 5 hidden tests, has 4" in outcome.output


def test_a_solution_with_static_findings_is_rejected_in_plain_words(invoke, fake_grader):
    fake_grader.script["solution.py"] = make_report(
        status=RunStatus.REJECTED,
        static=(
            StaticFinding(
                rule="banned-import",
                message="`numpy.linalg` is not allowed in this problem (line 7)",
                line=7,
            ),
        ),
    )
    fake_grader.script["starter.py"] = make_report(status=RunStatus.FAILED, tests=PUBLIC_PASS)

    outcome = invoke("content", "verify", TOPIC)

    assert outcome.exit_code == 1
    assert "line 7 uses numpy.linalg, which this problem bans" in outcome.output


def test_every_topic_issue_is_printed_not_only_the_first(invoke, repo, passing):
    lesson = repo / "content" / "cml" / "vector-operations" / "lesson.md"
    lesson.write_text(lesson.read_text().replace("## Connections", "## Links"))
    (repo / "content/cml/vector-operations/problems/p01-vector-projection/hints.md").unlink()

    outcome = invoke("content", "verify", TOPIC)

    assert outcome.exit_code == 1
    assert "LESSON_SECTIONS" in outcome.output
    assert "PROBLEM_FILE_MISSING" in outcome.output


def test_a_single_problem_path_verifies_only_that_problem(invoke, passing):
    outcome = invoke("content", "verify", f"{TOPIC}/p01-vector-projection")

    assert outcome.exit_code == 0, outcome.output
    assert {call[1] for call in passing.calls} == {"p01-vector-projection"}


def test_all_and_paths_together_is_a_usage_error(invoke):
    outcome = invoke("content", "verify", "--all", TOPIC)

    assert outcome.exit_code == 2
    assert "not both" in outcome.output


def test_verify_without_arguments_is_a_usage_error(invoke):
    outcome = invoke("content", "verify")

    assert outcome.exit_code == 2


def test_all_checks_the_repository_too(invoke, passing):
    outcome = invoke("content", "verify", "--all")

    assert outcome.exit_code == 0, outcome.output
    assert "Repository issues" not in outcome.output
