"""open, run and submit: workspace copies, exit codes, and how each GradeReport is rendered."""

from __future__ import annotations

from pathlib import Path

from tests.unit.cli.support import HIDDEN_PASS, PROBLEM, PUBLIC_PASS, make_report, result

from dogma.cli.render import NO_DETAIL
from dogma.grading.models import RunStatus, StaticFinding

WORKSPACE_FILE = "workspace/cml/vector-operations/p01-vector-projection.py"


def _text(output: str) -> str:
    return " ".join(output.split())


def test_open_copies_the_starter_into_the_workspace(invoke, settings, problem_dir):
    outcome = invoke("open", PROBLEM)

    dest = settings.repo_root / WORKSPACE_FILE
    assert outcome.exit_code == 0, outcome.output
    assert dest.read_text() == (problem_dir / "starter.py").read_text()
    assert str(dest) in outcome.output
    assert str(problem_dir / "problem.md") in outcome.output


def test_open_never_overwrites_the_learners_file(invoke, settings):
    invoke("open", PROBLEM)
    dest = settings.repo_root / WORKSPACE_FILE
    dest.write_text("def vector_projection(a, b):\n    return a\n")

    outcome = invoke("open", PROBLEM)

    assert outcome.exit_code == 0
    assert "not overwritten" in outcome.output
    assert dest.read_text() == "def vector_projection(a, b):\n    return a\n"


def test_open_says_how_to_start_over_when_the_file_exists(invoke):
    invoke("open", PROBLEM)

    outcome = invoke("open", PROBLEM)

    assert f"dogma open {PROBLEM} --fresh" in _text(outcome.output)


def test_open_fresh_restores_the_starter_and_keeps_the_previous_file(invoke, settings, problem_dir):
    invoke("open", PROBLEM)
    dest = settings.repo_root / WORKSPACE_FILE
    dest.write_text("def vector_projection(a, b):\n    return a\n")

    outcome = invoke("open", PROBLEM, "--fresh")

    assert outcome.exit_code == 0, outcome.output
    assert dest.read_text() == (problem_dir / "starter.py").read_text()
    history = settings.var_dir / "workspace-history" / "cml" / "vector-operations"
    kept = list(history.glob("p01-vector-projection.*.py"))
    assert len(kept) == 1
    assert kept[0].read_text() == "def vector_projection(a, b):\n    return a\n"
    assert "Previous file kept" in outcome.output


def test_open_fresh_with_nothing_open_just_copies_the_starter(invoke, settings):
    outcome = invoke("open", PROBLEM, "--fresh")

    assert outcome.exit_code == 0, outcome.output
    assert (settings.repo_root / WORKSPACE_FILE).is_file()
    assert "Previous file kept" not in outcome.output
    assert not (settings.var_dir / "workspace-history").exists()


def test_open_accepts_a_problem_directory_too(invoke, settings, problem_dir):
    outcome = invoke("open", str(problem_dir))

    assert outcome.exit_code == 0, outcome.output
    assert (settings.repo_root / WORKSPACE_FILE).is_file()


def test_open_an_unknown_problem_is_a_friendly_one_line_error(invoke):
    outcome = invoke("open", "cml/vector-operations/p09-nope")

    assert outcome.exit_code == 1
    assert "error [CONTENT_NOT_FOUND]" in outcome.output
    assert "Traceback" not in outcome.output


def test_run_grades_public_tests_only_and_passes(invoke, fake_grader):
    fake_grader.script["p01-vector-projection.py"] = make_report(
        tests=PUBLIC_PASS, status=RunStatus.PASSED
    )
    invoke("open", PROBLEM)

    outcome = invoke("run", PROBLEM)

    assert outcome.exit_code == 0, outcome.output
    assert fake_grader.calls[-1][2] is False, "run must not include hidden tests"
    assert "public 4/4" in outcome.output
    assert "passed" in outcome.output
    assert "run submit to include the hidden tests" in outcome.output


def test_submit_passes_only_when_public_and_hidden_all_pass(invoke, fake_grader):
    fake_grader.script["p01-vector-projection.py"] = make_report(
        tests=PUBLIC_PASS + HIDDEN_PASS, status=RunStatus.PASSED
    )
    invoke("open", PROBLEM)

    outcome = invoke("submit", PROBLEM)

    assert outcome.exit_code == 0, outcome.output
    assert fake_grader.calls[-1][2] is True
    assert "public 4/4 · hidden 11/11 · passed" in _text(outcome.output)


def test_submit_shows_hidden_failures_as_a_summary_without_values(invoke, fake_grader):
    message = (
        "shape mismatch: the output has the wrong shape\n"
        "Arrays are not almost equal to 1e-09 tolerance\n"
        "Mismatched elements: 1 / 2 (50%)\n"
        "Max absolute difference among violations: 3.0\n"
        " x: array([1.0, 2.0])\n"
        " y: array([7.0, 9.0])\n"
        "Falsifying example: test_matches(a=array([123.4, 5.6]))\n"
        "    a=array([123.4, 5.6])\n"
    )
    hidden = (
        *HIDDEN_PASS[:10],
        result("test_matches_reference", "hidden", "fail", message),
    )
    fake_grader.script["p01-vector-projection.py"] = make_report(
        status=RunStatus.FAILED, tests=PUBLIC_PASS + hidden
    )
    invoke("open", PROBLEM)

    outcome = invoke("submit", PROBLEM)

    assert outcome.exit_code == 1
    assert "shape mismatch: the output has the wrong shape" in outcome.output
    for leaked in ("array([1.0, 2.0])", "123.4", "Max absolute", "Falsifying", "x: array"):
        assert leaked not in outcome.output
    assert "hidden 10/11" in outcome.output
    assert "not passed yet" in outcome.output


def test_hidden_failure_without_a_message_still_says_something(invoke, fake_grader):
    fake_grader.script["p01-vector-projection.py"] = make_report(
        status=RunStatus.FAILED,
        tests=(*PUBLIC_PASS, result("test_edge", "hidden", "fail", "")),
    )
    invoke("open", PROBLEM)

    outcome = invoke("submit", PROBLEM)

    assert NO_DETAIL in outcome.output


def test_public_failures_show_the_full_message(invoke, fake_grader):
    message = "shape mismatch: expected (3,)\nthe second line is kept in full"
    fake_grader.script["p01-vector-projection.py"] = make_report(
        status=RunStatus.FAILED,
        tests=(result("test_example", "public", "fail", message), *PUBLIC_PASS[1:]),
    )
    invoke("open", PROBLEM)

    outcome = invoke("run", PROBLEM)

    assert outcome.exit_code == 1
    assert "shape mismatch: expected (3,)" in outcome.output
    assert "the second line is kept in full" in outcome.output


def test_rejected_attempts_name_each_finding_in_plain_words(invoke, fake_grader):
    findings = (
        StaticFinding(
            rule="banned-import", message="`numpy.linalg` is not allowed (line 3)", line=3
        ),
        StaticFinding(
            rule="missing-entry",
            message="define `vector_projection` as a top-level function or class",
        ),
    )
    fake_grader.script["p01-vector-projection.py"] = make_report(
        status=RunStatus.REJECTED,
        static=findings,
        message="rejected before running: fix the findings listed",
    )
    invoke("open", PROBLEM)

    outcome = invoke("submit", PROBLEM)

    assert outcome.exit_code == 1
    assert "line 3 uses numpy.linalg, which this problem bans" in outcome.output
    assert "the file must define `vector_projection`" in outcome.output
    assert "Rejected before running" in outcome.output
    assert "not run" in outcome.output


def test_timeout_shows_the_grade_report_message(invoke, fake_grader):
    fake_grader.script["p01-vector-projection.py"] = make_report(
        status=RunStatus.TIMEOUT, message="the run exceeded its 30 s time limit"
    )
    invoke("open", PROBLEM)

    outcome = invoke("submit", PROBLEM)

    assert outcome.exit_code == 1
    assert "Timed out" in outcome.output
    assert "the run exceeded its 30 s time limit" in outcome.output


def test_oom_shows_the_grade_report_message(invoke, fake_grader):
    fake_grader.script["p01-vector-projection.py"] = make_report(
        status=RunStatus.OOM, message="the run exceeded its 2048 MB memory limit"
    )
    invoke("open", PROBLEM)

    outcome = invoke("submit", PROBLEM)

    assert outcome.exit_code == 1
    assert "Out of memory" in outcome.output
    assert "2048 MB memory limit" in outcome.output


def test_error_shows_why_the_tests_could_not_run(invoke, fake_grader):
    fake_grader.script["p01-vector-projection.py"] = make_report(
        status=RunStatus.ERROR, message="the test harness exited without a report"
    )
    invoke("open", PROBLEM)

    outcome = invoke("submit", PROBLEM)

    assert outcome.exit_code == 1
    assert "Could not run the tests" in outcome.output
    assert "exited without a report" in outcome.output


def test_learner_text_is_escaped_not_read_as_markup(invoke, fake_grader):
    fake_grader.script["p01-vector-projection.py"] = make_report(
        status=RunStatus.FAILED,
        tests=(result("test_[bold]tag", "public", "fail", "got [red]boom[/red] here"),),
    )
    invoke("open", PROBLEM)

    outcome = invoke("run", PROBLEM)

    assert "got [red]boom[/red] here" in outcome.output
    assert "test_[bold]tag" in outcome.output


def test_submit_without_a_workspace_file_asks_for_open(invoke, fake_grader):
    outcome = invoke("submit", PROBLEM)

    assert outcome.exit_code == 2
    assert "run `dogma open" in outcome.output
    assert not fake_grader.calls


def test_submission_option_grades_the_given_file(invoke, fake_grader, tmp_path: Path):
    other = tmp_path / "mine.py"
    other.write_text("def vector_projection(a, b):\n    return a\n")
    fake_grader.script["mine.py"] = make_report(tests=PUBLIC_PASS, status=RunStatus.PASSED)

    outcome = invoke("run", PROBLEM, "--submission", str(other))

    assert outcome.exit_code == 0, outcome.output
    assert fake_grader.calls[-1][0] == "mine.py"
