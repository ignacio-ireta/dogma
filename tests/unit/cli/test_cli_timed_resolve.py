"""dogma resolve, and how submit, hint and progress talk about a timed re-solve.

The clock is fixed through a fixture: each command module reads ``datetime.now``, so the fixture
replaces that name in the modules that matter. The grader is the scripted FakeGrader from conftest.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from tests.unit.cli.support import HIDDEN_PASS, PROBLEM, PUBLIC_PASS, make_report

import dogma.cli.practice as practice_module
import dogma.cli.progress as progress_module
import dogma.cli.resolve_cmd as resolve_module
import dogma.cli.tutor as tutor_module
from dogma.cli import app
from dogma.core.config import Settings
from dogma.grading.models import GradeReport, RunStatus
from dogma.tutor.gateway import Gateway
from dogma.tutor.providers.fake import FakeProvider

WORKSPACE = "workspace/cml/vector-operations/p01-vector-projection.py"
START = datetime(2026, 10, 24, 9, 0, tzinfo=UTC)
PASSING = make_report(RunStatus.PASSED, tests=PUBLIC_PASS + HIDDEN_PASS)
FAILING = make_report(RunStatus.FAILED, tests=PUBLIC_PASS)
Run = Callable[..., object]


def fresh(report: GradeReport) -> GradeReport:
    """A copy with its own attempt id: every recorded submit needs a new one."""
    return report.model_copy(update={"attempt_id": uuid4().hex})


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, datetime]]:
    """The current instant every command reads. Tests move it with ``clock["now"] = ...``."""
    state = {"now": START}

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> datetime:  # type: ignore[override]
            value = state["now"]
            return value.astimezone(tz) if tz is not None else value.replace(tzinfo=None)  # type: ignore[arg-type]

    for module in (resolve_module, practice_module, tutor_module, progress_module):
        monkeypatch.setattr(module, "datetime", FixedDatetime)
    yield state


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> FakeProvider:
    provider = FakeProvider([])
    monkeypatch.setattr(tutor_module, "make_gateway", lambda s: Gateway(s, provider))
    return provider


@pytest.fixture
def run(cli, settings: Settings) -> Run:  # type: ignore[no-untyped-def]
    def _run(*args: str, input: str | None = None) -> object:
        return cli.invoke(app, list(args), obj=settings, env={"COLUMNS": "200"}, input=input)

    return _run


def _text(output: str) -> str:
    return " ".join(output.split())


def _pass_once(run: Run, fake_grader, clock) -> None:  # type: ignore[no-untyped-def]
    """Open the problem, pass it, so the learner has a first pass to re-solve from."""
    fake_grader.script["p01-vector-projection.py"] = fresh(PASSING)
    run("open", PROBLEM)
    assert run("submit", PROBLEM).exit_code == 0


def _start(run: Run, clock) -> object:  # type: ignore[no-untyped-def]
    clock["now"] = clock["now"] + timedelta(days=7)
    return run("resolve", PROBLEM)


def test_starting_before_a_solve_is_refused(run: Run, clock) -> None:  # type: ignore[no-untyped-def]
    outcome = run("resolve", PROBLEM)

    assert outcome.exit_code == 1
    assert "error [RESOLVE_NOT_SOLVED]" in outcome.output


def test_starting_after_a_pass_opens_the_window_and_resets_the_workspace(
    run: Run, clock, fake_grader, settings: Settings, problem_dir: Path
) -> None:  # type: ignore[no-untyped-def]
    _pass_once(run, fake_grader, clock)
    dest = settings.repo_root / WORKSPACE
    dest.write_text("def vector_projection(a, b):\n    return a\n")

    outcome = _start(run, clock)

    assert outcome.exit_code == 0, outcome.output
    text = _text(outcome.output)
    assert f"Timed re-solve open: {PROBLEM}" in text
    assert "budget 20 min" in text
    assert "no hints" in text
    assert "first submit decides" in text
    assert dest.read_text() == (problem_dir / "starter.py").read_text()
    kept = list((settings.var_dir / "workspace-history").rglob("p01-vector-projection.*.py"))
    assert [path.read_text() for path in kept] == ["def vector_projection(a, b):\n    return a\n"]


def test_a_second_start_names_the_open_one(run: Run, clock, fake_grader) -> None:  # type: ignore[no-untyped-def]
    _pass_once(run, fake_grader, clock)
    _start(run, clock)

    outcome = run("resolve", PROBLEM)

    assert outcome.exit_code == 1
    assert "error [RESOLVE_ALREADY_OPEN]" in outcome.output
    assert PROBLEM in outcome.output
    assert f"dogma resolve {PROBLEM} --abandon" in _text(outcome.output)


def test_a_pass_inside_the_budget_closes_the_window_as_passed(run: Run, clock, fake_grader) -> None:  # type: ignore[no-untyped-def]
    _pass_once(run, fake_grader, clock)
    _start(run, clock)
    clock["now"] += timedelta(minutes=10)
    fake_grader.script["p01-vector-projection.py"] = fresh(PASSING)

    outcome = run("submit", PROBLEM)

    assert outcome.exit_code == 0, outcome.output
    text = _text(outcome.output)
    assert "Timed re-solve passed: 10 min 00 s of 20 min, no hints." in text
    assert "Next timed re-solve:" in text
    assert "stage 1 of 3" in text
    assert "dogma resolve" not in _text(run("resolve", "--status").output)


def test_a_pass_after_the_budget_is_partial_and_says_why(run: Run, clock, fake_grader) -> None:  # type: ignore[no-untyped-def]
    _pass_once(run, fake_grader, clock)
    _start(run, clock)
    clock["now"] += timedelta(minutes=25)
    fake_grader.script["p01-vector-projection.py"] = fresh(PASSING)

    outcome = run("submit", PROBLEM)

    assert outcome.exit_code == 0, outcome.output
    assert "Timed re-solve partial: over budget (25 min 00 s of 20 min)." in _text(outcome.output)


def test_a_hint_during_the_window_is_confirmed_and_makes_the_pass_partial(
    run: Run, clock, fake_grader, model: FakeProvider
) -> None:  # type: ignore[no-untyped-def]
    _pass_once(run, fake_grader, clock)
    _start(run, clock)
    model.queue("A nudge [S1].")
    clock["now"] += timedelta(minutes=2)

    hinted = run("hint", PROBLEM, "--level", "1", input="y\n")
    fake_grader.script["p01-vector-projection.py"] = fresh(PASSING)
    submitted = run("submit", PROBLEM)

    assert hinted.exit_code == 0, hinted.output
    assert "Reviewed hint" in hinted.output
    assert submitted.exit_code == 0, submitted.output
    assert "Timed re-solve partial: hints used (1), so it counts as practice." in _text(
        submitted.output
    )


def test_declining_the_hint_shows_nothing_and_counts_nothing(
    run: Run, clock, fake_grader, model: FakeProvider
) -> None:  # type: ignore[no-untyped-def]
    _pass_once(run, fake_grader, clock)
    _start(run, clock)

    declined = run("hint", PROBLEM, "--level", "1", input="n\n")
    fake_grader.script["p01-vector-projection.py"] = fresh(PASSING)
    submitted = run("submit", PROBLEM)

    assert "Hint not shown" in _text(declined.output)
    assert "Reviewed hint" not in declined.output
    assert "Timed re-solve passed" in _text(submitted.output)


def test_yes_skips_the_hint_prompt_and_still_counts(
    run: Run, clock, fake_grader, model: FakeProvider
) -> None:  # type: ignore[no-untyped-def]
    _pass_once(run, fake_grader, clock)
    _start(run, clock)
    model.queue("A nudge [S1].")

    hinted = run("hint", PROBLEM, "--level", "1", "--yes")

    assert hinted.exit_code == 0, hinted.output
    assert "Reviewed hint" in hinted.output
    assert "Hint not shown" not in hinted.output


def test_a_failed_submit_closes_the_window_as_failed(run: Run, clock, fake_grader) -> None:  # type: ignore[no-untyped-def]
    _pass_once(run, fake_grader, clock)
    _start(run, clock)
    fake_grader.script["p01-vector-projection.py"] = fresh(FAILING)

    outcome = run("submit", PROBLEM)

    assert outcome.exit_code == 1
    assert "Timed re-solve failed: not every test passed." in _text(outcome.output)
    assert "stage 0 of 3" in _text(outcome.output)


def test_run_never_closes_the_window(run: Run, clock, fake_grader) -> None:  # type: ignore[no-untyped-def]
    _pass_once(run, fake_grader, clock)
    _start(run, clock)
    fake_grader.script["p01-vector-projection.py"] = fresh(PASSING)

    run("run", PROBLEM)
    status = run("resolve", "--status")

    assert "Open timed re-solve" in _text(status.output)


def test_grading_another_file_never_touches_the_window(
    run: Run, clock, fake_grader, settings: Settings, problem_dir: Path
) -> None:  # type: ignore[no-untyped-def]
    _pass_once(run, fake_grader, clock)
    _start(run, clock)
    fake_grader.script["solution.py"] = fresh(PASSING)

    run("submit", PROBLEM, "--submission", str(problem_dir / "solution.py"))

    assert "Open timed re-solve" in _text(run("resolve", "--status").output)


def test_switching_recording_off_refuses_to_start(
    run: Run, clock, fake_grader, settings: Settings
) -> None:  # type: ignore[no-untyped-def]
    _pass_once(run, fake_grader, clock)
    settings.record_attempts = False

    outcome = _start(run, clock)

    assert outcome.exit_code == 1
    assert "error [RESOLVE_RECORDING_OFF]" in outcome.output


def test_a_problem_that_is_not_core_cannot_be_timed(run: Run, problem_dir: Path) -> None:
    meta = problem_dir / "meta.yaml"
    meta.write_text(meta.read_text().replace("core: true", "core: false"))

    outcome = run("resolve", PROBLEM)

    assert outcome.exit_code == 1
    assert "error [RESOLVE_NOT_CORE]" in outcome.output


def test_a_draft_problem_cannot_be_timed(run: Run, problem_dir: Path) -> None:
    meta = problem_dir / "meta.yaml"
    meta.write_text(meta.read_text().replace("status: reviewed", "status: draft"))

    outcome = run("resolve", PROBLEM)

    assert outcome.exit_code == 1
    assert "error [RESOLVE_NOT_READY]" in outcome.output


def test_status_and_abandon_say_what_happened_and_abandoning_is_free(
    run: Run, clock, fake_grader
) -> None:  # type: ignore[no-untyped-def]
    assert "No timed re-solve is open." in run("resolve", "--status").output
    _pass_once(run, fake_grader, clock)
    _start(run, clock)

    abandoned = run("resolve", PROBLEM, "--abandon")
    restarted = run("resolve", PROBLEM)

    assert abandoned.exit_code == 0, abandoned.output
    assert "Abandoned the timed re-solve" in _text(abandoned.output)
    assert "No penalty" in _text(abandoned.output)
    assert restarted.exit_code == 0, restarted.output


def test_resolve_needs_a_problem_or_a_flag(run: Run) -> None:
    outcome = run("resolve")

    assert outcome.exit_code == 2
    assert "name a problem" in outcome.output


def test_progress_shows_the_next_re_solve_after_a_pass_and_empty_states_before(
    run: Run, clock, fake_grader
) -> None:  # type: ignore[no-untyped-def]
    before = _text(run("progress").output)
    _pass_once(run, fake_grader, clock)

    after = _text(run("progress").output)

    assert "Re-solves" in before
    assert "Next due: none yet" in before
    assert "Last timed results: none yet" in before
    assert "Next due:" in after and PROBLEM in after
    assert "stage 0 of 3" in after
    assert "Last timed results: none yet" in after


def test_progress_lists_the_last_timed_result(run: Run, clock, fake_grader) -> None:  # type: ignore[no-untyped-def]
    _pass_once(run, fake_grader, clock)
    _start(run, clock)
    clock["now"] += timedelta(minutes=12, seconds=5)
    fake_grader.script["p01-vector-projection.py"] = fresh(PASSING)
    run("submit", PROBLEM)

    text = _text(run("progress").output)

    assert f"{PROBLEM} · passed · 12 min 05 s of 20 min" in text
