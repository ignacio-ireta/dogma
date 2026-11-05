"""progress, backup and db upgrade through the CLI, and the attempt hook in run and submit.

The hook is checked against the real temporary database under the test repo: nothing here reads
or writes the real var/. Grading is faked (see conftest), so no sandbox runs.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from tests.unit.cli.support import HIDDEN_PASS, PROBLEM, PUBLIC_PASS, make_report, result

from dogma.cli.progress import summary_line
from dogma.content.loader import load_core_path
from dogma.core.config import Settings
from dogma.db.errors import StorageErrorCode
from dogma.db.migrate import head_revision
from dogma.db.models import Attempt, TopicState
from dogma.db.progress import AttemptRow, Snapshot
from dogma.db.session import open_session
from dogma.grading.models import RunStatus

SUBMISSION = "p01-vector-projection.py"  # the workspace copy's name, which the fake keys on
TOPIC = "cml/vector-operations"


def _text(output: str) -> str:
    return " ".join(output.split())


def _attempts(settings: Settings) -> list[dict[str, Any]]:
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        return [
            {
                "mode": row.mode,
                "status": row.status,
                "public": (row.public_passed, row.public_total),
                "hidden": (row.hidden_passed, row.hidden_total),
                "run_dir": row.run_dir,
            }
            for row in session.scalars(select(Attempt).order_by(Attempt.id))
        ]


def _mastery(settings: Settings, topic: str = TOPIC) -> str | None:
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        state = session.get(TopicState, topic)
        return None if state is None else state.mastery


def _at(month: int, day: int) -> datetime:
    """Noon on that day in this machine's zone, so the local date is exactly ``day``."""
    return datetime(2026, month, day, 12, 0).astimezone()


def _row(when: datetime) -> AttemptRow:
    return AttemptRow(
        problem_id=PROBLEM,
        mode="submit",
        status="passed",
        passed=4,
        total=4,
        submitted_at=when,
    )


def test_submit_records_one_attempt_with_public_and_hidden_counts(
    invoke, fake_grader, settings
) -> None:
    fake_grader.script[SUBMISSION] = make_report(tests=PUBLIC_PASS + HIDDEN_PASS)
    invoke("open", PROBLEM)

    outcome = invoke("submit", PROBLEM)

    assert outcome.exit_code == 0, outcome.output
    [row] = _attempts(settings)
    assert row["mode"] == "submit"
    assert row["status"] == "passed"
    assert row["public"] == (4, 4)
    assert row["hidden"] == (11, 11)
    assert row["run_dir"] == "var/runs/a1"


def test_run_records_a_run_with_no_hidden_tests(invoke, fake_grader, settings) -> None:
    fake_grader.script[SUBMISSION] = make_report(tests=PUBLIC_PASS)
    invoke("open", PROBLEM)

    outcome = invoke("run", PROBLEM)

    assert outcome.exit_code == 0, outcome.output
    [row] = _attempts(settings)
    assert row["mode"] == "run"
    assert row["hidden"] == (0, 0)


def test_a_passing_core_submit_solves_its_topic_and_progress_shows_it(
    invoke, fake_grader, settings
) -> None:
    fake_grader.script[SUBMISSION] = make_report(tests=PUBLIC_PASS + HIDDEN_PASS)
    invoke("open", PROBLEM)

    invoke("submit", PROBLEM)
    progress = invoke("progress", "--week", "1")

    assert _mastery(settings) == "solved"
    assert progress.exit_code == 0, progress.output
    assert "solved" in _text(progress.output)
    assert "Solved 1 of" in _text(progress.output)


def test_a_passing_run_does_not_solve_the_topic(invoke, fake_grader, settings) -> None:
    fake_grader.script[SUBMISSION] = make_report(tests=PUBLIC_PASS)
    invoke("open", PROBLEM)

    invoke("run", PROBLEM)

    assert _mastery(settings) is None


def test_a_passing_submit_of_a_non_core_problem_does_not_solve_the_topic(
    invoke, fake_grader, settings, problem_dir
) -> None:
    meta = problem_dir / "meta.yaml"
    meta.write_text(meta.read_text(encoding="utf-8").replace("core: true", "core: false"))
    fake_grader.script[SUBMISSION] = make_report(tests=PUBLIC_PASS + HIDDEN_PASS)
    invoke("open", PROBLEM)

    outcome = invoke("submit", PROBLEM)

    assert outcome.exit_code == 0, outcome.output
    assert _mastery(settings) is None
    assert len(_attempts(settings)) == 1


def test_a_failed_submit_is_recorded_and_the_exit_code_is_unchanged(
    invoke, fake_grader, settings
) -> None:
    tests = (*PUBLIC_PASS, *HIDDEN_PASS[:10], result("test_edge", "hidden", "fail", "x"))
    fake_grader.script[SUBMISSION] = make_report(RunStatus.FAILED, tests=tests)
    invoke("open", PROBLEM)

    outcome = invoke("submit", PROBLEM)

    assert outcome.exit_code == 1
    [row] = _attempts(settings)
    assert row["status"] == "failed"
    assert row["hidden"] == (10, 11)
    assert _mastery(settings) is None


def test_a_database_error_still_prints_the_grade_and_warns(invoke, fake_grader, settings) -> None:
    fake_grader.script[SUBMISSION] = make_report(tests=PUBLIC_PASS + HIDDEN_PASS)
    invoke("open", PROBLEM)
    (settings.repo_root / "var").write_text("a file where the folder should be")

    outcome = invoke("submit", PROBLEM)

    assert outcome.exit_code == 0, outcome.output
    text = _text(outcome.output)
    assert "public 4/4 · hidden 11/11 · passed" in text
    assert "warning [DB_UNAVAILABLE] attempt not recorded" in text
    assert "Traceback" not in outcome.output


def test_a_failing_grade_with_a_database_error_still_exits_1(invoke, fake_grader, settings) -> None:
    fake_grader.script[SUBMISSION] = make_report(RunStatus.FAILED, tests=PUBLIC_PASS[:1])
    invoke("open", PROBLEM)
    (settings.repo_root / "var").write_text("not a folder")

    outcome = invoke("submit", PROBLEM)

    assert outcome.exit_code == 1
    assert "not passed yet" in outcome.output
    assert "DB_UNAVAILABLE" in outcome.output


def test_a_corrupt_database_warns_with_a_migration_code(invoke, fake_grader, settings) -> None:
    fake_grader.script[SUBMISSION] = make_report(tests=PUBLIC_PASS)
    invoke("open", PROBLEM)
    settings.var_dir.mkdir(parents=True)
    (settings.var_dir / "dogma.db").write_bytes(b"definitely not sqlite" * 50)

    outcome = invoke("run", PROBLEM)

    assert outcome.exit_code == 0, outcome.output
    assert "public 4/4" in _text(outcome.output)
    assert "DB_MIGRATION_FAILED" in outcome.output


def test_progress_without_a_database_reads_as_not_started_and_creates_nothing(
    invoke, settings
) -> None:
    outcome = invoke("progress", "--week", "1")

    assert outcome.exit_code == 0, outcome.output
    assert "unseen" in outcome.output
    assert "Solved 0 of" in _text(outcome.output)
    assert not (settings.var_dir / "dogma.db").exists()


def test_progress_shows_attempts_and_the_best_result_for_the_topic(
    invoke, fake_grader, settings
) -> None:
    weak = (
        *PUBLIC_PASS[:2],
        result("test_b", "public", "fail"),
        result("test_c", "public", "fail"),
    )
    fake_grader.script[SUBMISSION] = make_report(RunStatus.FAILED, tests=weak)
    invoke("open", PROBLEM)
    invoke("run", PROBLEM)

    outcome = invoke("progress", "--week", "1")

    text = _text(outcome.output)
    assert outcome.exit_code == 0, outcome.output
    assert "2/4 · run" in text
    assert "unseen" in text
    assert "Solved 0 of" in text


def test_progress_without_a_week_flag_shows_every_week(invoke) -> None:
    outcome = invoke("progress")

    assert outcome.exit_code == 0, outcome.output
    assert "Week 1 " in outcome.output
    assert "Week 14 " in outcome.output


def test_progress_for_a_week_outside_the_path_is_a_usage_error(invoke) -> None:
    outcome = invoke("progress", "--week", "99")

    assert outcome.exit_code == 2
    assert "not 99" in outcome.output


def test_backup_writes_the_export_and_prints_where_it_went(invoke, fake_grader, settings) -> None:
    fake_grader.script[SUBMISSION] = make_report(tests=PUBLIC_PASS + HIDDEN_PASS)
    invoke("open", PROBLEM)
    invoke("submit", PROBLEM)

    outcome = invoke("backup")

    assert outcome.exit_code == 0, outcome.output
    [backup] = list((settings.var_dir / "backups").glob("dogma-*.json"))
    assert f"Backup written: {backup}" in " ".join(outcome.output.split())
    payload = json.loads(backup.read_text(encoding="utf-8"))
    assert len(payload["tables"]["attempt"]) == 1
    assert payload["tables"]["topic_state"][0]["mastery"] == "solved"


def test_a_backup_that_cannot_be_written_is_a_one_line_error(invoke, settings) -> None:
    settings.var_dir.mkdir(parents=True)
    (settings.var_dir / "backups").write_text("not a folder")

    outcome = invoke("backup")

    assert outcome.exit_code == 1
    assert "error [BACKUP_WRITE_FAILED]" in outcome.output
    assert "Traceback" not in outcome.output


def test_db_upgrade_creates_the_database_then_reports_it_current(invoke, settings) -> None:
    first = invoke("db", "upgrade")
    second = invoke("db", "upgrade")

    assert first.exit_code == 0, first.output
    assert "Created" in first.output
    assert (settings.var_dir / "dogma.db").is_file()
    assert second.exit_code == 0
    assert f"already at schema revision {head_revision()}" in second.output


def test_the_summary_before_the_path_counts_attempts_from_today(repo: Path) -> None:
    weeks = load_core_path(repo / "curriculum" / "paths" / "core.yaml").weeks

    line = summary_line(
        weeks, Snapshot(mastery={}, attempts=(_row(_at(10, 24)),)), date(2026, 10, 24)
    )

    assert "1 attempt in week 1 (from 24 Oct)" in line


def test_the_summary_counts_only_the_study_week_that_holds_today(repo: Path) -> None:
    weeks = load_core_path(repo / "curriculum" / "paths" / "core.yaml").weeks
    snapshot = Snapshot(mastery={}, attempts=(_row(_at(10, 24)), _row(_at(11, 3))))

    line = summary_line(weeks, snapshot, date(2026, 11, 3))

    assert "1 attempt in week 2 (from 02 Nov)" in line


def test_the_summary_counts_solved_topics_across_the_whole_path(repo: Path) -> None:
    weeks = load_core_path(repo / "curriculum" / "paths" / "core.yaml").weeks
    topics = {topic for item in weeks for topic in item.topics}
    snapshot = Snapshot(mastery={TOPIC: "solved"}, attempts=())

    line = summary_line(weeks, snapshot, date(2026, 10, 24))

    assert f"Solved 1 of {len(topics)} topics in the core path" in line


def _elsewhere(tmp_path: Path) -> Path:
    """A file graded with --submission (named like the workspace copy, which the fake keys on)."""
    other = tmp_path / "elsewhere" / SUBMISSION
    other.parent.mkdir(exist_ok=True)
    other.write_text("def vector_projection(a, b):\n    return a\n", encoding="utf-8")
    return other


def test_grading_another_file_is_not_recorded_by_default(
    invoke, fake_grader, settings, tmp_path
) -> None:
    """Authors mutation-test with --submission; that must never touch the learner's progress."""
    fake_grader.script[SUBMISSION] = make_report(tests=PUBLIC_PASS + HIDDEN_PASS)

    outcome = invoke("submit", PROBLEM, "--submission", str(_elsewhere(tmp_path)))

    assert outcome.exit_code == 0, outcome.output
    assert _attempts(settings) == []
    assert _mastery(settings) is None


def test_grading_another_file_with_record_is_recorded(
    invoke, fake_grader, settings, tmp_path
) -> None:
    fake_grader.script[SUBMISSION] = make_report(tests=PUBLIC_PASS + HIDDEN_PASS)

    invoke("submit", PROBLEM, "--submission", str(_elsewhere(tmp_path)), "--record")

    assert len(_attempts(settings)) == 1
    assert _mastery(settings) == "solved"


def test_workspace_submit_with_no_record_is_not_recorded(invoke, fake_grader, settings) -> None:
    fake_grader.script[SUBMISSION] = make_report(tests=PUBLIC_PASS + HIDDEN_PASS)
    invoke("open", PROBLEM)

    invoke("submit", PROBLEM, "--no-record")

    assert _attempts(settings) == []


def test_record_attempts_setting_switches_recording_off(invoke, fake_grader, settings) -> None:
    fake_grader.script[SUBMISSION] = make_report(tests=PUBLIC_PASS + HIDDEN_PASS)
    settings.record_attempts = False
    invoke("open", PROBLEM)

    invoke("submit", PROBLEM)

    assert _attempts(settings) == []
