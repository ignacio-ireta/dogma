"""``dogma warp`` and ``dogma course reset``: preview, confirm, decline, conflicts and reset.

The clock is fixed to Tuesday 3 Nov 2026 (authored week 2 of the real core path). The repo is the
scratch copy from conftest.py, so nothing here touches a real var/ or workspace/.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select
from tests.unit.db.support import make_report
from tests.unit.learning.support import utc_timezone  # noqa: F401  autouse: pins the zone to UTC
from typer.testing import CliRunner

from dogma.cli import app, warp_cmd
from dogma.core.config import Settings
from dogma.db.calendar_models import CalendarWarp
from dogma.db.errors import StorageErrorCode
from dogma.db.models import Attempt, AttemptMode
from dogma.db.session import database_path, open_session
from dogma.db.store import record_attempt
from dogma.grading.models import RunStatus

NOW = datetime(2026, 11, 3, 12, tzinfo=UTC)
WIDE = {"COLUMNS": "200"}
PROBLEM = "cml/vector-operations/p01-vector-projection"
WEEK_TWO_PROBLEM = "cml/derivatives/p01-chain-rule-warmup"


class FixedDatetime(datetime):
    @classmethod
    def now(cls, tz: object = None) -> datetime:  # type: ignore[override]
        return NOW.astimezone(tz) if tz is not None else NOW.replace(tzinfo=None)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(warp_cmd, "datetime", FixedDatetime)


def _run(cli: CliRunner, settings: Settings, *args: str, input: str | None = None) -> object:
    return cli.invoke(app, list(args), obj=settings, input=input, env=WIDE)


def _words(output: str) -> str:
    return " ".join(output.split())


def _warps(settings: Settings) -> list[tuple[str, int]]:
    if not database_path(settings).is_file():
        return []
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as db:
        rows = db.scalars(select(CalendarWarp).order_by(CalendarWarp.id)).all()
        return [(row.kind, row.offset_weeks) for row in rows]


def _attempts(settings: Settings) -> int:
    if not database_path(settings).is_file():
        return 0
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as db:
        return db.scalar(select(func.count()).select_from(Attempt)) or 0


def _run_attempt(settings: Settings, problem_id: str) -> None:
    record_attempt(
        settings,
        make_report(RunStatus.PASSED, public=(3, 3), problem_id=problem_id),
        mode=AttemptMode.RUN,
        core=False,
        started_at=NOW,
        submitted_at=NOW,
    )


def test_a_dry_run_prints_the_change_and_changes_nothing(
    cli: CliRunner, settings: Settings
) -> None:
    outcome = _run(cli, settings, "warp", "--dry-run")

    assert outcome.exit_code == 0, outcome.output
    text = _words(outcome.output)
    assert "You haven't run an exercise yet, so this week becomes week 1." in text
    assert "This week: week 2 → week 1" in text
    assert "Weekly mocks from: Mon 09 Nov → Mon 16 Nov" in text
    assert "Course ends: Sun 31 Jan 2027 → Sun 07 Feb 2027" in text
    assert "Dry run: nothing changed." in text
    assert not database_path(settings).is_file()


def test_yes_moves_the_calendar_and_the_path_says_so(cli: CliRunner, settings: Settings) -> None:
    outcome = _run(cli, settings, "warp", "--to-week", "1", "--yes")

    assert outcome.exit_code == 0, outcome.output
    assert "Calendar moved. This week is week 1." in _words(outcome.output)
    assert _warps(settings) == [("to_week", 1)]

    path = _run(cli, settings, "path", "--week", "1")

    assert path.exit_code == 0, path.output
    assert (
        "Dates moved 1 week later by time-warp. `dogma warp --original` restores them."
        in _words(path.output)
    )


def test_confirming_at_the_prompt_moves_the_calendar(cli: CliRunner, settings: Settings) -> None:
    outcome = _run(cli, settings, "warp", "--to-week", "1", input="y\n")

    assert outcome.exit_code == 0, outcome.output
    assert "Move the calendar? [y/N]" in outcome.output
    assert _warps(settings) == [("to_week", 1)]


def test_declining_at_the_prompt_changes_nothing(cli: CliRunner, settings: Settings) -> None:
    outcome = _run(cli, settings, "warp", "--to-week", "1", input="n\n")

    assert outcome.exit_code == 0, outcome.output
    assert "Nothing changed." in outcome.output
    assert _warps(settings) == []


def test_a_move_to_the_current_week_says_nothing_changed(
    cli: CliRunner, settings: Settings
) -> None:
    outcome = _run(cli, settings, "warp", "--original", "--yes")

    assert outcome.exit_code == 0, outcome.output
    assert "This week is already week 2. Nothing changed." in _words(outcome.output)
    assert _warps(settings) == []


@pytest.mark.parametrize(
    "options",
    [
        ("--to-week", "1", "--ahead", "2"),
        ("--back", "1", "--original"),
        ("--ahead", "1", "--to-week", "3", "--dry-run"),
    ],
)
def test_conflicting_options_are_a_usage_error(
    cli: CliRunner, settings: Settings, options: tuple[str, ...]
) -> None:
    outcome = _run(cli, settings, "warp", *options)

    assert outcome.exit_code == 2
    assert "choose one of --to-week, --ahead, --back or --original" in _words(outcome.output)
    assert _warps(settings) == []


def test_a_sync_names_the_furthest_exercise(cli: CliRunner, settings: Settings) -> None:
    _run_attempt(settings, WEEK_TWO_PROBLEM)

    outcome = _run(cli, settings, "warp", "--dry-run")

    assert outcome.exit_code == 0, outcome.output
    text = _words(outcome.output)
    assert f"Your furthest exercise is in week 2 ({WEEK_TWO_PROBLEM})." in text
    assert "This week is already week 2. Nothing changed." in text


def test_the_reset_needs_the_typed_word(cli: CliRunner, settings: Settings) -> None:
    _run_attempt(settings, PROBLEM)

    outcome = _run(cli, settings, "course", "reset", input="yes\n")

    assert outcome.exit_code == 0, outcome.output
    assert "Cancelled. Nothing was cleared." in _words(outcome.output)
    assert _attempts(settings) == 1


def test_the_typed_reset_clears_progress_and_sets_the_workspace_aside(
    cli: CliRunner, settings: Settings
) -> None:
    _run_attempt(settings, PROBLEM)
    workspace = settings.workspace_dir / "cml" / "vector-operations"
    workspace.mkdir(parents=True)
    (workspace / "p01-vector-projection.py").write_text("def solve(x):\n    return x\n")

    outcome = _run(cli, settings, "course", "reset", input="reset\n")

    assert outcome.exit_code == 0, outcome.output
    text = _words(outcome.output)
    assert "Backup:" in text
    assert "Cleared 1 rows from" in text
    assert "This week is week 1." in text
    assert _attempts(settings) == 0
    assert _warps(settings)[-1] == ("course_reset", 1)
    assert not (workspace / "p01-vector-projection.py").exists()
    [history] = list((settings.var_dir / "workspace-history").glob("reset-*"))
    assert (history / "cml" / "vector-operations" / "p01-vector-projection.py").is_file()
    [backup] = sorted((settings.var_dir / "backups").glob("dogma-*.json"))
    assert len(json.loads(backup.read_text(encoding="utf-8"))["tables"]["attempt"]) == 1


def test_a_reset_with_recording_off_is_refused(cli: CliRunner, settings: Settings) -> None:
    off = Settings(repo_root=settings.repo_root, record_attempts=False)

    outcome = cli.invoke(app, ["course", "reset", "--yes"], obj=off, env=WIDE)

    assert outcome.exit_code == 1
    assert "CALENDAR_RECORDING_OFF" in outcome.output
    assert not database_path(settings).is_file()


def test_the_warp_command_help_has_its_example(cli: CliRunner, settings: Settings) -> None:
    outcome = _run(cli, settings, "warp", "--help")

    assert outcome.exit_code == 0
    assert "--to-week" in outcome.output and "--dry-run" in outcome.output
    assert "Example: dogma warp --to-week 3" in _words(outcome.output)
