"""``dogma runs prune``: dry run by default, explicit --apply, recorded attempts always kept."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from click.testing import Result
from typer.testing import CliRunner

from dogma.cli import app
from dogma.core.config import Settings
from dogma.db.engine import DB_FILENAME
from dogma.db.session import upgrade_database
from dogma.runner.retention import MARKER_NAME

NOW = datetime.now(UTC)
RECORDED = "20250101T000000Z-recorded"
OLD = "20250102T000000Z-oldrun1"
NEWER = "20261020T000000Z-newer01"
LEGACY = "20250103T000000Z-legacy1"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(repo_root=tmp_path / "repo")


def _run_dir(settings: Settings, name: str, completed: datetime | None) -> Path:
    path = settings.runs_dir / name
    path.mkdir(parents=True)
    (path / "report.json").write_text("{}", encoding="utf-8")
    if completed is not None:
        payload = {
            "schema_version": 1,
            "attempt_id": name,
            "completed_at": completed.isoformat(timespec="microseconds"),
        }
        (path / MARKER_NAME).write_text(json.dumps(payload), encoding="utf-8")
    return path


def _record(settings: Settings, attempt_id: str, run_dir: str | None) -> None:
    upgrade_database(settings)
    with sqlite3.connect(settings.var_dir / DB_FILENAME) as conn:
        conn.execute(
            "INSERT INTO attempt (id, problem_id, problem_version, mode, status, public_passed,"
            " public_total, hidden_passed, hidden_total, hints_used, run_dir, started_at,"
            " submitted_at, duration_ms) VALUES (?, 'cml/x/p01', 1, 'submit', 'passed', 1, 1, 1,"
            " 1, 0, ?, '2025-01-01T00:00:00+00:00', '2025-01-01T00:00:01+00:00', 1000)",
            (attempt_id, run_dir),
        )
    conn.close()


def _invoke(cli: CliRunner, settings: Settings, *args: str) -> Result:
    return cli.invoke(app, ["runs", "prune", *args], obj=settings, env={"COLUMNS": "200"})


def test_help_shows_the_dry_run_example_and_the_apply_switch(cli: CliRunner) -> None:
    result = cli.invoke(app, ["runs", "prune", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "dogma runs prune --older-than-days 30" in result.output
    assert "--older-than-days" in result.output
    assert "--apply" in result.output


def test_default_is_a_dry_run_that_deletes_nothing(cli: CliRunner, settings: Settings) -> None:
    old = _run_dir(settings, OLD, NOW - timedelta(days=60))
    result = _invoke(cli, settings)

    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert "--apply" in result.output
    assert OLD in result.output
    assert old.is_dir()


def test_apply_removes_eligible_runs_and_reports_the_count(
    cli: CliRunner, settings: Settings
) -> None:
    old = _run_dir(settings, OLD, NOW - timedelta(days=60))
    result = _invoke(cli, settings, "--apply")

    assert result.exit_code == 0, result.output
    assert "Removed 1" in result.output
    assert not old.exists()


def test_recorded_attempts_are_kept_even_when_old_and_marked(
    cli: CliRunner, settings: Settings
) -> None:
    recorded = _run_dir(settings, RECORDED, NOW - timedelta(days=400))
    _record(settings, RECORDED, f"var/runs/{RECORDED}")
    relative_only = _run_dir(settings, "20250104T000000Z-relonly1", NOW - timedelta(days=400))
    _record(settings, "20250104T000000Z-relonly1", None)
    _run_dir(settings, OLD, NOW - timedelta(days=60))

    result = _invoke(cli, settings, "--apply")
    assert result.exit_code == 0, result.output
    assert recorded.is_dir()
    assert relative_only.is_dir(), "an attempt id in the database protects its directory"


def test_the_default_age_is_thirty_days(cli: CliRunner, settings: Settings) -> None:
    fresh_enough = _run_dir(settings, NEWER, NOW - timedelta(days=10))
    result = _invoke(cli, settings, "--apply")
    assert result.exit_code == 0, result.output
    assert fresh_enough.is_dir()


def test_the_age_must_be_at_least_one_day(cli: CliRunner, settings: Settings) -> None:
    old = _run_dir(settings, OLD, NOW - timedelta(days=400))
    result = _invoke(cli, settings, "--older-than-days", "0", "--apply")
    assert result.exit_code == 2
    assert old.is_dir()


def test_a_missing_database_creates_no_state(cli: CliRunner, settings: Settings) -> None:
    _run_dir(settings, OLD, NOW - timedelta(days=60))
    result = _invoke(cli, settings, "--apply", "--older-than-days", "30")

    assert result.exit_code == 0, result.output
    assert not (settings.var_dir / DB_FILENAME).exists()
    assert not (settings.workspace_dir).exists()


def test_a_malformed_database_aborts_before_any_deletion(
    cli: CliRunner, settings: Settings
) -> None:
    old = _run_dir(settings, OLD, NOW - timedelta(days=60))
    db = settings.var_dir / DB_FILENAME
    db.write_bytes(b"not a database" * 100)

    result = _invoke(cli, settings, "--apply")
    assert result.exit_code == 1
    assert "DB_READ_FAILED" in result.output
    assert old.is_dir()
    assert db.read_bytes().startswith(b"not a database")


def test_a_run_without_a_marker_is_never_removed(cli: CliRunner, settings: Settings) -> None:
    legacy = _run_dir(settings, LEGACY, None)
    result = _invoke(cli, settings, "--apply")
    assert result.exit_code == 0, result.output
    assert legacy.is_dir()


def test_nothing_eligible_says_so(cli: CliRunner, settings: Settings) -> None:
    result = _invoke(cli, settings)
    assert result.exit_code == 0
    assert "No run directories are eligible" in result.output
