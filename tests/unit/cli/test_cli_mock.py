"""``dogma mock``: start, reveal, score (file or interactive), debrief, abandon and status.

The repo is a scratch copy of the real curriculum with the example topic and the sys cases. The
learner has passed the example core problem and retained its topic, so every part can be chosen.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select
from tests.unit.db.support import make_report
from tests.unit.learning.support import fix_clock, utc_timezone  # noqa: F401  autouse clock
from typer.testing import CliRunner

from dogma.cli import app
from dogma.content.rubrics import load_rubric_version
from dogma.core.config import Settings
from dogma.db.answer_models import Answer
from dogma.db.errors import StorageErrorCode
from dogma.db.mock_models import MockSession
from dogma.db.models import AttemptMode, Mastery, TopicState
from dogma.db.session import database_path, open_session
from dogma.db.store import record_attempt

REPO_ROOT = Path(__file__).resolve().parents[3]
TOPIC = "cml/vector-operations"
NOON = datetime(2026, 11, 10, 12, tzinfo=UTC)  # week 3 of the core path
WIDE = {"COLUMNS": "200"}


@pytest.fixture
def mock_settings(settings: Settings, repo: Path) -> Settings:
    """The cli repo plus the sys track, and a learner who passed and retained the example topic."""
    shutil.copytree(REPO_ROOT / "content" / "sys", repo / "content" / "sys")
    record_attempt(
        settings,
        make_report(public=(3, 3), hidden=(2, 2), attempt_id="b" * 32),
        mode=AttemptMode.SUBMIT,
        core=True,
        started_at=NOON - timedelta(days=30, seconds=1),
        submitted_at=NOON - timedelta(days=30),
    )
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        state = session.get(TopicState, TOPIC)
        assert state is not None
        state.mastery = Mastery.RETAINED.value
        state.retained_at = NOON - timedelta(days=20)
    return settings


def run(cli: CliRunner, settings: Settings, *args: str, input: str | None = None) -> object:
    return cli.invoke(app, ["mock", *args], obj=settings, input=input, env=WIDE)


def _words(output: str) -> str:
    return " ".join(output.split())


def _score_file(tmp_path: Path, score: int) -> Path:
    rubric = load_rubric_version(REPO_ROOT / "curriculum", "mock", 1)
    path = tmp_path / "scores.json"
    path.write_text(json.dumps(dict.fromkeys(rubric.criteria_ids, score)), encoding="utf-8")
    return path


def _answers(settings: Settings) -> int:
    if not database_path(settings).is_file():
        return 0
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        return session.scalar(select(func.count()).select_from(Answer)) or 0


def _mocks(settings: Settings) -> list[str]:
    if not database_path(settings).is_file():
        return []
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        return [row.status for row in session.scalars(select(MockSession))]


def test_start_prints_the_plan_with_part_timings(
    cli: CliRunner, mock_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    fix_clock(monkeypatch, NOON)

    result = run(cli, mock_settings, "start")

    assert result.exit_code == 0, result.output
    text = _words(result.output)
    assert "Mock started" in text
    assert "Live coding: 10-15 min" in text
    assert "Case, reframe: 2 min" in text
    assert "Case, questions: 4 min" in text
    assert "Case, structure: 12 min" in text
    assert "Case, synthesis: 3 min" in text
    assert "Overall communication: throughout" in text
    assert "Vector projection" in text
    assert "dogma mock reveal" in text
    assert _mocks(mock_settings) == ["open"]


def test_a_second_start_is_refused_with_the_code(
    cli: CliRunner, mock_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    fix_clock(monkeypatch, NOON)
    run(cli, mock_settings, "start")

    result = run(cli, mock_settings, "start")

    assert result.exit_code == 1
    assert "[MOCK_ALREADY_OPEN]" in result.output
    assert _mocks(mock_settings) == ["open"]


def test_an_unknown_case_is_a_content_error(
    cli: CliRunner, mock_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    fix_clock(monkeypatch, NOON)

    result = run(cli, mock_settings, "start", "--case", "sys/no-such-case")

    assert result.exit_code == 1
    assert "[CONTENT_NOT_FOUND]" in result.output
    assert _mocks(mock_settings) == []


def test_reveal_gives_the_matching_answer_or_says_there_is_nothing(
    cli: CliRunner, mock_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    fix_clock(monkeypatch, NOON)
    run(cli, mock_settings, "start", "--case", "sys/churn-prediction-case")

    found = run(cli, mock_settings, "reveal", "What does the current retention process do?")
    missed = run(cli, mock_settings, "reveal", "What is the weather like?")

    assert found.exit_code == 0, found.output
    assert "20 percent discount" in _words(found.output)
    assert "The interviewer has nothing on that." in _words(missed.output)


def test_reveal_without_an_open_mock_says_how_to_start(
    cli: CliRunner, mock_settings: Settings
) -> None:
    result = run(cli, mock_settings, "reveal", "retention")

    assert result.exit_code == 1
    assert "[MOCK_NOT_OPEN]" in result.output
    assert "dogma mock start" in result.output


def test_score_from_a_file_stores_the_mock_and_prints_the_band(
    cli: CliRunner, mock_settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fix_clock(monkeypatch, NOON)
    run(cli, mock_settings, "start")

    result = run(cli, mock_settings, "score", "--scores-file", str(_score_file(tmp_path, 3)))

    assert result.exit_code == 0, result.output
    text = _words(result.output)
    assert "90 of 120" in text
    assert "Stored" in text
    assert _answers(mock_settings) == 1
    assert _mocks(mock_settings) == ["scored"]


def test_a_score_file_with_a_missing_item_stores_nothing(
    cli: CliRunner, mock_settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fix_clock(monkeypatch, NOON)
    run(cli, mock_settings, "start")
    path = _score_file(tmp_path, 3)
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["coding.speed"]
    path.write_text(json.dumps(data), encoding="utf-8")

    result = run(cli, mock_settings, "score", "--scores-file", str(path))

    assert result.exit_code == 1
    assert "[MOCK_SCORE_INVALID]" in result.output
    assert "coding.speed is missing" in result.output
    assert _answers(mock_settings) == 0
    assert _mocks(mock_settings) == ["open"]


def test_interactive_score_walks_all_thirty_items_with_optional_notes(
    cli: CliRunner, mock_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    fix_clock(monkeypatch, NOON)
    run(cli, mock_settings, "start")
    typed = "4\n\n" * 29 + "2\nSkipped the edge cases.\n"

    result = run(cli, mock_settings, "score", input=typed)

    assert result.exit_code == 0, result.output
    assert "Stored" in result.output
    assert _answers(mock_settings) == 1


def test_debrief_shows_totals_strength_weakness_and_the_case_debrief(
    cli: CliRunner, mock_settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fix_clock(monkeypatch, NOON)
    run(cli, mock_settings, "start", "--case", "sys/churn-prediction-case")
    run(cli, mock_settings, "score", "--scores-file", str(_score_file(tmp_path, 3)))

    result = run(cli, mock_settings, "debrief")

    assert result.exit_code == 0, result.output
    text = _words(result.output)
    assert "Total: 90 of 120" in text
    assert "Strongest part:" in text
    assert "Improve first:" in text
    assert "Tier by lifetime value times risk" in text


def test_abandon_closes_the_mock_and_status_says_so(
    cli: CliRunner, mock_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    fix_clock(monkeypatch, NOON)
    run(cli, mock_settings, "start")

    abandoned = run(cli, mock_settings, "abandon")
    status = run(cli, mock_settings, "status")

    assert abandoned.exit_code == 0, abandoned.output
    assert _mocks(mock_settings) == ["abandoned"]
    assert "No mock is open" in _words(status.output)
    assert _answers(mock_settings) == 0


def test_recording_off_refuses_to_start_and_creates_no_database(
    cli: CliRunner, mock_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    fix_clock(monkeypatch, NOON)
    off = Settings(repo_root=mock_settings.repo_root, _env_file=None, record_attempts=False)
    database_path(off).unlink(missing_ok=True)

    result = run(cli, off, "start")

    assert result.exit_code == 1
    assert "[MOCK_RECORDING_OFF]" in result.output
    assert not database_path(off).exists() or _mocks(off) == []


def test_in_week_three_the_third_session_of_the_day_is_the_mock(
    cli: CliRunner, mock_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    fix_clock(monkeypatch, NOON)
    for _ in range(2):
        today = cli.invoke(app, ["today"], obj=mock_settings, env=WIDE)
        assert today.exit_code == 0, today.output
        closed = cli.invoke(
            app,
            [
                "close",
                "--stopped-where",
                "Practice, example 2",
                "--next-step",
                "Solve it again",
            ],
            obj=mock_settings,
            env=WIDE,
        )
        assert closed.exit_code == 0, closed.output

    third = cli.invoke(app, ["today"], obj=mock_settings, env=WIDE)

    assert third.exit_code == 0, third.output
    text = _words(third.output)
    assert "session 3 of 3" in text
    assert "Interview" in text
    assert "dogma mock start" in text


def test_mock_help_ends_with_an_example(cli: CliRunner, mock_settings: Settings) -> None:
    result = cli.invoke(app, ["mock", "--help"], obj=mock_settings, env=WIDE)

    assert result.exit_code == 0
    assert "Example: dogma mock start" in _words(result.output)
