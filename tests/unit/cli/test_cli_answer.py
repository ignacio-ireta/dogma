"""``dogma answer``: the question, the answer text, the four scores, and the mastery it earns."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import select
from typer.testing import CliRunner

from dogma.cli import app
from dogma.core.config import Settings
from dogma.db.answer_models import Answer
from dogma.db.errors import StorageErrorCode
from dogma.db.models import Mastery, TopicState
from dogma.db.session import database_path, open_session

TOPIC = "cml/vector-operations"
READY = "accuracy=3,structure=3,trade-offs=3,communication=4"
NOW = datetime(2026, 10, 27, 9, tzinfo=UTC)
WIDE = {"COLUMNS": "200"}


def _text(output: str) -> str:
    return " ".join(output.split())


def _set_retained(settings: Settings) -> None:
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        session.add(
            TopicState(
                topic_id=TOPIC,
                mastery=Mastery.RETAINED.value,
                solved_at=NOW - timedelta(days=12),
                retained_at=NOW - timedelta(days=3),
            )
        )


def _stored(settings: Settings) -> list[tuple[str | None, dict[str, int], str]]:
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        return [
            (row.text, row.scores, row.prompt_ref)
            for row in session.scalars(select(Answer).order_by(Answer.created_at))
        ]


def _state(settings: Settings) -> str | None:
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        state = session.get(TopicState, TOPIC)
        return None if state is None else state.mastery


def test_scores_on_the_command_line_are_stored_and_the_anchors_are_printed(
    cli: CliRunner, settings: Settings
) -> None:
    outcome = cli.invoke(
        app, ["answer", TOPIC, "--spoken", "--scores", READY], obj=settings, env=WIDE
    )

    text = _text(outcome.output)
    assert outcome.exit_code == 0, outcome.output
    assert "What does the dot product measure, geometrically?" in text
    assert "The claim is wrong, vague, or missing." in text  # accuracy's anchor for a score of 1
    assert "Stored answer" in text
    assert "Mastery: not started" in text
    assert _stored(settings) == [
        (
            None,
            {"accuracy": 3, "structure": 3, "trade-offs": 3, "communication": 4},
            "interview.md#Q1",
        )
    ]


def test_a_written_answer_from_a_file_is_stored_as_text(
    cli: CliRunner, settings: Settings, tmp_path: Path
) -> None:
    answer = tmp_path / "answer.txt"
    answer.write_text("It measures how much two vectors agree in direction.\n", encoding="utf-8")

    outcome = cli.invoke(
        app, ["answer", TOPIC, "--file", str(answer), "--scores", READY], obj=settings
    )

    assert outcome.exit_code == 0, outcome.output
    [(text, _scores, _ref)] = _stored(settings)
    assert text == "It measures how much two vectors agree in direction."


def test_an_interactive_answer_ends_at_an_empty_line_and_scores_are_prompted(
    cli: CliRunner, settings: Settings
) -> None:
    typed = "It measures agreement.\n\n3\n3\n3\n4\n"

    outcome = cli.invoke(app, ["answer", TOPIC], obj=settings, input=typed)

    assert outcome.exit_code == 0, outcome.output
    [(text, scores, _ref)] = _stored(settings)
    assert text == "It measures agreement."
    assert scores == {"accuracy": 3, "structure": 3, "trade-offs": 3, "communication": 4}


def test_an_interactive_score_outside_one_to_four_is_asked_again(
    cli: CliRunner, settings: Settings
) -> None:
    typed = "Agreement.\n\n9\n3\n3\n3\n4\n"

    outcome = cli.invoke(app, ["answer", TOPIC], obj=settings, input=typed)

    assert outcome.exit_code == 0, outcome.output
    assert "whole number from 1 to 4" in _text(outcome.output)


def test_no_answer_text_and_no_flag_is_a_usage_error(cli: CliRunner, settings: Settings) -> None:
    outcome = cli.invoke(app, ["answer", TOPIC, "--scores", READY], obj=settings, input="")

    assert outcome.exit_code == 2
    assert "--spoken" in outcome.output
    assert not database_path(settings).exists()


def test_file_and_spoken_together_are_a_usage_error(cli: CliRunner, settings: Settings) -> None:
    outcome = cli.invoke(
        app, ["answer", TOPIC, "--spoken", "--file", "x.txt", "--scores", READY], obj=settings
    )

    assert outcome.exit_code == 2


def test_a_malformed_scores_string_is_a_usage_error(cli: CliRunner, settings: Settings) -> None:
    outcome = cli.invoke(
        app, ["answer", TOPIC, "--spoken", "--scores", "accuracy=high"], obj=settings
    )

    assert outcome.exit_code == 2
    assert "accuracy" in outcome.output


def test_scores_that_do_not_match_the_rubric_fail_with_its_code(
    cli: CliRunner, settings: Settings
) -> None:
    outcome = cli.invoke(
        app, ["answer", TOPIC, "--spoken", "--scores", "accuracy=3,structure=3"], obj=settings
    )

    assert outcome.exit_code == 1
    assert "[ANSWER_SCORE_INVALID]" in outcome.output
    assert "trade-offs is missing" in outcome.output


def test_a_question_number_that_does_not_exist_is_a_content_error(
    cli: CliRunner, settings: Settings
) -> None:
    outcome = cli.invoke(
        app, ["answer", TOPIC, "--question", "9", "--spoken", "--scores", READY], obj=settings
    )

    assert outcome.exit_code == 1
    assert "[CONTENT_NOT_FOUND]" in outcome.output


def test_a_retained_topic_becomes_interview_ready_and_the_output_says_why(
    cli: CliRunner, settings: Settings
) -> None:
    _set_retained(settings)

    outcome = cli.invoke(
        app, ["answer", TOPIC, "--spoken", "--scores", READY], obj=settings, env=WIDE
    )

    text = _text(outcome.output)
    assert outcome.exit_code == 0, outcome.output
    assert "Mastery: retained -> interview-ready" in text
    assert "Interview-ready: retained, and every dimension scored 3 or higher." in text
    assert _state(settings) == "interview-ready"


def test_a_weak_answer_on_a_retained_topic_names_the_dimension_below_three(
    cli: CliRunner, settings: Settings
) -> None:
    _set_retained(settings)
    weak = "accuracy=4,structure=4,trade-offs=2,communication=4"

    outcome = cli.invoke(
        app, ["answer", TOPIC, "--spoken", "--scores", weak], obj=settings, env=WIDE
    )

    text = _text(outcome.output)
    assert outcome.exit_code == 0, outcome.output
    assert "Mastery: retained (unchanged)" in text
    assert "trade-offs scored below 3" in text


def test_recording_switched_off_writes_nothing_and_says_so(
    cli: CliRunner, settings: Settings
) -> None:
    off = Settings(repo_root=settings.repo_root, record_attempts=False, _env_file=None)

    outcome = cli.invoke(app, ["answer", TOPIC, "--spoken", "--scores", READY], obj=off, env=WIDE)

    assert outcome.exit_code == 1
    assert "[ANSWER_RECORDING_OFF]" in outcome.output
    assert not database_path(settings).exists()
