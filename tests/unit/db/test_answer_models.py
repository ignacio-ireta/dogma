"""The answer table: a stored answer, its rubric version, its scores and the CHECK constraints."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from dogma.core.config import Settings
from dogma.db.answer_models import Answer, AnswerKind
from dogma.db.errors import StorageError, StorageErrorCode
from dogma.db.session import database_path, open_session, upgrade_database

WHEN = datetime(2026, 10, 27, 9, 0, tzinfo=UTC)
TOPIC = "cml/vector-operations"


def _answer(**overrides: object) -> Answer:
    values: dict[str, object] = {
        "id": "answer-1",
        "topic_id": TOPIC,
        "kind": AnswerKind.INTERVIEW.value,
        "prompt_ref": "interview.md#Q1",
        "text": "A dot product measures agreement in direction.",
        "rubric_id": "topic-answer",
        "rubric_version": 1,
        "scores": {"accuracy": 3, "structure": 3, "trade-offs": 2, "communication": 4},
        "notes": None,
        "created_at": WHEN,
    }
    values.update(overrides)
    return Answer(**values)  # type: ignore[arg-type]


def test_an_answer_round_trips_with_its_rubric_version_and_scores(settings: Settings) -> None:
    upgrade_database(settings)
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        session.add(_answer())

    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        stored = session.scalars(select(Answer)).one()
        found = (
            stored.topic_id,
            stored.rubric_id,
            stored.rubric_version,
            stored.scores,
            stored.created_at,
            stored.text,
        )

    assert found == (
        TOPIC,
        "topic-answer",
        1,
        {"accuracy": 3, "structure": 3, "trade-offs": 2, "communication": 4},
        WHEN,
        "A dot product measures agreement in direction.",
    )


def test_a_spoken_answer_has_no_text(settings: Settings) -> None:
    upgrade_database(settings)
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        session.add(_answer(id="spoken-1", text=None))

    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        assert session.get(Answer, "spoken-1") is not None


@pytest.mark.parametrize("kind", ["predict", "debrief", ""])
def test_an_unknown_kind_is_refused_by_the_database(settings: Settings, kind: str) -> None:
    upgrade_database(settings)

    with (
        pytest.raises(StorageError) as caught,
        open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session,
    ):
        session.add(_answer(id="bad-kind", kind=kind))

    assert caught.value.code is StorageErrorCode.DB_WRITE_FAILED


def test_the_scores_column_is_json_on_disk(settings: Settings) -> None:
    upgrade_database(settings)
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        session.add(_answer(id="json-1"))

    connection = sqlite3.connect(database_path(settings))
    try:
        raw = connection.execute("SELECT scores FROM answer WHERE id = 'json-1'").fetchone()[0]
    finally:
        connection.close()
    assert json.loads(raw) == {"accuracy": 3, "structure": 3, "trade-offs": 2, "communication": 4}
