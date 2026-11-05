"""The I-18d migration adds one nullable JSON column to answer, and nothing else changes."""

from __future__ import annotations

from sqlalchemy import inspect

from dogma.core.config import Settings
from dogma.db.errors import StorageErrorCode
from dogma.db.session import open_session


def test_answer_gets_a_nullable_model_score_column(settings: Settings) -> None:
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        columns = {
            column["name"]: column for column in inspect(session.connection()).get_columns("answer")
        }

    assert "model_score" in columns
    assert columns["model_score"]["nullable"] is True
    assert "scores" in columns  # the self-scores are untouched
