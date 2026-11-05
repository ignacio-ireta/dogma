"""Advisory model scores beside self-scores: the comparison, storage, and the earned-reveal gate.

The learner's self-score is the source of truth. Storing a model score must leave the self-score,
the mastery and the topic state exactly as they were.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from dogma.core.config import Settings
from dogma.db.answer_models import Answer
from dogma.db.errors import StorageErrorCode
from dogma.db.models import Mastery, TopicState
from dogma.db.session import open_session
from dogma.learning.advisory import (
    attach_to_answer,
    compare,
    model_score_payload,
)
from dogma.learning.answers import record_answer
from dogma.learning.mock_rules import shares_keyword
from dogma.tutor.models import AdvisoryScore, AdvisoryScores

REPO_ROOT = Path(__file__).resolve().parents[3]
TOPIC = "cml/vector-operations"
NOW = datetime(2026, 10, 27, 9, tzinfo=UTC)
SELF = {"accuracy": 3, "structure": 3, "trade-offs": 2, "communication": 4}


def _advisory(*scores: tuple[str, int]) -> AdvisoryScores:
    return AdvisoryScores(
        scores=tuple(AdvisoryScore(name, value, "quoted words") for name, value in scores),
        dropped=1,
        model="qwen3.8-27b-q6",
        prompt_version=1,
    )


def test_a_matching_advisory_score_is_reported_as_agreement() -> None:
    result = compare(SELF, _advisory(("accuracy", 3), ("structure", 3)))

    assert result.compared == 2
    assert result.far == () and result.near == ()
    assert result.summary == (
        "The model agrees with your scores on all 2 it scored. Your scores stand."
    )


def test_differences_are_split_into_two_points_and_one_point() -> None:
    result = compare(SELF, _advisory(("accuracy", 1), ("structure", 2), ("communication", 4)))

    assert result.far == ("accuracy",)
    assert result.near == ("structure",)
    assert result.summary == (
        "The model differs from you by 2 or more on accuracy, and by 1 on structure. "
        "Your scores stand."
    )


def test_no_advisory_scores_means_nothing_to_compare() -> None:
    result = compare(SELF, _advisory())

    assert result.compared == 0
    assert result.summary == (
        "The model gave no scores with a quote from your text, so there is nothing to compare. "
        "Your scores stand."
    )


def test_a_criterion_the_learner_did_not_score_is_not_compared() -> None:
    result = compare({"accuracy": 3}, _advisory(("structure", 1)))

    assert result.compared == 0


def test_the_payload_keeps_scores_quotes_and_the_rubric_version() -> None:
    payload = model_score_payload(
        _advisory(("accuracy", 3)), rubric_id="topic-answer", rubric_version=1
    )

    assert payload["scores"] == {"accuracy": {"score": 3, "quote": "quoted words"}}
    assert payload["rubric_id"] == "topic-answer"
    assert payload["rubric_version"] == 1
    assert payload["dropped"] == 1
    assert payload["model"] == "qwen3.8-27b-q6"


def test_storing_a_model_score_leaves_the_self_score_and_mastery_alone(
    settings: Settings,
) -> None:
    shutil.copytree(REPO_ROOT / "curriculum", settings.repo_root / "curriculum")
    answer = record_answer(
        settings,
        topic_id=TOPIC,
        prompt_ref="interview.md#Q1",
        text="A written answer.",
        scores=SELF,
        now=NOW,
    )
    payload = model_score_payload(
        _advisory(("accuracy", 1)), rubric_id="topic-answer", rubric_version=1
    )

    attach_to_answer(settings, answer.answer_id, payload)

    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        row = session.get(Answer, answer.answer_id)
        assert row is not None
        assert row.scores == SELF
        assert row.model_score == payload
        assert session.scalars(select(TopicState)).all() == []
        assert answer.after is Mastery.UNSEEN


def test_shares_keyword_needs_one_real_word_in_common() -> None:
    assert shares_keyword("How do the segments differ?", "How do the segments differ?")
    assert shares_keyword("what about prepaid segments", "How do the segments differ?")
    assert not shares_keyword("what is the weather", "How do the segments differ?")
    assert not shares_keyword("what is the", "How do the segments differ?")
