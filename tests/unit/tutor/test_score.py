"""Score mode: the advisory rubric scores are strict, quoted and never change the learner's scores.

The model is a FakeProvider behind a real Gateway, so the call log and the checks are real.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dogma.content.rubrics import Rubric, load_rubric_version
from dogma.core.config import Settings
from dogma.core.errors import DogmaError, ErrorCode
from dogma.tutor.gateway import Gateway, call_log_path
from dogma.tutor.models import AdvisoryScore, ProviderError
from dogma.tutor.providers.fake import FakeProvider
from dogma.tutor.score import build_user_text, parse_scores, score_answer

CURRICULUM = Path(__file__).resolve().parents[3] / "curriculum"
ANSWER = (
    "I would start with a churn label defined as a cancelled line. Then I would segment by "
    "plan, because prepaid behaves differently. The trade-off is that a holdout costs revenue."
)


@pytest.fixture
def topic_rubric() -> Rubric:
    return load_rubric_version(CURRICULUM, "topic-answer", 1)


@pytest.fixture
def mock_rubric() -> Rubric:
    return load_rubric_version(CURRICULUM, "mock", 1)


def _reply(*items: object) -> str:
    return json.dumps({"scores": list(items)})


def _item(criterion: str, score: object, quote: object) -> dict[str, object]:
    return {"id": criterion, "score": score, "quote": quote}


def test_user_text_lists_every_criterion_and_the_learner_text(topic_rubric: Rubric) -> None:
    text = build_user_text(topic_rubric, ANSWER)

    for criterion in topic_rubric.criteria_ids:
        assert criterion in text
    assert ANSWER in text
    assert "1: " in text  # the anchors are shown, so the score means what the rubric says


def test_user_text_lists_every_mock_item_once(mock_rubric: Rubric) -> None:
    text = build_user_text(mock_rubric, "coding.narration: I talked through the loop.")

    for criterion in mock_rubric.criteria_ids:
        assert text.count(f"- {criterion}:") == 1


def test_a_quoted_score_is_kept(topic_rubric: Rubric) -> None:
    reply = _reply(_item("accuracy", 3, "a churn label defined as a cancelled line"))

    scores, dropped = parse_scores(reply, criteria=topic_rubric.criteria_ids, learner_text=ANSWER)

    assert scores == (AdvisoryScore("accuracy", 3, "a churn label defined as a cancelled line"),)
    assert dropped == 0


def test_a_fenced_json_reply_is_accepted(topic_rubric: Rubric) -> None:
    reply = "```json\n" + _reply(_item("structure", 2, "Then I would segment by plan")) + "\n```"

    scores, dropped = parse_scores(reply, criteria=topic_rubric.criteria_ids, learner_text=ANSWER)

    assert [score.criterion for score in scores] == ["structure"]
    assert dropped == 0


@pytest.mark.parametrize(
    "item",
    [
        _item("unknown-dimension", 3, "churn label"),  # id is not a criterion
        _item("accuracy", 0, "churn label"),  # below the scale
        _item("accuracy", 5, "churn label"),  # above the scale
        _item("accuracy", "3", "churn label"),  # a string, not a whole number
        _item("accuracy", True, "churn label"),  # a bool is not a score
        _item("accuracy", 3, "a sentence the learner never wrote"),  # quote not in the text
        _item("accuracy", 3, ""),  # empty quote
        _item("accuracy", 3, "churn label " * 40),  # quote over the length limit
        {"id": "accuracy", "score": 3},  # no quote at all
        {"id": "accuracy", "score": 3, "quote": "churn label", "why": "extra"},  # extra key
        "accuracy=3",  # not an object
    ],
)
def test_an_item_that_breaks_a_rule_is_dropped_and_counted(
    topic_rubric: Rubric, item: object
) -> None:
    scores, dropped = parse_scores(
        _reply(item), criteria=topic_rubric.criteria_ids, learner_text=ANSWER
    )

    assert scores == ()
    assert dropped == 1


def test_a_second_score_for_the_same_criterion_is_dropped(topic_rubric: Rubric) -> None:
    reply = _reply(
        _item("accuracy", 3, "churn label"),
        _item("accuracy", 1, "churn label"),
    )

    scores, dropped = parse_scores(reply, criteria=topic_rubric.criteria_ids, learner_text=ANSWER)

    assert [(score.criterion, score.score) for score in scores] == [("accuracy", 3)]
    assert dropped == 1


@pytest.mark.parametrize(
    "reply",
    [
        "Your answer is correct. The grader decides.",  # prose, not JSON
        '{"scores": "accuracy"}',  # scores is not a list
        '{"rating": []}',  # no scores key
        "[]",  # not an object
        "",  # empty
    ],
)
def test_a_reply_that_is_not_the_score_object_is_refused(topic_rubric: Rubric, reply: str) -> None:
    with pytest.raises(DogmaError) as refused:
        parse_scores(reply, criteria=topic_rubric.criteria_ids, learner_text=ANSWER)

    assert refused.value.code is ErrorCode.TUTOR_SCORE_REPLY_INVALID


def test_score_answer_calls_the_gateway_once_and_logs_metadata_only(
    settings: Settings, topic_rubric: Rubric
) -> None:
    provider = FakeProvider([_reply(_item("trade-offs", 2, "The trade-off is that"))])
    gateway = Gateway(settings, provider)

    result = score_answer(gateway, topic_rubric, ANSWER)

    assert [score.criterion for score in result.scores] == ["trade-offs"]
    assert result.prompt_version == 1
    [request] = provider.requests
    assert request.messages[0].content.startswith("You are the dogma scorer")
    log = call_log_path(settings).read_text(encoding="utf-8")
    assert '"mode": "score"' in log
    assert ANSWER not in log


def test_a_down_model_raises_the_degradable_error(settings: Settings, topic_rubric: Rubric) -> None:
    provider = FakeProvider(
        [ProviderError(ErrorCode.LLM_UNAVAILABLE, "unavailable", "the local model is down")]
    )

    with pytest.raises(DogmaError) as down:
        score_answer(Gateway(settings, provider), topic_rubric, ANSWER)

    assert down.value.code is ErrorCode.LLM_UNAVAILABLE
