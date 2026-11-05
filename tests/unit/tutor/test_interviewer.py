"""Interviewer mode: the model picks one reveal by number, or none. It never sees the answers."""

from __future__ import annotations

import pytest

from dogma.core.config import Settings
from dogma.core.errors import DogmaError, ErrorCode
from dogma.tutor.gateway import Gateway
from dogma.tutor.interviewer import (
    MAX_QUESTION_CHARS,
    build_user_text,
    clean_question,
    pick_reveal,
)
from dogma.tutor.models import ProviderError
from dogma.tutor.providers.fake import FakeProvider

BRIEF = "A mid-size mobile operator has 12 million subscribers. Churn rose this year."
WHEN_ASKED = (
    "What does the current retention process do?",
    "How do the segments differ?",
)
REVEAL_ANSWER = "REVEAL_ANSWER_SENTINEL_61aa: Subscribers who call to cancel get a discount."
DEBRIEF = "DEBRIEF_SENTINEL_0c4e: the strong answer tiers by lifetime value."


def test_the_user_text_has_the_brief_the_numbered_questions_and_the_question() -> None:
    text = build_user_text(BRIEF, WHEN_ASKED, "How are the segments different?")

    assert BRIEF in text
    assert "1: What does the current retention process do?" in text
    assert "2: How do the segments differ?" in text
    assert "CANDIDATE QUESTION" in text
    assert "How are the segments different?" in text


def test_reveal_answers_and_the_debrief_are_never_in_the_prompt() -> None:
    text = build_user_text(BRIEF, WHEN_ASKED, "What does retention do?")

    assert REVEAL_ANSWER not in text
    assert DEBRIEF not in text


def test_a_matching_number_is_released(settings: Settings) -> None:
    provider = FakeProvider(['{"reveal": 2}'])

    picked = pick_reveal(
        Gateway(settings, provider),
        brief=BRIEF,
        when_asked=WHEN_ASKED,
        question="How do the segments differ?",
    )

    assert picked.number == 2
    assert picked.prompt_version == 1


def test_null_means_the_interviewer_has_nothing_on_that(settings: Settings) -> None:
    provider = FakeProvider(['{"reveal": null}'])

    picked = pick_reveal(
        Gateway(settings, provider), brief=BRIEF, when_asked=WHEN_ASKED, question="Weather?"
    )

    assert picked.number is None


@pytest.mark.parametrize(
    "reply",
    [
        '{"reveal": 3}',  # past the end of the list
        '{"reveal": 0}',  # numbers start at 1
        '{"reveal": "2"}',  # a string, not a number
        '{"reveal": true}',  # a bool is not a number
        '{"reveal": 1, "answer": "free text"}',  # extra key: the model may not write text
        '{"pick": 1}',  # wrong key
        "The interviewer releases reveal 2.",  # prose
    ],
)
def test_a_reply_that_is_not_a_listed_number_is_refused(settings: Settings, reply: str) -> None:
    provider = FakeProvider([reply])

    with pytest.raises(DogmaError) as refused:
        pick_reveal(Gateway(settings, provider), brief=BRIEF, when_asked=WHEN_ASKED, question="x")

    assert refused.value.code is ErrorCode.TUTOR_INTERVIEWER_REPLY_INVALID


def test_a_down_model_raises_the_degradable_error(settings: Settings) -> None:
    provider = FakeProvider(
        [ProviderError(ErrorCode.LLM_UNAVAILABLE, "unavailable", "the local model is down")]
    )

    with pytest.raises(DogmaError) as down:
        pick_reveal(Gateway(settings, provider), brief=BRIEF, when_asked=WHEN_ASKED, question="x")

    assert down.value.code is ErrorCode.LLM_UNAVAILABLE


def test_a_question_is_trimmed_and_checked() -> None:
    assert clean_question("  How do the segments differ?  ") == "How do the segments differ?"
    for bad in ["", "   ", "x" * (MAX_QUESTION_CHARS + 1)]:
        with pytest.raises(DogmaError) as invalid:
            clean_question(bad)
        assert invalid.value.code is ErrorCode.TUTOR_QUESTION_INVALID


def test_an_invalid_question_never_reaches_the_model(settings: Settings) -> None:
    provider = FakeProvider([])

    with pytest.raises(DogmaError):
        pick_reveal(Gateway(settings, provider), brief=BRIEF, when_asked=WHEN_ASKED, question=" ")

    assert provider.requests == []
