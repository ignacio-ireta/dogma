"""The SSE machinery: frames, the think-tag filter, and the signals of one gateway call."""

from __future__ import annotations

import pytest
from pydantic import BaseModel
from tests.unit.api.support import ChunkedFakeProvider

from dogma.api.deps import LocalModel
from dogma.api.streaming import (
    Delta,
    Failed,
    Finished,
    Started,
    ThinkFilter,
    frame,
    model_signals,
    text_frame,
)
from dogma.core.config import Settings
from dogma.core.errors import DogmaError, ErrorCode
from dogma.tutor.gateway import CallMeta, strip_thinking
from dogma.tutor.models import CallMode, Message, ProviderError, RenderedPrompt

DOWN = ProviderError(ErrorCode.LLM_UNAVAILABLE, "unavailable", "the local model server is down")


class _Payload(BaseModel):
    text: str


def _meta() -> CallMeta:
    prompt = RenderedPrompt(
        mode=CallMode.HINT,
        level=1,
        version=1,
        sha256="0" * 64,
        messages=(Message(role="user", content="hi"),),
    )
    return CallMeta(mode=CallMode.HINT, prompt=prompt, max_tokens=16)


def _visible(chunks: list[str]) -> str:
    hidden = ThinkFilter()
    return "".join(hidden.feed(chunk) for chunk in chunks) + hidden.finish()


@pytest.mark.parametrize(
    "text",
    [
        "plain text",
        "a<think>hidden</think>b",
        "<think>x</think><think>y</think>end",
        "start<think>never closes",
        "<think>",
        "a < b and <thin not a tag",
        "end with <",
    ],
)
def test_the_visible_text_matches_strip_thinking_at_every_split(text: str) -> None:
    expected = strip_thinking(text)
    for cut in range(len(text) + 1):
        assert _visible([text[:cut], text[cut:]]) == expected, cut
    assert _visible(list(text)) == expected


def test_a_tag_split_across_chunks_is_still_hidden() -> None:
    assert _visible(["a<thi", "nk>secret</th", "ink>b"]) == "ab"


def test_a_frame_is_one_event_with_one_json_line() -> None:
    assert frame("done", _Payload(text="a\nb")) == 'event: done\ndata: {"text":"a\\nb"}\n\n'
    assert text_frame("nudge", "x") == 'event: nudge\ndata: {"text":"x"}\n\n'


def test_a_successful_call_streams_deltas_and_then_finishes(settings: Settings) -> None:
    provider = ChunkedFakeProvider(["hello there"], size=4)

    signals = list(
        model_signals(
            lambda gateway: gateway.complete(_meta()).text, settings=settings, model=provider
        )
    )

    assert isinstance(signals[0], Started)
    assert "".join(s.text for s in signals if isinstance(s, Delta)) == "hello there"
    assert signals[-1] == Finished(value="hello there")


def test_a_model_failure_is_delivered_even_when_the_call_absorbs_it(settings: Settings) -> None:
    def absorb(gateway: object) -> str:
        try:
            gateway.complete(_meta())  # type: ignore[attr-defined]
        except DogmaError:
            return "absorbed"
        return "answered"

    signals = list(model_signals(absorb, settings=settings, model=ChunkedFakeProvider([DOWN])))

    assert isinstance(signals[0], Started)
    assert isinstance(signals[-1], Failed)
    assert signals[-1].error.code == ErrorCode.LLM_UNAVAILABLE


def test_a_refused_endpoint_fails_before_anything_is_streamed(settings: Settings) -> None:
    refused = LocalModel("https://models.example.com/v1")

    signals = list(
        model_signals(lambda gateway: gateway.complete(_meta()), settings=settings, model=refused)
    )

    assert len(signals) == 1
    assert isinstance(signals[0], Failed)
    assert signals[0].error.code == ErrorCode.LLM_ENDPOINT_REFUSED


def test_an_unexpected_exception_in_the_call_is_delivered_not_lost(settings: Settings) -> None:
    def explode(gateway: object) -> str:
        raise RuntimeError("boom")

    signals = list(model_signals(explode, settings=settings, model=ChunkedFakeProvider()))

    assert len(signals) == 1
    assert isinstance(signals[0], Failed)
    assert isinstance(signals[0].error, RuntimeError)
