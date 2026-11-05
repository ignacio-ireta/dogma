"""The local provider: loopback enforcement, request shape, timeouts and failure mapping.

Every HTTP exchange goes through httpx.MockTransport, so no test reaches a real server.
"""

from __future__ import annotations

import json

import httpx
import pytest

from dogma.core.errors import DogmaError, ErrorCode
from dogma.tutor.models import CompletionRequest, Message, ProviderError
from dogma.tutor.providers import local
from dogma.tutor.providers.fake import FakeProvider
from dogma.tutor.providers.local import (
    CONNECT_TIMEOUT_S,
    FIRST_TOKEN_TIMEOUT_S,
    TOTAL_TIMEOUT_S,
    LocalProvider,
    build_payload,
    check_loopback,
    parse_stream_line,
)

BASE = "http://127.0.0.1:8080/v1"


def _request(max_tokens: int = 50) -> CompletionRequest:
    return CompletionRequest(
        model="qwen-test", messages=(Message("user", "hi"),), max_tokens=max_tokens
    )


def _sse(*events: str) -> bytes:
    return "".join(f"data: {event}\n\n" for event in events).encode()


def _delta(text: str) -> str:
    return json.dumps({"choices": [{"delta": {"content": text}}]})


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1:8080/v1", "http://localhost:8080/v1", "http://[::1]:8080/v1"],
)
def test_loopback_hosts_are_accepted(url: str) -> None:
    check_loopback(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://192.168.1.20:8080/v1",
        "http://0.0.0.0:8080/v1",
        "https://api.example.com/v1",
        "http://127.0.0.1.example.org/v1",
        "http://127.0.0.1@example.org/v1",
        "http://user:hunter2@127.0.0.1:8080/v1",
        "ftp://127.0.0.1/v1",
        "http://:8080/v1",
        "http://127.0.0.1:notaport/v1",
        "not a url",
    ],
)
def test_non_loopback_urls_are_refused_without_leaking_credentials(url: str) -> None:
    with pytest.raises(DogmaError) as refused:
        check_loopback(url)
    assert refused.value.code is ErrorCode.LLM_ENDPOINT_REFUSED
    assert "hunter2" not in refused.value.message


def test_provider_construction_refuses_before_any_http_client_exists(monkeypatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("no HTTP client may be created for a refused endpoint")

    monkeypatch.setattr(local.httpx, "Client", boom)
    with pytest.raises(DogmaError) as refused:
        LocalProvider("http://10.1.2.3:8080/v1")
    assert refused.value.code is ErrorCode.LLM_ENDPOINT_REFUSED


def test_timeouts_match_the_standard() -> None:
    assert (CONNECT_TIMEOUT_S, FIRST_TOKEN_TIMEOUT_S, TOTAL_TIMEOUT_S) == (3.0, 60.0, 180.0)
    provider = LocalProvider(BASE)
    assert provider.timeout.connect == 3.0
    assert provider.timeout.read == 60.0


def test_payload_asks_for_a_direct_streamed_answer_with_usage() -> None:
    body = build_payload(_request(max_tokens=77))

    assert body["model"] == "qwen-test"
    assert body["max_tokens"] == 77
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["messages"] == [{"role": "user", "content": "hi"}]


def test_complete_joins_the_stream_and_reads_usage() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        body = _sse(
            _delta("Hel"),
            _delta("lo"),
            json.dumps({"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 2}}),
            "[DONE]",
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    provider = LocalProvider(BASE, transport=httpx.MockTransport(handler))
    result = provider.complete(_request())

    assert result.text == "Hello"
    assert (result.tokens_in, result.tokens_out) == (11, 2)
    assert result.model == "qwen-test"
    assert seen["url"] == "http://127.0.0.1:8080/v1/chat/completions"
    assert isinstance(seen["body"], dict) and seen["body"]["stream"] is True


def test_stream_yields_text_pieces_in_order() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        body = _sse(_delta("a"), json.dumps({"choices": [{"delta": {"role": "assistant"}}]}))
        return httpx.Response(200, content=body)

    provider = LocalProvider(BASE, transport=httpx.MockTransport(handler))

    assert [chunk.text for chunk in provider.stream(_request())] == ["a"]


@pytest.mark.parametrize(
    ("error", "outcome"),
    [
        (httpx.ConnectError("connection refused"), "unavailable"),
        (httpx.ReadTimeout("no first token"), "timeout"),
    ],
)
def test_transport_failures_become_llm_unavailable(error: Exception, outcome: str) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise error

    provider = LocalProvider(BASE, transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderError) as failed:
        provider.complete(_request())

    assert failed.value.code is ErrorCode.LLM_UNAVAILABLE
    assert failed.value.outcome == outcome
    assert "\n" not in failed.value.message
    assert "httpx" not in failed.value.message.lower()


def test_an_invalid_address_is_reported_as_a_refused_endpoint() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.InvalidURL("bad address")

    provider = LocalProvider(BASE, transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderError) as failed:
        provider.complete(_request())

    assert failed.value.code is ErrorCode.LLM_ENDPOINT_REFUSED
    assert failed.value.outcome == "refused"


def test_http_error_status_is_unavailable_and_names_the_status() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"loading model")

    provider = LocalProvider(BASE, transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderError) as failed:
        provider.complete(_request())

    assert failed.value.code is ErrorCode.LLM_UNAVAILABLE
    assert "503" in failed.value.message


def test_malformed_stream_is_an_invalid_response() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse("{not json"))

    provider = LocalProvider(BASE, transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderError) as failed:
        provider.complete(_request())

    assert failed.value.code is ErrorCode.LLM_RESPONSE_INVALID
    assert failed.value.outcome == "invalid"


def test_total_deadline_stops_a_slow_stream(monkeypatch) -> None:
    clock = iter([0.0, 0.0, TOTAL_TIMEOUT_S + 1.0, TOTAL_TIMEOUT_S + 1.0])
    monkeypatch.setattr(local, "_now", lambda: next(clock))

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse(_delta("a"), _delta("b")))

    provider = LocalProvider(BASE, transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderError) as failed:
        provider.complete(_request())

    assert failed.value.outcome == "timeout"
    assert failed.value.code is ErrorCode.LLM_UNAVAILABLE


def test_parse_stream_line_ignores_role_only_deltas_and_reads_usage() -> None:
    assert parse_stream_line(json.dumps({"choices": [{"delta": {"role": "assistant"}}]})) is None
    chunk = parse_stream_line(
        json.dumps({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 4}})
    )
    assert chunk is not None
    assert (chunk.text, chunk.tokens_in, chunk.tokens_out) == ("", 3, 4)


def test_fake_provider_replays_scripted_replies_and_records_requests() -> None:
    fake = FakeProvider(
        ["first answer", ProviderError(ErrorCode.LLM_UNAVAILABLE, "unavailable", "x")]
    )

    assert fake.complete(_request()).text == "first answer"
    with pytest.raises(ProviderError):
        fake.complete(_request(max_tokens=9))
    assert [req.max_tokens for req in fake.requests] == [50, 9]
    assert [chunk.text for chunk in FakeProvider(["ab"]).stream(_request())] == ["ab"]
