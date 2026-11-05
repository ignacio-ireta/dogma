"""The ``local`` provider: an OpenAI-compatible chat endpoint on loopback, called through httpx.

Loopback is checked in the constructor, before any socket or HTTP client exists (ADR-0008).
Timeouts: connect 3 s; first token 60 s (the read timeout, which also bounds any stall between
chunks); total 180 s (a deadline checked for every chunk). ``complete`` is ``stream`` joined, so
every call gets the same three limits.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field

from dogma.core.errors import DogmaError, ErrorCode
from dogma.tutor.models import (
    CompletionChunk,
    CompletionRequest,
    CompletionResult,
    ProviderError,
)

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
CONNECT_TIMEOUT_S = 3.0
FIRST_TOKEN_TIMEOUT_S = 60.0
TOTAL_TIMEOUT_S = 180.0
_WRITE_TIMEOUT_S = 10.0
_POOL_TIMEOUT_S = 3.0
_DATA_PREFIX = "data:"
_DONE = "[DONE]"


def _now() -> float:
    return time.monotonic()


def check_loopback(base_url: str) -> None:
    """Refuse a base URL that is not http(s) to 127.0.0.1, ::1 or localhost, or has secrets."""
    try:
        parts = urlsplit(base_url)
        host = parts.hostname
        _ = parts.port
    except ValueError:
        host, parts = None, None
    if parts is None or host is None or parts.scheme not in ("http", "https"):
        raise DogmaError(
            ErrorCode.LLM_ENDPOINT_REFUSED,
            "the model URL must be http(s) to 127.0.0.1, ::1 or localhost (ADR-0008)",
        )
    if parts.username is not None or parts.password is not None:
        raise DogmaError(ErrorCode.LLM_ENDPOINT_REFUSED, "the model URL must not carry credentials")
    if host not in LOOPBACK_HOSTS:
        raise DogmaError(
            ErrorCode.LLM_ENDPOINT_REFUSED,
            f"the model URL must point at 127.0.0.1, ::1 or localhost, not {host} (ADR-0008)",
        )


def build_payload(req: CompletionRequest) -> dict[str, object]:
    """The OpenAI-compatible body. Usage is requested on the stream; reasoning stays off."""
    return {
        "model": req.model,
        "messages": [
            {"role": message.role, "content": message.content} for message in req.messages
        ],
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }


class _Delta(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    content: str | None = None


class _Choice(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    delta: _Delta = Field(default_factory=_Delta)


class _Usage(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class _Event(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    choices: tuple[_Choice, ...] = ()
    usage: _Usage | None = None


def parse_stream_line(payload: str) -> CompletionChunk | None:
    """One ``data:`` payload as a chunk. Role-only or empty events carry nothing and give None."""
    try:
        event = _Event.model_validate_json(payload)
    except ValueError as exc:
        raise ProviderError(
            ErrorCode.LLM_RESPONSE_INVALID,
            "invalid",
            "the local model sent a malformed stream event",
        ) from exc
    text = "".join(choice.delta.content or "" for choice in event.choices)
    usage = event.usage
    if not text and usage is None:
        return None
    return CompletionChunk(
        text=text,
        tokens_in=usage.prompt_tokens if usage is not None else None,
        tokens_out=usage.completion_tokens if usage is not None else None,
    )


class LocalProvider:
    """Calls ``<base_url>/chat/completions`` on loopback. Construction fails for other hosts."""

    name = "local"

    def __init__(self, base_url: str, *, transport: httpx.BaseTransport | None = None) -> None:
        check_loopback(base_url)
        self._base_url = base_url.rstrip("/")
        self._endpoint = f"{self._base_url}/chat/completions"
        self._transport = transport
        self.timeout = httpx.Timeout(
            FIRST_TOKEN_TIMEOUT_S,
            connect=CONNECT_TIMEOUT_S,
            write=_WRITE_TIMEOUT_S,
            pool=_POOL_TIMEOUT_S,
        )

    def complete(self, req: CompletionRequest) -> CompletionResult:
        parts: list[str] = []
        tokens_in: int | None = None
        tokens_out: int | None = None
        for chunk in self.stream(req):
            parts.append(chunk.text)
            if chunk.tokens_in is not None:
                tokens_in = chunk.tokens_in
            if chunk.tokens_out is not None:
                tokens_out = chunk.tokens_out
        return CompletionResult(
            text="".join(parts), model=req.model, tokens_in=tokens_in, tokens_out=tokens_out
        )

    def stream(self, req: CompletionRequest) -> Iterator[CompletionChunk]:
        started = _now()
        try:
            with (
                httpx.Client(transport=self._transport, timeout=self.timeout) as client,
                client.stream("POST", self._endpoint, json=build_payload(req)) as response,
            ):
                if response.status_code >= 400:
                    raise ProviderError(
                        ErrorCode.LLM_UNAVAILABLE,
                        "unavailable",
                        f"the local model answered HTTP {response.status_code}; check the server",
                    )
                for line in response.iter_lines():
                    if not line.startswith(_DATA_PREFIX):
                        continue
                    if _now() - started > TOTAL_TIMEOUT_S:
                        raise self._timed_out()
                    payload = line[len(_DATA_PREFIX) :].strip()
                    if payload == _DONE:
                        return
                    chunk = parse_stream_line(payload)
                    if chunk is not None:
                        yield chunk
        except httpx.TimeoutException as exc:
            raise self._timed_out() from exc
        except httpx.InvalidURL as exc:
            raise ProviderError(
                ErrorCode.LLM_ENDPOINT_REFUSED, "refused", "the model URL is not a valid address"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                ErrorCode.LLM_UNAVAILABLE,
                "unavailable",
                f"the local model server is not reachable at {self._base_url}; start it, or "
                "mount the External SSD if the model lives there",
            ) from exc

    @staticmethod
    def _timed_out() -> ProviderError:
        return ProviderError(
            ErrorCode.LLM_UNAVAILABLE,
            "timeout",
            f"the local model did not answer in time (first token {FIRST_TOKEN_TIMEOUT_S:.0f} s, "
            f"total {TOTAL_TIMEOUT_S:.0f} s)",
        )
