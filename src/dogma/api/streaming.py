"""Server-sent events for the tutor (ADR-0017): frames, a relay that sees the model's deltas, and
one gateway call run on a worker thread while its response streams.

The relay is an ``LLMProvider`` that the ordinary ``Gateway`` calls. The call log, the loopback
check and the reasoning-tag strip therefore still happen in the gateway, and the relay only copies
each delta out as it arrives. The final text, after the tutor's output guards, is what ``done``
carries. A client replaces what it streamed with ``done``.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Final

from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from dogma.api.errors import Problem, problem_for_code, status_for
from dogma.api.schemas.tutor import TextDeltaOut, UnavailableOut
from dogma.core.config import Settings
from dogma.core.errors import DogmaError, ErrorCode
from dogma.db.errors import StorageError
from dogma.tutor.gateway import Gateway, LLMProvider
from dogma.tutor.models import CompletionChunk, CompletionRequest, CompletionResult

log = logging.getLogger(__name__)

SSE_MEDIA_TYPE: Final = "text/event-stream"
# A model that is down or answers badly ends a stream as ``unavailable``, not as an ``error``.
MODEL_DOWN_CODES: Final = frozenset({ErrorCode.LLM_UNAVAILABLE, ErrorCode.LLM_RESPONSE_INVALID})
_THINK_OPEN: Final = "<think>"
_THINK_CLOSE: Final = "</think>"


@dataclass(frozen=True)
class Started:
    """The model accepted the request. Nothing has been streamed yet."""


@dataclass(frozen=True)
class Delta:
    """A piece of the model's raw text, before the tutor's output guards."""

    text: str


@dataclass(frozen=True)
class Finished[T]:
    """The call returned. ``value`` is what the call returned."""

    value: T


@dataclass(frozen=True)
class Failed:
    """The call raised, or the model failed while the call absorbed the failure."""

    error: Exception


type Signal[T] = Started | Delta | Finished[T] | Failed


def frame(event: str, payload: BaseModel) -> str:
    """One SSE event. The JSON is a single line, so it can never break the framing."""
    return f"event: {event}\ndata: {payload.model_dump_json()}\n\n"


def event_stream(frames: Iterator[str]) -> StreamingResponse:
    return StreamingResponse(
        frames,
        media_type=SSE_MEDIA_TYPE,
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def text_frame(event: str, text: str) -> str:
    return frame(event, TextDeltaOut(text=text))


def failure_frame(error: Exception, *, instance: str, prefix: str = "") -> str:
    """The event that ends a stream after a failure: ``unavailable`` for a model that is down."""
    if isinstance(error, DogmaError) and error.code in MODEL_DOWN_CODES:
        return frame(
            "unavailable",
            UnavailableOut(code=str(error.code), note=f"{prefix}{error.message}"),
        )
    return frame("error", _problem(error, instance))


def _problem(error: Exception, instance: str) -> Problem:
    if isinstance(error, DogmaError | StorageError):
        code = str(error.code)
        return problem_for_code(code, error.message, instance=instance, status=status_for(code))
    log.error("unhandled error in a tutor stream on %s", instance, exc_info=error)
    return problem_for_code(
        "API_INTERNAL_ERROR",
        "something went wrong on the server; the details are in the server log",
        instance=instance,
        status=500,
    )


class ThinkFilter:
    """Hides ``<think>`` reasoning from a stream, also when a tag is split across chunks.

    It mirrors ``gateway.strip_thinking`` on the joined text: a block that never closes hides
    everything after it. Only the streamed view uses this; the final text comes from the gateway.
    """

    def __init__(self) -> None:
        self._inside = False
        self._held = ""

    def feed(self, text: str) -> str:
        """The visible part of ``text``. A tag prefix at the end is held until the next chunk."""
        buffer = self._held + text
        self._held = ""
        visible: list[str] = []
        while buffer:
            tag = _THINK_CLOSE if self._inside else _THINK_OPEN
            at = buffer.find(tag)
            if at == -1:
                keep = _partial_suffix(buffer, tag)
                if not self._inside:
                    visible.append(buffer[: len(buffer) - keep])
                self._held = buffer[len(buffer) - keep :]
                break
            if not self._inside:
                visible.append(buffer[:at])
            buffer = buffer[at + len(tag) :]
            self._inside = not self._inside
        return "".join(visible)

    def finish(self) -> str:
        """The held text at the end. Inside an unclosed block it is reasoning, so it is dropped."""
        held, self._held = self._held, ""
        return "" if self._inside else held


def _partial_suffix(text: str, tag: str) -> int:
    """The length of the longest end of ``text`` that starts ``tag`` (and is shorter than it)."""
    for size in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:size]):
            return size
    return 0


class _Relay:
    """The provider the gateway calls. It forwards each delta and remembers a model failure."""

    def __init__(self, model: LLMProvider, emit: Callable[[Started | Delta], None]) -> None:
        self.name = model.name
        self.failure: DogmaError | None = None
        self._model = model
        self._emit = emit

    def complete(self, req: CompletionRequest) -> CompletionResult:
        try:
            chunks = self._model.stream(req)
            self._emit(Started())
            return self._collect(req, chunks)
        except DogmaError as exc:
            self.failure = exc
            raise

    def stream(self, req: CompletionRequest) -> Iterator[CompletionChunk]:
        return self._model.stream(req)

    def _collect(
        self, req: CompletionRequest, chunks: Iterator[CompletionChunk]
    ) -> CompletionResult:
        hidden = ThinkFilter()
        parts: list[str] = []
        tokens_in: int | None = None
        tokens_out: int | None = None
        for chunk in chunks:
            parts.append(chunk.text)
            visible = hidden.feed(chunk.text)
            if visible:
                self._emit(Delta(visible))
            if chunk.tokens_in is not None:
                tokens_in = chunk.tokens_in
            if chunk.tokens_out is not None:
                tokens_out = chunk.tokens_out
        tail = hidden.finish()
        if tail:
            self._emit(Delta(tail))
        return CompletionResult(
            text="".join(parts), model=req.model, tokens_in=tokens_in, tokens_out=tokens_out
        )


def model_signals[T](
    call: Callable[[Gateway], T], *, settings: Settings, model: LLMProvider
) -> Iterator[Signal[T]]:
    """Run ``call`` with a gateway on a worker thread, and yield its signals in order.

    The worker starts when this function is called. The iterator ends after ``Finished`` or
    ``Failed``. Until the model accepts the request, the first signal is either ``Started`` or a
    ``Failed`` (for example a refused endpoint). A failure is delivered, never swallowed.
    """
    signals: queue.Queue[Signal[T]] = queue.Queue()
    worker = threading.Thread(
        target=_work,
        args=(call, settings, model, signals),
        name="dogma-tutor-call",
        daemon=True,
    )
    worker.start()
    return _drain(signals)


def _work[T](
    call: Callable[[Gateway], T],
    settings: Settings,
    model: LLMProvider,
    signals: queue.Queue[Signal[T]],
) -> None:
    relay = _Relay(model, signals.put)
    try:
        value = call(Gateway(settings, provider=relay))
    except Exception as exc:
        # Not swallowed: the consumer re-raises it before the stream starts, or frames it.
        signals.put(Failed(exc))
        return
    if relay.failure is not None:
        # The call absorbed a model failure (the reviewed hint stood), so the stream ends as one.
        signals.put(Failed(relay.failure))
    else:
        signals.put(Finished(value))


def _drain[T](signals: queue.Queue[Signal[T]]) -> Iterator[Signal[T]]:
    while True:
        signal = signals.get()
        yield signal
        if isinstance(signal, Finished | Failed):
            return
