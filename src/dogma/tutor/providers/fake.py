"""A scripted provider for tests: it replays queued replies and records every request."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from dogma.tutor.models import CompletionChunk, CompletionRequest, CompletionResult, ProviderError


class FakeProvider:
    """Replies come from a queue, in order. An entry that is a ProviderError is raised instead."""

    name = "fake"

    def __init__(self, replies: Iterable[str | ProviderError] = ()) -> None:
        self._replies: list[str | ProviderError] = list(replies)
        self.requests: list[CompletionRequest] = []

    def queue(self, reply: str | ProviderError) -> None:
        self._replies.append(reply)

    def complete(self, req: CompletionRequest) -> CompletionResult:
        text = self._next(req)
        return CompletionResult(text=text, model=req.model, tokens_in=None, tokens_out=None)

    def stream(self, req: CompletionRequest) -> Iterator[CompletionChunk]:
        yield CompletionChunk(text=self._next(req))

    def _next(self, req: CompletionRequest) -> str:
        self.requests.append(req)
        if not self._replies:
            raise AssertionError("the fake provider has no scripted reply left")
        reply = self._replies.pop(0)
        if isinstance(reply, ProviderError):
            raise reply
        return reply
