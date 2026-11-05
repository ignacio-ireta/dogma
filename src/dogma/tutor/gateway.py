"""The one door to a model (ENGINEERING_STANDARDS §10). Every call appends one metadata line.

The log holds ts, mode, level, problem id, provider, model, prompt version and hash, token counts,
latency and outcome. It never holds prompt or answer text. A refused endpoint, a failed call and an
empty answer are logged too, with their outcome.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from dogma.core.config import Settings
from dogma.core.errors import DogmaError, ErrorCode
from dogma.core.fsio import atomic_write
from dogma.tutor.models import (
    CallMode,
    CompletionChunk,
    CompletionRequest,
    CompletionResult,
    ProviderError,
    RenderedPrompt,
)
from dogma.tutor.providers.local import LocalProvider

CALL_LOG_NAME = "llm_calls.jsonl"
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_OPEN_THINK = re.compile(r"<think>.*\Z", re.DOTALL)


class LLMProvider(Protocol):
    """What a provider offers: a blocking completion and a stream of chunks (ARCHITECTURE §4.5)."""

    name: str

    def complete(self, req: CompletionRequest) -> CompletionResult: ...

    def stream(self, req: CompletionRequest) -> Iterator[CompletionChunk]: ...


@dataclass(frozen=True)
class CallMeta:
    """What the call log records about a call, besides the outcome."""

    mode: CallMode
    prompt: RenderedPrompt
    max_tokens: int
    problem_id: str | None = None


def call_log_path(settings: Settings) -> Path:
    return settings.var_dir / CALL_LOG_NAME


def strip_thinking(text: str) -> str:
    """Remove reasoning blocks some models emit. An unclosed block is reasoning that never ended."""
    return _OPEN_THINK.sub("", _THINK_BLOCK.sub("", text))


class Gateway:
    """Sends prompts to the configured provider. The default is the loopback local provider."""

    def __init__(self, settings: Settings, provider: LLMProvider | None = None) -> None:
        self._settings = settings
        self._provider = provider

    def complete(self, meta: CallMeta) -> CompletionResult:
        started = time.monotonic()
        provider_name = self._provider.name if self._provider is not None else LocalProvider.name
        try:
            result = self._call(meta)
        except DogmaError as exc:
            outcome = _failure_outcome(exc)
            self._record(meta, provider_name, _elapsed_ms(started), outcome, None)
            raise
        latency = _elapsed_ms(started)
        result = replace(result, latency_ms=latency)
        self._record(meta, provider_name, latency, "ok", result)
        return result

    def _call(self, meta: CallMeta) -> CompletionResult:
        provider = (
            self._provider
            if self._provider is not None
            else LocalProvider(self._settings.llm_base_url)
        )
        request = CompletionRequest(
            model=self._settings.llm_model,
            messages=meta.prompt.messages,
            max_tokens=meta.max_tokens,
        )
        result = provider.complete(request)
        text = strip_thinking(result.text).strip()
        if not text:
            raise ProviderError(
                ErrorCode.LLM_RESPONSE_INVALID,
                "invalid",
                "the local model returned an empty answer; try again",
            )
        return replace(result, text=text)

    def _record(
        self,
        meta: CallMeta,
        provider: str,
        latency_ms: int,
        outcome: str,
        result: CompletionResult | None,
    ) -> None:
        row: dict[str, object] = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "mode": meta.mode.value,
            "level": meta.prompt.level,
            "problem": meta.problem_id,
            "provider": provider,
            "model": self._settings.llm_model,
            "prompt_version": meta.prompt.version,
            "prompt_sha256": meta.prompt.sha256,
            "tokens_in": result.tokens_in if result is not None else None,
            "tokens_out": result.tokens_out if result is not None else None,
            "latency_ms": latency_ms,
            "outcome": outcome,
        }
        append_log_line(call_log_path(self._settings), json.dumps(row, sort_keys=True))


def append_log_line(path: Path, line: str) -> None:
    """Append one line atomically (ENGINEERING §4.7). A log that cannot be written is an error."""
    try:
        existing = path.read_text(encoding="utf-8") if path.is_file() else ""
        atomic_write(path, f"{existing}{line}\n")
    except (OSError, ValueError) as exc:
        raise DogmaError(
            ErrorCode.LLM_LOG_FAILED, f"cannot append to {path.name}: {exc.__class__.__name__}"
        ) from exc


def _failure_outcome(exc: DogmaError) -> str:
    if isinstance(exc, ProviderError):
        return exc.outcome
    if exc.code is ErrorCode.LLM_ENDPOINT_REFUSED:
        return "refused"
    return "error"


def _elapsed_ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)
