"""The tutor's use cases through its public functions only (ARCHITECTURE §4.5).

The tutor package is being edited in parallel, so this module depends on ``hint``, ``debug``,
``prepare_hint``, ``PreparedHint`` and ``Target`` alone. Any change to those signatures shows up
here as a type error. The ``*_events`` functions are the SSE variants (ADR-0017): they run the same
calls, so the ``done`` event is the same answer as the JSON response.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from datetime import datetime

from dogma.api.schemas.tutor import (
    HintOut,
    ReviewedEventOut,
    ReviewedHintOut,
    SourceOut,
    TutorAnswerOut,
)
from dogma.api.services.content import problem_ref
from dogma.api.streaming import (
    Delta,
    Failed,
    Finished,
    Signal,
    Started,
    failure_frame,
    frame,
    model_signals,
    text_frame,
)
from dogma.core.config import Settings
from dogma.learning import resolve_service
from dogma.tutor import service as tutor
from dogma.tutor.gateway import Gateway, LLMProvider
from dogma.tutor.models import HintAnswer, ReviewedHint, TutorAnswer
from dogma.tutor.service import PreparedHint

_NUDGE_PREFIX = "Personalised nudge needs the local model: "


def hint(
    settings: Settings,
    gateway: Gateway,
    problem_id: str,
    *,
    level: int,
    confirm_walkthrough: bool,
    now: datetime,
) -> HintOut:
    target = _target(settings, problem_id)
    answer = tutor.hint(
        settings,
        target,
        level=level,
        yes=confirm_walkthrough,
        ask=_decline,
        gateway=gateway,
    )
    _count_hint(settings, problem_id, now=now)
    return hint_out(answer)


def debug(
    settings: Settings, gateway: Gateway, problem_id: str, *, now: datetime
) -> TutorAnswerOut:
    target = _target(settings, problem_id)
    answer = tutor.debug(settings, target, gateway=gateway)
    _count_hint(settings, problem_id, now=now)
    return answer_out(answer)


def hint_events(
    settings: Settings,
    model: LLMProvider,
    problem_id: str,
    *,
    level: int,
    confirm_walkthrough: bool,
    instance: str,
    now: datetime,
) -> Iterator[str]:
    """The SSE frames of a hint. The L4 lock and unknown ids raise here, before any frame exists."""
    target = _target(settings, problem_id)
    prepared = tutor.prepare_hint(
        settings,
        target,
        level=level,
        yes=confirm_walkthrough,
        ask=_decline,
    )
    _count_hint(settings, problem_id, now=now)
    return _hint_frames(settings, model, prepared, instance=instance)


def debug_events(
    settings: Settings,
    model: LLMProvider,
    problem_id: str,
    *,
    instance: str,
    now: datetime,
) -> Iterator[str]:
    """The SSE frames of a debug answer. A refusal before any output raises here, as JSON errors."""
    target = _target(settings, problem_id)
    signals = model_signals(
        lambda gateway: tutor.debug(settings, target, gateway=gateway),
        settings=settings,
        model=model,
    )
    first = next(signals)
    if isinstance(first, Failed):
        raise first.error
    _count_hint(settings, problem_id, now=now)
    return _debug_frames(itertools.chain([first], signals), instance=instance)


def _count_hint(settings: Settings, problem_id: str, *, now: datetime) -> None:
    """A hint or debug shown during the timed re-solve of this problem counts against it.

    The CLI asks first; the browser warns first. This route never asks (ARCHITECTURE §4.4).
    """
    if resolve_service.open_window_for(settings, problem_id, now=now) is not None:
        resolve_service.record_hint(settings, problem_id, now=now)


def hint_out(answer: HintAnswer) -> HintOut:
    return HintOut(
        level=answer.level,
        reviewed=_reviewed_out(answer.reviewed),
        nudge=None if answer.nudge is None else answer_out(answer.nudge),
        nudge_note=answer.nudge_note,
    )


def answer_out(answer: TutorAnswer) -> TutorAnswerOut:
    return TutorAnswerOut(
        mode=answer.mode,
        level=answer.level,
        text=answer.text,
        sources=[
            SourceOut(tag=source.tag, citation=source.citation, kind=source.kind)
            for source in answer.sources
        ],
        cited=answer.cited,
        model=answer.model,
        prompt_version=answer.prompt_version,
        latency_ms=answer.latency_ms,
        notes=list(answer.notes),
    )


def _hint_frames(
    settings: Settings,
    model: LLMProvider,
    prepared: PreparedHint,
    *,
    instance: str,
) -> Iterator[str]:
    # The reviewed text goes out before the model is called: nothing below runs until it is sent.
    yield frame(
        "reviewed",
        ReviewedEventOut(
            level=prepared.level,
            reviewed=_reviewed_out(prepared.reviewed),
            hints_note=prepared.hints_note,
        ),
    )
    for signal in model_signals(prepared.complete, settings=settings, model=model):
        if isinstance(signal, Started):
            continue
        if isinstance(signal, Delta):
            yield text_frame("nudge", signal.text)
        elif isinstance(signal, Finished):
            yield frame("done", hint_out(signal.value))
        else:
            yield failure_frame(signal.error, instance=instance, prefix=_NUDGE_PREFIX)


def _debug_frames(signals: Iterator[Signal[TutorAnswer]], *, instance: str) -> Iterator[str]:
    for signal in signals:
        if isinstance(signal, Started):
            continue
        if isinstance(signal, Delta):
            yield text_frame("answer", signal.text)
        elif isinstance(signal, Finished):
            yield frame("done", answer_out(signal.value))
        else:
            yield failure_frame(signal.error, instance=instance)


def _reviewed_out(reviewed: ReviewedHint | None) -> ReviewedHintOut | None:
    if reviewed is None:
        return None
    return ReviewedHintOut(level=reviewed.level, label=reviewed.label, text=reviewed.text)


def _target(settings: Settings, problem_id: str) -> tutor.Target:
    ref = problem_ref(settings, problem_id)
    return tutor.Target(problem_dir=ref.dir, problem_id=ref.id, meta=ref.meta)


def _decline(question: str) -> bool:
    """The API never opens the walkthrough by asking; the request must say so explicitly."""
    return False
