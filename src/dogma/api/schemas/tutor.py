"""Tutor requests, answers and the payloads of the SSE variants (ADR-0017).

The JSON answers are the contract for the final result. Each stream ends with a ``done`` event whose
payload is exactly that JSON answer, so a client that ignores the deltas still gets the same result.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from dogma.tutor.models import Mode, SourceKind


class HintRequest(BaseModel):
    problem_id: str
    level: int = Field(default=1, ge=1, le=4)
    confirm_walkthrough: bool = False


class DebugRequest(BaseModel):
    problem_id: str


class SourceOut(BaseModel):
    tag: str
    citation: str
    kind: SourceKind


class ReviewedHintOut(BaseModel):
    level: int
    label: str
    text: str


class TutorAnswerOut(BaseModel):
    mode: Mode
    level: int | None
    text: str
    sources: list[SourceOut]
    cited: bool
    model: str
    prompt_version: int
    latency_ms: int
    notes: list[str]


class HintOut(BaseModel):
    """The reviewed hint for the level (when hints.md has one), then the model's nudge if any."""

    level: int
    reviewed: ReviewedHintOut | None
    nudge: TutorAnswerOut | None
    nudge_note: str | None


class ReviewedEventOut(BaseModel):
    """The first event of a hint stream: the reviewed text, sent before any model call."""

    level: int
    reviewed: ReviewedHintOut | None
    hints_note: str | None


class TextDeltaOut(BaseModel):
    """A piece of the model's text as it arrives. It is raw: ``done`` carries the guarded text."""

    text: str


class UnavailableOut(BaseModel):
    """The model did not answer. This ends the stream; any reviewed text already sent stands."""

    code: str
    note: str
