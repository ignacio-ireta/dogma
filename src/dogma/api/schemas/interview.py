"""Interview answers and rubrics: a topic's questions, the two rubrics and self-scored answers.

A rubric is shown with its anchors, so the learner scores against the words, not a memory of them.
Scores are whole numbers from 1 to 4. ``mastery`` is the topic's state as it was read, and the
answer response says what the answer did to it (``before``, ``after``, ``reason``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt

RubricId = Literal["topic-answer", "mock"]
MAX_ANSWER_TEXT = 8000
MAX_ANSWER_NOTES = 1000


class AnchorOut(BaseModel):
    """What one score (1 to 4) means on one criterion."""

    level: int
    text: str


class DimensionOut(BaseModel):
    """A topic-answer dimension: its question for the learner and its four anchors."""

    id: str
    title: str
    question: str
    anchors: list[AnchorOut]


class ItemOut(BaseModel):
    """One mock item, scored 1 to 4 on its own."""

    id: str
    text: str


class PartOut(BaseModel):
    """A block of mock items: its title, the minutes it is timed for and its maximum score."""

    id: str
    title: str
    minutes: str
    maximum: int
    items: list[ItemOut]


class BandOut(BaseModel):
    """An interpretation band of the mock total: an inclusive range and what it means."""

    label: str
    low: int
    high: int
    text: str


class RubricOut(BaseModel):
    """One rubric version as the browser shows it. ``anchors`` is the scale shared by every item.

    A topic-answer rubric has ``dimensions`` and a ``ready_score``; a mock has ``parts`` and
    ``bands``. ``criteria`` lists the keys a score is recorded under, in rubric order.
    """

    id: str
    version: int
    kind: Literal["mock", "topic-answer"]
    title: str
    anchors: list[AnchorOut]
    ready_score: int | None
    maximum: int
    criteria: list[str]
    dimensions: list[DimensionOut]
    parts: list[PartOut]
    bands: list[BandOut]


class InterviewQuestionOut(BaseModel):
    """A numbered question from interview.md. Its 60-second answer is not sent: you write it."""

    number: int
    title: str


class InterviewOut(BaseModel):
    """A topic's interview questions, its mastery and the topic-answer rubric to score against."""

    topic_id: str
    title: str
    mastery: str
    questions: list[InterviewQuestionOut]
    rubric: RubricOut


class AnswerRequest(BaseModel):
    """One answer to one question. Give ``text`` (written) or ``spoken`` (said aloud, no text kept).

    ``scores`` must hold every dimension id, each a whole number from 1 to 4. Unknown fields are a
    422, as is a missing text when the answer is not spoken.
    """

    model_config = ConfigDict(extra="forbid")

    question: int = Field(default=1, ge=1, le=999)
    text: str | None = Field(default=None, max_length=MAX_ANSWER_TEXT)
    spoken: bool = False
    scores: dict[str, StrictInt]
    notes: str | None = Field(default=None, max_length=MAX_ANSWER_NOTES)


class AnswerOut(BaseModel):
    """What the stored answer did. ``promoted`` is true when the topic moved to interview-ready."""

    answer_id: str
    topic_id: str
    prompt_ref: str
    scores: dict[str, int]
    spoken: bool
    before: str
    after: str
    promoted: bool
    ready_at: datetime | None
    reason: str


class WeekInterviewReadyOut(BaseModel):
    """How many of one study week's topics are interview-ready."""

    week: int
    ready: int
    total: int


class InterviewReadyOut(BaseModel):
    """Interview-ready counts across the core path, and per study week. A count, never a streak."""

    ready: int
    total: int
    weeks: list[WeekInterviewReadyOut]
