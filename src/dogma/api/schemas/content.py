"""Catalog, topic and problem reads: what the Map and Problem pages show."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from dogma.content.schemas import Difficulty, Language, Status, Tier


class ToleranceOut(BaseModel):
    atol: float
    rtol: float


class CatalogProblemOut(BaseModel):
    id: str
    title: str
    difficulty: Difficulty
    minutes: int
    language: Language
    status: Status


class CatalogTopicOut(BaseModel):
    id: str
    title: str
    track: str
    tier: Tier
    status: Status
    requires: list[str]
    problems: list[CatalogProblemOut]


class CatalogOut(BaseModel):
    version: Literal[1]
    generated_at: datetime
    topics: list[CatalogTopicOut]


class MinutesOut(BaseModel):
    lesson: int
    practice: int
    min_viable: int


class LessonOut(BaseModel):
    title: str
    minutes: int
    min_viable: str
    markdown: str


class ProblemSummaryOut(BaseModel):
    id: str
    title: str
    difficulty: Difficulty
    minutes: int
    language: Language
    status: Status
    core: bool


class TopicOut(BaseModel):
    id: str
    track: str
    title: str
    tier: Tier
    status: Status
    requires: list[str]
    aliases: list[str]
    objectives: list[str]
    minutes: MinutesOut
    interview_angle: str
    sources: list[str]
    lesson: LessonOut | None
    cards_count: int
    interview_present: bool
    problems: list[ProblemSummaryOut]


class ProblemOut(BaseModel):
    """A problem's statement and starter. Solutions and test files are never part of it."""

    model_config = ConfigDict(frozen=True)

    id: str
    topic_id: str
    track: str
    version: int
    title: str
    language: Language
    runtime: str
    entry: str
    difficulty: Difficulty
    minutes: int
    core: bool
    status: Status
    timeout_s: int
    tolerance: ToleranceOut
    banned_imports: list[str]
    problem_md: str | None
    starter: str | None
    hints_count: int
