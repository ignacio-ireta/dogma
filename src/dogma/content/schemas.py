"""Pydantic models for every content file: syllabus, core path, topic, lesson, cards, problem.

Shapes follow docs/CONTENT_SPEC.md. Every model forbids unknown keys and is frozen. Shape and
type errors fail here (the loader raises ContentError); cross-file rules live in validate.py
and are reported as Issues.
"""

from __future__ import annotations

import keyword
import re
from collections import Counter
from collections.abc import Iterable
from datetime import date
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator, model_validator

Tier = Literal["A", "B", "C"]
Status = Literal["draft", "verified", "reviewed", "retired"]
Language = Literal["python", "sql"]
Difficulty = Literal["easy", "medium", "hard"]
# topic: lesson, cards and 1-3 auto-graded problems. case: a system-design case package with no
# problems (CONTENT_SPEC section 9). Existing topics omit kind and are topics.
TopicKind = Literal["topic", "case"]

TOPIC_ID_PATTERN = r"^[a-z0-9]+/[a-z0-9][a-z0-9-]*$"
TRACK_CODE_PATTERN = r"^[a-z0-9]+$"
PROBLEM_DIR_PATTERN = r"^p\d{2}-[a-z0-9-]+$"
SOURCE_PATTERN = r"^(book|paper|library|source|corpus): \S"
EXHIBIT_PATTERN = r"^exhibits/[a-z0-9][a-z0-9._-]*\.(png|csv)$"
CLOZE_MARKER = "{{c1::"

_TOPIC_ID = re.compile(TOPIC_ID_PATTERN)
_SOURCE = re.compile(SOURCE_PATTERN)


class ContentModel(BaseModel):
    """Base for every content model: unknown keys are errors and instances are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _duplicate_values(values: Iterable[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


# --- curriculum/syllabus.yaml -------------------------------------------------------------


class SyllabusTopic(ContentModel):
    id: str = Field(pattern=TOPIC_ID_PATTERN)
    title: str = Field(min_length=1)
    tier: Tier
    kind: TopicKind = "topic"


class Track(ContentModel):
    code: str = Field(pattern=TRACK_CODE_PATTERN)
    order: int = Field(ge=1)
    title: str = Field(min_length=1)
    origin: str = Field(min_length=1)
    topic_count: int = Field(ge=0)
    topics: tuple[SyllabusTopic, ...]

    @model_validator(mode="after")
    def check_counts_and_prefixes(self) -> Self:
        if self.topic_count != len(self.topics):
            raise ValueError(
                f"track {self.code}: topic_count {self.topic_count} != {len(self.topics)} topics"
            )
        for topic in self.topics:
            if not topic.id.startswith(f"{self.code}/"):
                raise ValueError(f"topic {topic.id} must start with '{self.code}/'")
        return self


class Syllabus(ContentModel):
    version: int = Field(ge=1)
    source: str = Field(min_length=1)
    tracks: tuple[Track, ...]

    @model_validator(mode="after")
    def check_unique_ids(self) -> Self:
        codes = _duplicate_values(track.code for track in self.tracks)
        if codes:
            raise ValueError(f"duplicate track codes: {codes}")
        ids = _duplicate_values(topic.id for track in self.tracks for topic in track.topics)
        if ids:
            raise ValueError(f"duplicate topic ids: {ids}")
        return self

    def topics_by_id(self) -> dict[str, SyllabusTopic]:
        """Every syllabus topic keyed by its id."""
        return {topic.id: topic for track in self.tracks for topic in track.topics}


# --- curriculum/paths/*.yaml --------------------------------------------------------------


class PathWeek(ContentModel):
    week: int = Field(ge=1)
    starts: date
    theme: str = Field(min_length=1)
    topics: tuple[str, ...]


class CorePath(ContentModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    start: date
    end: date
    sessions_per_day: int = Field(ge=1)
    session_minutes: int = Field(ge=1)
    days_per_week: int = Field(ge=1, le=7)
    weeks: tuple[PathWeek, ...]

    @model_validator(mode="after")
    def check_dates(self) -> Self:
        if self.end < self.start:
            raise ValueError(f"path ends ({self.end}) before it starts ({self.start})")
        return self


# --- content/<track>/<slug>/topic.yaml ----------------------------------------------------


class TopicMinutes(ContentModel):
    lesson: int = Field(ge=1)
    practice: int = Field(ge=1)
    min_viable: int = Field(ge=1)


class TopicMeta(ContentModel):
    id: str = Field(pattern=TOPIC_ID_PATTERN)
    title: str = Field(min_length=1)
    tier: Tier
    kind: TopicKind = "topic"
    requires: tuple[str, ...]
    aliases: tuple[str, ...]
    objectives: tuple[str, ...] = Field(min_length=2, max_length=4)
    minutes: TopicMinutes
    interview_angle: str = Field(min_length=1)
    sources: tuple[str, ...] = Field(min_length=1)
    status: Status

    @field_validator("requires")
    @classmethod
    def requires_are_topic_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        bad = [item for item in value if not _TOPIC_ID.fullmatch(item)]
        if bad:
            raise ValueError(f"requires entries must be topic ids: {bad}")
        return value

    @field_validator("sources")
    @classmethod
    def sources_use_known_formats(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        bad = [item for item in value if not _SOURCE.match(item)]
        if bad:
            raise ValueError(
                f"sources must start with book:, paper:, library:, source: or corpus: {bad}"
            )
        return value


# --- content/<track>/<slug>/lesson.md front matter ----------------------------------------


class LessonFrontMatter(ContentModel):
    title: str = Field(min_length=1)
    minutes: int = Field(ge=1)
    min_viable: str = Field(min_length=1)


# --- content/<track>/<slug>/reveals.yaml (case packages only) -------------------------------


class Reveal(ContentModel):
    """A fact the interviewer releases only when the candidate asks for it."""

    when_asked: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    exhibit: str | None = Field(default=None, pattern=EXHIBIT_PATTERN)


class RevealsFile(RootModel[tuple[Reveal, ...]]):
    """reveals.yaml is a YAML list of reveals at the top level."""

    model_config = ConfigDict(frozen=True)


# --- content/<track>/<slug>/cards.yaml ----------------------------------------------------


class QACard(ContentModel):
    id: str = Field(min_length=1)
    type: Literal["qa"]
    front: str = Field(min_length=1)
    back: str = Field(min_length=1)


class ClozeCard(ContentModel):
    id: str = Field(min_length=1)
    type: Literal["cloze"]
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def has_cloze_marker(cls, value: str) -> str:
        if CLOZE_MARKER not in value:
            raise ValueError(f"cloze text must contain {CLOZE_MARKER}")
        return value


Card = Annotated[QACard | ClozeCard, Field(discriminator="type")]


class CardsFile(ContentModel):
    cards: tuple[Card, ...]


# --- problems/<pNN-slug>/meta.yaml --------------------------------------------------------


class Tolerance(ContentModel):
    atol: float = Field(ge=0)
    rtol: float = Field(ge=0)


class ProblemMeta(ContentModel):
    id: str = Field(min_length=1)
    version: int = Field(ge=1)
    title: str = Field(min_length=1)
    language: Language
    runtime: str = Field(min_length=1)
    entry: str = Field(min_length=1)
    difficulty: Difficulty
    minutes: int = Field(ge=1)
    core: bool
    tolerance: Tolerance
    banned_imports: tuple[str, ...]
    timeout_s: int = Field(ge=1)
    status: Status

    @model_validator(mode="after")
    def check_entry(self) -> Self:
        if self.language == "python":
            if not self.entry.isidentifier() or keyword.iskeyword(self.entry):
                raise ValueError(f"python entry must be a function or class name: {self.entry!r}")
        elif self.entry != "query":
            raise ValueError(f"sql entry must be 'query', got {self.entry!r}")
        return self


# --- problems/<pNN-slug>/variants/manifest.yaml -------------------------------------------

VARIANT_ACCEPT_PATTERN = r"^accept_[a-z0-9_]+\.(py|sql)$"
VARIANT_REJECT_PATTERN = r"^reject_[a-z0-9_]+\.(py|sql)$"
TEST_FUNCTION_PATTERN = r"^test_[A-Za-z0-9_]+$"


class AcceptVariant(ContentModel):
    file: str = Field(pattern=VARIANT_ACCEPT_PATTERN)
    origin: Literal["blind", "freedom"]
    freedom: str | None = None
    clause: str | None = None

    @model_validator(mode="after")
    def check_origin(self) -> Self:
        if self.origin == "blind":
            if self.file.rsplit(".", 1)[0] != "accept_blind":
                raise ValueError(f"a blind accept must be named accept_blind: {self.file!r}")
            if self.freedom is not None:
                raise ValueError("a blind accept must not set freedom")
        elif not self.freedom:
            raise ValueError("a freedom accept must say what it does differently (freedom)")
        return self


class RejectVariant(ContentModel):
    file: str = Field(pattern=VARIANT_REJECT_PATTERN)
    mistake: str = Field(min_length=1)
    clause: str = Field(min_length=1)
    killed_by: str = Field(pattern=TEST_FUNCTION_PATTERN)
    visible: Literal["public", "hidden"]
    message_has: str | None = None


class UncoveredClause(ContentModel):
    clause: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class VariantsManifest(ContentModel):
    problem_version: int = Field(ge=1)
    accept: tuple[AcceptVariant, ...] = ()
    reject: tuple[RejectVariant, ...] = ()
    uncovered: tuple[UncoveredClause, ...] = ()
    sweeps: tuple[Literal["reject", "mutants"], ...] = ()

    @model_validator(mode="after")
    def check_manifest(self) -> Self:
        files = [variant.file for variant in self.accept] + [
            variant.file for variant in self.reject
        ]
        duplicates = _duplicate_values(files)
        if duplicates:
            raise ValueError(f"a variant file is listed more than once: {duplicates}")
        blind = sum(1 for variant in self.accept if variant.origin == "blind")
        if blind > 1:
            raise ValueError(f"at most one blind accept is allowed; found {blind}")
        repeated = _duplicate_values(self.sweeps)
        if repeated:
            raise ValueError(f"a sweep is listed more than once: {repeated}")
        return self
