"""Scoring rubrics as versioned data in ``curriculum/rubrics/`` (ARCHITECTURE §4.4).

A rubric is either a ``mock`` (parts of scored items, with score bands) or a ``topic-answer`` (a
few scored dimensions, with a ready score). Every model forbids unknown keys and is frozen, and
the cross-field rules (unique ids, part maxima, contiguous bands, anchors for 1 to 4) are checked
when a file loads, so a broken rubric fails at load time and never at scoring time.

Scores are stored with the rubric id and version they were given under, so a rubric edit never
rewrites what a learner was told before.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import Field, ValidationError, model_validator

from dogma.content.schemas import ContentModel
from dogma.core.errors import ContentError, ErrorCode

RUBRICS_DIRNAME = "rubrics"
ID_PATTERN = r"^[a-z][a-z-]*$"
ITEM_ID_PATTERN = r"^[a-z]+\.[a-z][a-z-]*$"
MINUTES_PATTERN = r"^(\d+(-\d+)?|throughout)$"
ANCHOR_LEVELS = frozenset({1, 2, 3, 4})

Kind = Literal["mock", "topic-answer"]


class Scale(ContentModel):
    """The 1-4 scale every criterion uses, with the four anchor descriptions."""

    low: Literal[1]
    high: Literal[4]
    anchors: dict[int, str]

    @model_validator(mode="after")
    def check_anchors(self) -> Self:
        _check_anchors(self.anchors, "scale")
        return self


class RubricItem(ContentModel):
    """One mock item: a paraphrased criterion, scored 1-4 on its own."""

    id: str = Field(pattern=ITEM_ID_PATTERN)
    text: str = Field(min_length=1)


class RubricPart(ContentModel):
    """A block of mock items with the minutes it is timed for and its maximum score."""

    id: str = Field(pattern=ID_PATTERN)
    title: str = Field(min_length=1)
    minutes: str = Field(pattern=MINUTES_PATTERN)
    maximum: int = Field(ge=1)
    items: tuple[RubricItem, ...] = Field(min_length=1)


class Dimension(ContentModel):
    """A topic-answer dimension, with its own four anchors."""

    id: str = Field(pattern=ID_PATTERN)
    title: str = Field(min_length=1)
    question: str = Field(min_length=1)
    anchors: dict[int, str]

    @model_validator(mode="after")
    def check_anchors(self) -> Self:
        _check_anchors(self.anchors, f"dimension {self.id}")
        return self


class Band(ContentModel):
    """A score range of a mock, inclusive at both ends, with the reading it means."""

    label: str = Field(min_length=1)
    low: int = Field(ge=0)
    high: int = Field(ge=0)
    text: str = Field(min_length=1)


class Rubric(ContentModel):
    """One rubric version. ``kind`` decides which of parts or dimensions it holds."""

    id: str = Field(pattern=ID_PATTERN)
    version: int = Field(ge=1)
    kind: Kind
    title: str = Field(min_length=1)
    scale: Scale
    ready_score: int | None = Field(default=None, ge=1, le=4)
    parts: tuple[RubricPart, ...] = ()
    dimensions: tuple[Dimension, ...] = ()
    bands: tuple[Band, ...] = ()

    @model_validator(mode="after")
    def check_shape(self) -> Self:
        if self.kind == "mock":
            _require(self.parts and self.bands, "a mock rubric needs parts and bands")
            _require(not self.dimensions, "a mock rubric has parts, not dimensions")
            _require(self.ready_score is None, "only a topic-answer rubric has a ready_score")
            _check_mock(self)
        else:
            _require(self.dimensions, "a topic-answer rubric needs dimensions")
            _require(not self.parts and not self.bands, "a topic-answer rubric has no parts")
            _require(self.ready_score is not None, "a topic-answer rubric needs a ready_score")
            _check_dimensions(self)
        return self

    @property
    def criteria_ids(self) -> tuple[str, ...]:
        """The keys a score is recorded under: item ids for a mock, dimension ids otherwise."""
        if self.kind == "mock":
            return tuple(item.id for part in self.parts for item in part.items)
        return tuple(dimension.id for dimension in self.dimensions)

    @property
    def maximum(self) -> int:
        """The highest total a complete scoring can reach."""
        if self.kind == "mock":
            return sum(part.maximum for part in self.parts)
        return self.scale.high * len(self.dimensions)


def _duplicates(values: Iterable[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_anchors(anchors: Mapping[int, str], where: str) -> None:
    if set(anchors) != ANCHOR_LEVELS:
        raise ValueError(f"{where}: anchors must describe scores 1, 2, 3 and 4")
    empty = [level for level, text in anchors.items() if not text.strip()]
    if empty:
        raise ValueError(f"{where}: anchors {empty} are empty")


def _check_mock(rubric: Rubric) -> None:
    part_ids = _duplicates(part.id for part in rubric.parts)
    if part_ids:
        raise ValueError(f"duplicate part ids: {part_ids}")
    item_ids = _duplicates(rubric.criteria_ids)
    if item_ids:
        raise ValueError(f"duplicate item ids: {item_ids}")
    for part in rubric.parts:
        expected = rubric.scale.high * len(part.items)
        if part.maximum != expected:
            raise ValueError(
                f"part {part.id} has {len(part.items)} items, so its maximum is {expected}, "
                f"not {part.maximum}"
            )
    _check_bands(rubric)


def _check_bands(rubric: Rubric) -> None:
    """Bands must start at 0, touch with no gap or overlap, and end at the maximum total."""
    expected_low = 0
    for band in rubric.bands:
        if band.low > band.high:
            raise ValueError(f"band {band.label!r} runs backwards: {band.low} to {band.high}")
        if band.low != expected_low:
            raise ValueError(f"band {band.label!r} should start at {expected_low}, not {band.low}")
        expected_low = band.high + 1
    if expected_low - 1 != rubric.maximum:
        raise ValueError(
            f"bands end at {expected_low - 1}, but the maximum total is {rubric.maximum}"
        )


def _check_dimensions(rubric: Rubric) -> None:
    dimension_ids = _duplicates(rubric.criteria_ids)
    if dimension_ids:
        raise ValueError(f"duplicate dimension ids: {dimension_ids}")
    if not rubric.ready_score or not rubric.scale.low <= rubric.ready_score <= rubric.scale.high:
        raise ValueError("ready_score must lie on the scale")


def check_scores(rubric: Rubric, scores: Mapping[str, object]) -> list[str]:
    """Why ``scores`` does not fit ``rubric``: missing, unknown or out-of-range keys. Empty if fine.

    Every criterion needs a whole-number score from 1 to 4, and no other key is allowed.
    """
    problems: list[str] = []
    expected = set(rubric.criteria_ids)
    for key in sorted(expected - set(scores)):
        problems.append(f"{key} is missing")
    for key in sorted(set(scores) - expected):
        problems.append(f"{key} is not a criterion of this rubric")
    for key in sorted(set(scores) & expected):
        value = scores[key]
        if type(value) is not int:
            problems.append(f"{key} must be a whole number from 1 to 4")
        elif not rubric.scale.low <= value <= rubric.scale.high:
            problems.append(f"{key} must be between 1 and 4")
    return problems


def load_rubric(path: Path) -> Rubric:
    """Load and validate one rubric file. Its name must be ``<id>-v<version>.yaml``."""
    text = _read(path)
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise _schema_error(path, f"invalid YAML: {' '.join(str(exc).split())}") from exc
    try:
        rubric = Rubric.model_validate(data)
    except ValidationError as exc:
        raise _schema_error(path, _reason(exc)) from exc
    except ValueError as exc:
        raise _schema_error(path, str(exc)) from exc
    if path.name != f"{rubric.id}-v{rubric.version}.yaml":
        raise _schema_error(path, f"the file name must be {rubric.id}-v{rubric.version}.yaml")
    return rubric


def load_rubric_version(curriculum_dir: Path, rubric_id: str, version: int) -> Rubric:
    """The rubric stored as ``<curriculum>/rubrics/<id>-v<version>.yaml``."""
    path = curriculum_dir / RUBRICS_DIRNAME / f"{rubric_id}-v{version}.yaml"
    if not path.is_file():
        raise ContentError(
            ErrorCode.CONTENT_NOT_FOUND, f"no rubric {rubric_id} version {version} at {path}"
        )
    return load_rubric(path)


def _read(path: Path) -> str:
    if not path.is_file():
        raise ContentError(ErrorCode.CONTENT_NOT_FOUND, f"{path}: file not found")
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise _schema_error(path, f"unreadable: {exc}") from exc


def _reason(exc: ValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors()[:3]:
        where = ".".join(str(part) for part in error["loc"]) or "<root>"
        parts.append(f"{where}: {error['msg'].removeprefix('Value error, ')}")
    return "; ".join(parts)


def _schema_error(path: Path, reason: str) -> ContentError:
    return ContentError(ErrorCode.CONTENT_SCHEMA_INVALID, f"{path}: {reason}")
