"""Load the syllabus, core paths and topic packages from disk into validated models.

Missing files raise ContentError(CONTENT_NOT_FOUND). YAML and schema failures raise
ContentError(CONTENT_SCHEMA_INVALID) with the path and a readable reason. Rule violations are
not raised here; validate.py reports them as Issues. A topic whose lesson, cards, interview or
problem files are absent still loads, with that part set to None or empty, so validation can
say exactly what is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from dogma.content.markdown import Outline, parse_outline, split_front_matter
from dogma.content.pyfacts import PythonFacts, parse_python_facts
from dogma.content.schemas import (
    Card,
    CardsFile,
    CorePath,
    LessonFrontMatter,
    ProblemMeta,
    Reveal,
    RevealsFile,
    Syllabus,
    TopicMeta,
)
from dogma.core.errors import ContentError, ErrorCode

_PROBLEM_PYTHON_FILES = ("starter.py", "solution.py", "test_public.py", "test_hidden.py")


@dataclass(frozen=True)
class CasePackage:
    """The case-only parts of a ``kind: case`` package. Absent files are None (or empty)."""

    brief: str | None
    reveals: tuple[Reveal, ...] | None
    twist: str | None
    debrief: Outline | None
    # Every entry under exhibits/, relative to the package, sorted. Folders appear too, so the
    # validator can reject them.
    exhibits: tuple[str, ...]


@dataclass(frozen=True)
class ProblemPackage:
    """One ``problems/pNN-slug`` directory. ``meta`` is None when meta.yaml is absent."""

    dir: Path
    meta: ProblemMeta | None
    files: frozenset[str]
    problem: Outline | None
    hints: Outline | None
    python: dict[str, PythonFacts]


@dataclass(frozen=True)
class TopicPackage:
    """One ``content/<track>/<slug>`` directory. Absent optional parts are None."""

    dir: Path
    meta: TopicMeta
    lesson_front_matter: LessonFrontMatter | None
    lesson: Outline | None
    cards: tuple[Card, ...] | None
    interview: Outline | None
    problems: tuple[ProblemPackage, ...]
    case: CasePackage | None = None

    @property
    def interview_present(self) -> bool:
        return self.interview is not None


def load_syllabus(path: Path) -> Syllabus:
    """Load and validate ``curriculum/syllabus.yaml``."""
    return _validate_yaml(path, _read_text(path), Syllabus)


def load_core_path(path: Path) -> CorePath:
    """Load and validate one learning path file such as ``curriculum/paths/core.yaml``."""
    return _validate_yaml(path, _read_text(path), CorePath)


def load_topic(topic_dir: Path) -> TopicPackage:
    """Load a topic package. Raises ContentError if a file it does have is malformed."""
    meta = _validate_yaml(topic_dir / "topic.yaml", _read_text(topic_dir / "topic.yaml"), TopicMeta)
    front, lesson = _load_lesson(topic_dir / "lesson.md")
    cards_path = topic_dir / "cards.yaml"
    cards = (
        _validate_yaml(cards_path, _read_text(cards_path), CardsFile).cards
        if cards_path.is_file()
        else None
    )
    return TopicPackage(
        dir=topic_dir,
        meta=meta,
        lesson_front_matter=front,
        lesson=lesson,
        cards=cards,
        interview=_load_outline(topic_dir / "interview.md"),
        problems=tuple(load_problem(path) for path in _problem_dirs(topic_dir)),
        case=_load_case(topic_dir) if meta.kind == "case" else None,
    )


def _load_case(topic_dir: Path) -> CasePackage:
    reveals_path = topic_dir / "reveals.yaml"
    reveals = (
        _validate_yaml(reveals_path, _read_text(reveals_path), RevealsFile).root
        if reveals_path.is_file()
        else None
    )
    brief_path = topic_dir / "brief.md"
    twist_path = topic_dir / "twist.md"
    return CasePackage(
        brief=_read_text(brief_path) if brief_path.is_file() else None,
        reveals=reveals,
        twist=_read_text(twist_path) if twist_path.is_file() else None,
        debrief=_load_outline(topic_dir / "debrief.md"),
        exhibits=_exhibit_entries(topic_dir),
    )


def _exhibit_entries(topic_dir: Path) -> tuple[str, ...]:
    root = topic_dir / "exhibits"
    if not root.is_dir():
        return ()
    return tuple(sorted(path.relative_to(topic_dir).as_posix() for path in root.rglob("*")))


def load_problem(problem_dir: Path) -> ProblemPackage:
    """Load one problem directory: its meta, which files exist, and the structure of its text."""
    meta_path = problem_dir / "meta.yaml"
    meta = (
        _validate_yaml(meta_path, _read_text(meta_path), ProblemMeta)
        if meta_path.is_file()
        else None
    )
    files = frozenset(path.name for path in problem_dir.iterdir() if path.is_file())
    python = {
        name: parse_python_facts(_read_text(problem_dir / name))
        for name in _PROBLEM_PYTHON_FILES
        if name in files
    }
    return ProblemPackage(
        dir=problem_dir,
        meta=meta,
        files=files,
        problem=_load_outline(problem_dir / "problem.md"),
        hints=_load_outline(problem_dir / "hints.md"),
        python=python,
    )


def discover_topic_dirs(content_dir: Path) -> list[Path]:
    """Every directory under ``content_dir`` that contains a ``topic.yaml``, sorted."""
    if not content_dir.is_dir():
        raise ContentError(
            ErrorCode.CONTENT_NOT_FOUND, f"{content_dir}: content directory not found"
        )
    return sorted(path.parent for path in content_dir.rglob("topic.yaml"))


def _load_lesson(path: Path) -> tuple[LessonFrontMatter | None, Outline | None]:
    if not path.is_file():
        return None, None
    text = _read_text(path)
    try:
        raw_front, body = split_front_matter(text)
    except ValueError as exc:
        raise _schema_error(path, str(exc)) from exc
    if raw_front is None:
        raise _schema_error(path, "missing YAML front matter (a '---' block at the top)")
    return _validate_yaml(path, raw_front, LessonFrontMatter), parse_outline(body)


def _load_outline(path: Path) -> Outline | None:
    return parse_outline(_read_text(path)) if path.is_file() else None


def _problem_dirs(topic_dir: Path) -> list[Path]:
    root = topic_dir / "problems"
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise ContentError(ErrorCode.CONTENT_NOT_FOUND, f"{path}: file not found")
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise _schema_error(path, f"unreadable: {exc}") from exc


def _validate_yaml[M: BaseModel](path: Path, text: str, model: type[M]) -> M:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        reason = " ".join(str(exc).split())
        raise _schema_error(path, f"invalid YAML: {reason}") from exc
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise _schema_error(path, _pydantic_reason(exc)) from exc


def _pydantic_reason(exc: ValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors()[:3]:
        where = ".".join(str(part) for part in error["loc"]) or "<root>"
        parts.append(f"{where}: {error['msg'].removeprefix('Value error, ')}")
    return "; ".join(parts)


def _schema_error(path: Path, reason: str) -> ContentError:
    return ContentError(ErrorCode.CONTENT_SCHEMA_INVALID, f"{path}: {reason}")
