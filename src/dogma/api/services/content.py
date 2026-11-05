"""Published content for the API: the catalog, a topic's lesson, a problem's statement and starter.

Only what a learner may see is mapped. ``solution.py`` and the test files are never read here.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from dogma.api.ids import is_topic_id, problem_parts
from dogma.api.schemas.content import (
    CatalogOut,
    LessonOut,
    MinutesOut,
    ProblemOut,
    ProblemSummaryOut,
    ToleranceOut,
    TopicOut,
)
from dogma.content.catalog import Catalog, build_catalog
from dogma.content.loader import load_problem, load_topic
from dogma.content.markdown import split_front_matter
from dogma.content.resolve import ProblemRef, resolve_problem
from dogma.content.schemas import LessonFrontMatter
from dogma.core.config import Settings
from dogma.core.errors import ContentError, DogmaError, ErrorCode


def problem_ref(settings: Settings, problem_id: str) -> ProblemRef:
    """The problem a well-formed id names. Anything else is not found, never a path lookup."""
    parts = problem_parts(problem_id)
    if parts is None:
        raise _not_found(f"no problem {problem_id!r}")
    topic_id, directory_name = parts
    directory = settings.content_dir.joinpath(*topic_id.split("/"), "problems", directory_name)
    if not directory.is_dir():
        raise _not_found(f"no problem {problem_id!r}")
    return resolve_problem(settings, str(directory))


def catalog(settings: Settings, now: datetime) -> CatalogOut:
    """The compiled catalog when it reads back, else one built from valid content (not written)."""
    compiled = _read_compiled(settings.var_dir / "catalog.json")
    source = compiled if compiled is not None else build_catalog(settings, now=now)
    return CatalogOut.model_validate(source.model_dump(mode="json"))


def topic(settings: Settings, topic_id: str) -> TopicOut:
    if not is_topic_id(topic_id):
        raise _not_found(f"no topic {topic_id!r}")
    directory = settings.content_dir.joinpath(*topic_id.split("/"))
    if not (directory / "topic.yaml").is_file():
        raise _not_found(f"no topic {topic_id!r}")
    package = load_topic(directory)
    meta = package.meta
    lesson = _lesson(package.dir, package.lesson_front_matter)
    return TopicOut(
        id=meta.id,
        track=meta.id.split("/", 1)[0],
        title=meta.title,
        tier=meta.tier,
        status=meta.status,
        requires=list(meta.requires),
        aliases=list(meta.aliases),
        objectives=list(meta.objectives),
        minutes=MinutesOut(
            lesson=meta.minutes.lesson,
            practice=meta.minutes.practice,
            min_viable=meta.minutes.min_viable,
        ),
        interview_angle=meta.interview_angle,
        sources=list(meta.sources),
        lesson=lesson,
        cards_count=len(package.cards or ()),
        interview_present=package.interview_present,
        problems=[
            ProblemSummaryOut(
                id=f"{meta.id}/{problem.dir.name}",
                title=problem.meta.title,
                difficulty=problem.meta.difficulty,
                minutes=problem.meta.minutes,
                language=problem.meta.language,
                status=problem.meta.status,
                core=problem.meta.core,
            )
            for problem in package.problems
            if problem.meta is not None
        ],
    )


def problem(settings: Settings, problem_id: str) -> ProblemOut:
    ref = problem_ref(settings, problem_id)
    package = load_problem(ref.dir)
    meta = ref.meta
    hints_count = len(package.hints.sections) if package.hints is not None else 0
    return ProblemOut(
        id=ref.id,
        topic_id=ref.id.rsplit("/", 1)[0],
        track=ref.id.split("/", 1)[0],
        version=meta.version,
        title=meta.title,
        language=meta.language,
        runtime=meta.runtime,
        entry=meta.entry,
        difficulty=meta.difficulty,
        minutes=meta.minutes,
        core=meta.core,
        status=meta.status,
        timeout_s=meta.timeout_s,
        tolerance=ToleranceOut(atol=meta.tolerance.atol, rtol=meta.tolerance.rtol),
        banned_imports=list(meta.banned_imports),
        problem_md=_read_optional(ref.problem_md),
        starter=_read_optional(ref.starter),
        hints_count=hints_count,
    )


def _lesson(topic_dir: Path, front: LessonFrontMatter | None) -> LessonOut | None:
    path = topic_dir / "lesson.md"
    if front is None or not path.is_file():
        return None
    _, body = split_front_matter(path.read_text(encoding="utf-8"))
    return LessonOut(
        title=front.title, minutes=front.minutes, min_viable=front.min_viable, markdown=body
    )


def _read_compiled(path: Path) -> Catalog | None:
    if not path.is_file():
        return None
    try:
        return Catalog.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError):
        return None


def _read_optional(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.is_file() else None


def _not_found(message: str) -> DogmaError:
    return ContentError(ErrorCode.CONTENT_NOT_FOUND, message)
