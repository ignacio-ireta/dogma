"""The compiled catalog served to the app: ``var/catalog.json``, built only from valid content."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime

from dogma.content.loader import TopicPackage, discover_topic_dirs, load_topic
from dogma.content.schemas import (
    ContentModel,
    Difficulty,
    Language,
    Status,
    Tier,
    TopicKind,
)
from dogma.content.validate import Issue, validate_repo
from dogma.core.config import Settings
from dogma.core.errors import ContentError, ErrorCode
from dogma.core.fsio import atomic_write


class CatalogProblem(ContentModel):
    id: str
    title: str
    difficulty: Difficulty
    minutes: int
    language: Language
    status: Status


class CatalogTopic(ContentModel):
    id: str
    title: str
    track: str
    tier: Tier
    kind: TopicKind = "topic"
    status: Status
    requires: tuple[str, ...]
    problems: tuple[CatalogProblem, ...]


class Catalog(ContentModel):
    version: Literal[1] = 1
    generated_at: AwareDatetime
    topics: tuple[CatalogTopic, ...]


def build_catalog(settings: Settings, *, now: datetime | None = None) -> Catalog:
    """Validate all content, then compile every topic package into a Catalog.

    Raises ContentError (CONTENT_SCHEMA_INVALID) listing the first issue when validation finds
    anything, so a catalog is never built from content that fails the gate. ``now`` (timezone-
    aware, for tests) defaults to the current UTC time.
    """
    issues = validate_repo(settings)
    if issues:
        raise ContentError(ErrorCode.CONTENT_SCHEMA_INVALID, _summarize(issues))
    packages = [load_topic(path) for path in discover_topic_dirs(settings.content_dir)]
    topics = sorted((_catalog_topic(pkg) for pkg in packages), key=lambda topic: topic.id)
    return Catalog(generated_at=now or datetime.now(UTC), topics=tuple(topics))


def write_catalog(catalog: Catalog, path: Path) -> bool:
    """Write the catalog atomically. Returns False and leaves the file alone when nothing changed.

    "Nothing changed" ignores ``generated_at``, so re-running the build on unchanged content is a
    no-op, as ENGINEERING_STANDARDS 4.7 requires.
    """
    payload = catalog.model_dump(mode="json", exclude={"generated_at"})
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = None
        if isinstance(existing, dict):
            previous = {key: value for key, value in existing.items() if key != "generated_at"}
            if previous == payload:
                return False
    atomic_write(path, catalog.model_dump_json(indent=2) + "\n")
    return True


def _catalog_topic(pkg: TopicPackage) -> CatalogTopic:
    return CatalogTopic(
        id=pkg.meta.id,
        title=pkg.meta.title,
        track=pkg.meta.id.split("/", 1)[0],
        tier=pkg.meta.tier,
        kind=pkg.meta.kind,
        status=pkg.meta.status,
        requires=pkg.meta.requires,
        problems=tuple(
            CatalogProblem(
                id=problem.meta.id,
                title=problem.meta.title,
                difficulty=problem.meta.difficulty,
                minutes=problem.meta.minutes,
                language=problem.meta.language,
                status=problem.meta.status,
            )
            for problem in pkg.problems
            if problem.meta is not None
        ),
    )


def _summarize(issues: list[Issue]) -> str:
    first = issues[0]
    return (
        f"content has {len(issues)} validation issue(s); first: "
        f"{first.code} {first.path}: {first.message}"
    )
