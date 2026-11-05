"""Narrow the syllabus to one track or tier, and read every topic package, for coverage reports.

``dogma.content.coverage`` stays pure: it counts what it is given. This module supplies the
inputs: the syllabus scoped by a filter, and the packages that load. The API's ``GET /coverage``
uses it, so the Map page counts the same topics as the coverage report.
"""

from __future__ import annotations

from pathlib import Path

from dogma.content.loader import TopicPackage, discover_topic_dirs, load_topic
from dogma.content.schemas import Syllabus, Tier, Track
from dogma.core.errors import ContentError


def scope_syllabus(
    syllabus: Syllabus, *, tier: Tier | None = None, track: str | None = None
) -> Syllabus:
    """The syllabus narrowed to one track and/or tier. Tracks left with no topics are dropped."""
    tracks: list[Track] = []
    for item in syllabus.tracks:
        if track is not None and item.code != track:
            continue
        topics = tuple(topic for topic in item.topics if tier is None or topic.tier == tier)
        if topics:
            tracks.append(item.model_copy(update={"topics": topics, "topic_count": len(topics)}))
    return syllabus.model_copy(update={"tracks": tuple(tracks)})


def read_packages(content_dir: Path) -> tuple[list[TopicPackage], list[str]]:
    """Every topic package that loads, and the ids of those that do not (not counted)."""
    packages: list[TopicPackage] = []
    unreadable: list[str] = []
    for topic_dir in discover_topic_dirs(content_dir):
        try:
            packages.append(load_topic(topic_dir))
        except ContentError:
            unreadable.append(topic_dir.relative_to(content_dir).as_posix())
    return packages, unreadable
