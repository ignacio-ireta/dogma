"""Coverage of the syllabus by authored content. Pure data: nothing is printed here."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from dogma.content.loader import TopicPackage
from dogma.content.schemas import ContentModel, Syllabus, SyllabusTopic, Tier, TopicKind

TIERS: tuple[Tier, ...] = ("A", "B", "C")


class CoverageRow(ContentModel):
    """One syllabus topic and what has been authored for it."""

    id: str
    track: str
    tier: Tier
    title: str
    kind: TopicKind
    has_package: bool
    has_lesson: bool
    has_cards: bool
    has_interview: bool
    n_problems: int
    status: str  # the topic's status, or "missing" when no package exists yet


class CoverageSummary(ContentModel):
    """Counts for a group of rows (a track, or a tier)."""

    key: str
    topics: int
    with_package: int
    with_lesson: int
    with_cards: int
    with_interview: int
    with_problems: int
    # Case topics in the group, and how many of them have a package (a case is authored content).
    cases: int
    with_cases: int
    statuses: dict[str, int]


class CoverageReport(ContentModel):
    rows: tuple[CoverageRow, ...]
    by_track: tuple[CoverageSummary, ...]
    by_tier: tuple[CoverageSummary, ...]


def coverage(syllabus: Syllabus, packages: Iterable[TopicPackage]) -> CoverageReport:
    """One row per syllabus topic, in syllabus order, plus per-track and per-tier summaries.

    Packages whose id is not in the syllabus are ignored; validation reports them instead.
    """
    by_id = {pkg.meta.id: pkg for pkg in packages}
    rows = tuple(
        _row(track.code, topic, by_id.get(topic.id))
        for track in syllabus.tracks
        for topic in track.topics
    )
    return CoverageReport(
        rows=rows,
        by_track=tuple(
            _summarize(track.code, [row for row in rows if row.track == track.code])
            for track in syllabus.tracks
        ),
        by_tier=tuple(
            _summarize(tier, [row for row in rows if row.tier == tier]) for tier in TIERS
        ),
    )


def _row(track: str, topic: SyllabusTopic, pkg: TopicPackage | None) -> CoverageRow:
    topic_id, tier, title, kind = topic.id, topic.tier, topic.title, topic.kind
    if pkg is None:
        return CoverageRow(
            id=topic_id,
            track=track,
            tier=tier,
            title=title,
            kind=kind,
            has_package=False,
            has_lesson=False,
            has_cards=False,
            has_interview=False,
            n_problems=0,
            status="missing",
        )
    return CoverageRow(
        id=topic_id,
        track=track,
        tier=tier,
        title=title,
        kind=kind,
        has_package=True,
        has_lesson=pkg.lesson is not None,
        has_cards=bool(pkg.cards),
        has_interview=pkg.interview_present,
        n_problems=len(pkg.problems),
        status=pkg.meta.status,
    )


def _summarize(key: str, rows: list[CoverageRow]) -> CoverageSummary:
    return CoverageSummary(
        key=key,
        topics=len(rows),
        with_package=sum(row.has_package for row in rows),
        with_lesson=sum(row.has_lesson for row in rows),
        with_cards=sum(row.has_cards for row in rows),
        with_interview=sum(row.has_interview for row in rows),
        with_problems=sum(row.n_problems > 0 for row in rows),
        cases=sum(row.kind == "case" for row in rows),
        with_cases=sum(row.kind == "case" and row.has_package for row in rows),
        statuses=dict(Counter(row.status for row in rows)),
    )
