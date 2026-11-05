"""The coverage report for the Map page: the scoping and counting of ``dogma content coverage``."""

from __future__ import annotations

from dogma.api.schemas.coverage import CoverageOut, CoverageRowOut, CoverageSummaryOut
from dogma.content.coverage import CoverageRow, CoverageSummary, coverage
from dogma.content.coverage_scope import read_packages, scope_syllabus
from dogma.content.loader import load_syllabus
from dogma.content.schemas import Tier
from dogma.core.config import Settings
from dogma.core.errors import ContentError, ErrorCode


def report(settings: Settings, *, tier: Tier | None, track: str | None) -> CoverageOut:
    """Per-track and per-tier counts. Topic rows come only with a filter (500+ topics)."""
    syllabus = load_syllabus(settings.syllabus_path)
    titles = {item.code: item.title for item in syllabus.tracks}
    if track is not None and track not in titles:
        raise ContentError(
            ErrorCode.CONTENT_NOT_FOUND,
            f"no track {track!r}; the tracks are {', '.join(sorted(titles))}",
        )
    packages, unreadable = read_packages(settings.content_dir)
    result = coverage(scope_syllabus(syllabus, tier=tier, track=track), packages)
    filtered = tier is not None or track is not None
    return CoverageOut(
        tier=tier,
        track=track,
        by_track=[_summary_out(item, titles.get(item.key)) for item in result.by_track],
        by_tier=[_summary_out(item, None) for item in result.by_tier],
        rows=[_row_out(row) for row in result.rows] if filtered else None,
        unreadable_topics=unreadable,
    )


def _summary_out(summary: CoverageSummary, title: str | None) -> CoverageSummaryOut:
    return CoverageSummaryOut(
        key=summary.key,
        title=title,
        topics=summary.topics,
        with_package=summary.with_package,
        with_lesson=summary.with_lesson,
        with_cards=summary.with_cards,
        with_interview=summary.with_interview,
        with_problems=summary.with_problems,
        cases=summary.cases,
        with_cases=summary.with_cases,
        statuses=summary.statuses,
    )


def _row_out(row: CoverageRow) -> CoverageRowOut:
    return CoverageRowOut(
        id=row.id,
        track=row.track,
        tier=row.tier,
        title=row.title,
        kind=row.kind,
        has_package=row.has_package,
        has_lesson=row.has_lesson,
        has_cards=row.has_cards,
        has_interview=row.has_interview,
        n_problems=row.n_problems,
        status=row.status,
    )
