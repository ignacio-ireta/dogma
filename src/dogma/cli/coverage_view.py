"""``dogma content coverage``: what has been authored for each part of the syllabus.

The numbers come from ``dogma.content.coverage`` (pure data), scoped by ``--tier`` or ``--track``
with ``dogma.content.coverage_scope``, the same helpers as the API's ``GET /coverage``. This module
only renders the result as Rich tables. Per-topic rows appear only when a filter is given, because
the full syllabus has more than 500 topics.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

import typer
from rich.markup import escape
from rich.table import Table

from dogma.cli.context import CliError, err, out, settings_of
from dogma.content.coverage import TIERS, CoverageRow, CoverageSummary, coverage
from dogma.content.coverage_scope import read_packages, scope_syllabus
from dogma.content.loader import load_syllabus
from dogma.content.schemas import Tier

_TIER_BY_CODE: dict[str, Tier] = {code: code for code in TIERS}
_STATUS_ORDER = ("reviewed", "verified", "draft", "retired", "missing")


def coverage_command(
    ctx: typer.Context,
    tier: Annotated[
        str | None, typer.Option("--tier", help="Only topics of this tier: A, B or C.")
    ] = None,
    track: Annotated[
        str | None, typer.Option("--track", help="Only this track code, e.g. cml or sql.")
    ] = None,
) -> None:
    """Show how much of the syllabus has content: per track, per tier, and per topic."""
    settings = settings_of(ctx)
    tier_code = _TIER_BY_CODE.get(tier.upper()) if tier is not None else None
    if tier is not None and tier_code is None:
        raise CliError(f"--tier must be A, B or C, not {tier!r}")
    syllabus = load_syllabus(settings.syllabus_path)
    codes = sorted(item.code for item in syllabus.tracks)
    if track is not None and track not in codes:
        raise CliError(f"no track {track!r}; the tracks are {', '.join(codes)}")
    packages, unreadable = read_packages(settings.content_dir)
    scoped = scope_syllabus(syllabus, tier=tier_code, track=track)
    report = coverage(scoped, packages)
    console = out()
    titles = {item.code: item.title for item in scoped.tracks}
    console.print(_summary_table("By track", report.by_track, titles))
    if tier_code is None:
        console.print(_summary_table("By tier", report.by_tier, {}))
    if tier_code is not None or track is not None:
        console.print(_topic_table(report.rows))
    for topic_id in unreadable:
        err().print(
            f"warning: package not counted: {escape(topic_id)} "
            f"(see why: dogma content verify {escape(topic_id)})",
            soft_wrap=True,
        )


def _summary_table(title: str, rows: Sequence[CoverageSummary], titles: dict[str, str]) -> Table:
    table = Table(title=title, title_justify="left")
    for name in ("Key", "Topics", "Package", "Lesson", "Cards", "Interview", "Problems", "Cases"):
        table.add_column(name, justify="left" if name == "Key" else "right")
    table.add_column("Status")
    for row in rows:
        total = row.topics
        table.add_row(
            escape(row.key) + (f" {escape(titles[row.key])}" if row.key in titles else ""),
            str(total),
            _fraction(row.with_package, total),
            _fraction(row.with_lesson, total),
            _fraction(row.with_cards, total),
            _fraction(row.with_interview, total),
            _fraction(row.with_problems, total),
            _fraction(row.with_cases, row.cases),
            _statuses(row.statuses),
        )
    return table


def _topic_table(rows: Sequence[CoverageRow]) -> Table:
    table = Table(title="Topics", title_justify="left")
    for name in ("Topic", "Tier", "Kind", "Lesson", "Cards", "Interview", "Problems", "Status"):
        table.add_column(name, overflow="fold" if name == "Topic" else "ellipsis")
    for row in rows:
        table.add_row(
            escape(row.id),
            row.tier,
            row.kind,
            _tick(row.has_lesson),
            _tick(row.has_cards),
            _tick(row.has_interview),
            str(row.n_problems),
            escape(row.status),
        )
    return table


def _fraction(part: int, total: int) -> str:
    return f"{part}/{total}"


def _statuses(counts: dict[str, int]) -> str:
    parts = [f"{name} {counts[name]}" for name in _STATUS_ORDER if counts.get(name)]
    return ", ".join(parts) or "none"


def _tick(value: bool) -> str:
    return "✓" if value else "-"
