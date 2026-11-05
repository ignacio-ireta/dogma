"""Coverage of the syllabus by authored content: what the Map page overlays (ROADMAP I-23)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from dogma.content.schemas import Tier


class CoverageSummaryOut(BaseModel):
    """Counts for one track (``title`` set) or one tier (``title`` null)."""

    key: str
    title: str | None
    topics: int
    with_package: int
    with_lesson: int
    with_cards: int
    with_interview: int
    with_problems: int
    cases: int
    with_cases: int
    statuses: dict[str, int]


class CoverageRowOut(BaseModel):
    """One syllabus topic and what is authored for it. Status is ``missing`` without a package."""

    id: str
    track: str
    tier: Tier
    title: str
    kind: Literal["topic", "case"]
    has_package: bool
    has_lesson: bool
    has_cards: bool
    has_interview: bool
    n_problems: int
    status: str


class CoverageOut(BaseModel):
    tier: Tier | None
    track: str | None
    by_track: list[CoverageSummaryOut]
    by_tier: list[CoverageSummaryOut]
    rows: list[CoverageRowOut] | None
    unreadable_topics: list[str]
