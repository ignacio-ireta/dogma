from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from dogma.api.deps import SettingsDep
from dogma.api.openapi import error_responses
from dogma.api.schemas.coverage import CoverageOut
from dogma.api.services import coverage
from dogma.content.schemas import TRACK_CODE_PATTERN, Tier

router = APIRouter(tags=["coverage"])


@router.get(
    "/coverage",
    response_model=CoverageOut,
    summary="What has been authored per track and tier, with topic rows when filtered",
    description=(
        "Counts come from the same scoping as `dogma content coverage`. `rows` is null without "
        "a filter. `unreadable_topics` lists topic ids whose package does not load, which are "
        "not counted."
    ),
    responses=error_responses((404, "No track with this code")),
)
def get_coverage(
    settings: SettingsDep,
    tier: Annotated[Tier | None, Query(description="Only topics of this tier")] = None,
    track: Annotated[
        str | None,
        Query(max_length=32, pattern=TRACK_CODE_PATTERN, description="Only this track code"),
    ] = None,
) -> CoverageOut:
    return coverage.report(settings, tier=tier, track=track)
