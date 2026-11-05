from __future__ import annotations

from fastapi import APIRouter

from dogma.api.deps import ClockDep, SettingsDep
from dogma.api.openapi import error_responses
from dogma.api.schemas.content import CatalogOut, ProblemOut, TopicOut
from dogma.api.services import content

router = APIRouter(tags=["content"])


@router.get(
    "/catalog",
    response_model=CatalogOut,
    summary="Topics and problems, with status and tier",
)
def get_catalog(settings: SettingsDep, clock: ClockDep) -> CatalogOut:
    return content.catalog(settings, clock())


@router.get(
    "/topics/{topic_id:path}",
    response_model=TopicOut,
    summary="A topic's lesson, metadata, card count and problems",
    responses=error_responses((404, "No topic with this id")),
)
def get_topic(topic_id: str, settings: SettingsDep) -> TopicOut:
    return content.topic(settings, topic_id)


@router.get(
    "/problems/{problem_id:path}",
    response_model=ProblemOut,
    summary="A problem's statement, starter and metadata (never solutions or tests)",
    responses=error_responses((404, "No problem with this id")),
)
def get_problem(problem_id: str, settings: SettingsDep) -> ProblemOut:
    return content.problem(settings, problem_id)
