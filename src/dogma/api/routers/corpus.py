from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from dogma.api.deps import SettingsDep
from dogma.api.openapi import error_responses
from dogma.api.schemas.corpus import SearchOut
from dogma.api.services import corpus

router = APIRouter(tags=["corpus"])


@router.get(
    "/search",
    response_model=SearchOut,
    summary="Cited passages from the local corpus index",
    responses=error_responses((503, "The corpus index is missing: run dogma corpus sync")),
)
def search(
    settings: SettingsDep,
    q: Annotated[str, Query(max_length=4096)] = "",
    topic: Annotated[list[str] | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 5,
) -> SearchOut:
    return corpus.find(settings, q, topic or [], limit)
