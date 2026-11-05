from __future__ import annotations

from fastapi import APIRouter

from dogma import __version__
from dogma.api.schemas.system import HealthOut

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthOut, summary="Is the API up")
def health() -> HealthOut:
    return HealthOut(status="ok", version=__version__)
