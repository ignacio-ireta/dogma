"""The app factory: ``create_app(settings)`` builds the whole local API (ARCHITECTURE §4.7).

Everything is served under ``/api/v1``. There is no interactive docs page (its assets come from a
CDN) and no authentication: the API is for one learner on loopback (ADR-0001).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dogma import __version__
from dogma.api.errors import install_handlers
from dogma.api.middleware import MAX_REQUEST_BYTES, BodyLimit, HostOriginGuard
from dogma.api.openapi import build_openapi
from dogma.api.routers import (
    attempts,
    calendar,
    content,
    corpus,
    coverage,
    interviews,
    learning,
    mocks,
    resolves,
    system,
    tutor,
    workspace,
)
from dogma.api.schemas.tutor import ReviewedEventOut, TextDeltaOut, UnavailableOut
from dogma.api.static import DEFAULT_WEB_DIST, SameOriginAsServed, mount_spa
from dogma.core.config import Settings

API_PREFIX: Final = "/api/v1"
DEFAULT_ALLOWED_ORIGINS: Final[tuple[str, ...]] = ("http://127.0.0.1:5173",)
_CORS_METHODS: Final = ["GET", "POST", "PUT", "OPTIONS"]
_CORS_HEADERS: Final = ["Content-Type"]


# The payloads of the tutor's SSE events are not response models, so the contract lists them here.
# The browser generates its event types from them (ENGINEERING_STANDARDS §7, ADR-0017).
_STREAM_EVENT_MODELS: Final[tuple[type[BaseModel], ...]] = (
    ReviewedEventOut,
    TextDeltaOut,
    UnavailableOut,
)


class DogmaAPI(FastAPI):
    """FastAPI with the problem-format OpenAPI description (see ``dogma.api.openapi``)."""

    def openapi(self) -> dict[str, Any]:
        if self.openapi_schema is None:
            schema = build_openapi(self)
            _add_stream_events(schema)
            self.openapi_schema = schema
        return self.openapi_schema


def _add_stream_events(schema: dict[str, Any]) -> None:
    components: dict[str, Any] = schema.setdefault("components", {}).setdefault("schemas", {})
    for model in _STREAM_EVENT_MODELS:
        definition = model.model_json_schema(ref_template="#/components/schemas/{model}")
        for name, nested in definition.pop("$defs", {}).items():
            components.setdefault(name, nested)
        components[model.__name__] = definition


def create_app(
    settings: Settings,
    *,
    allowed_origins: Sequence[str] = DEFAULT_ALLOWED_ORIGINS,
    web_dist: Path = DEFAULT_WEB_DIST,
) -> FastAPI:
    """Build the API for one repo. ``allowed_origins`` are the browser origins that may call it.

    ``web_dist`` is the built web app. It is served at ``/`` only when its ``index.html`` exists.
    The default is the code checkout's build, not the data repo's, so a scratch data repo still
    serves the UI.
    """
    origins = tuple(allowed_origins)
    app = DogmaAPI(
        title="dogma API",
        version=__version__,
        openapi_url=f"{API_PREFIX}/openapi.json",
        docs_url=None,
        redoc_url=None,
    )
    app.state.settings = settings
    install_handlers(app)
    serves_ui = mount_spa(app, web_dist)
    app.include_router(_api_router())
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(origins),
        allow_methods=_CORS_METHODS,
        allow_headers=_CORS_HEADERS,
        allow_credentials=False,
    )
    app.add_middleware(HostOriginGuard, allowed_origins=frozenset(origins))
    app.add_middleware(BodyLimit, max_bytes=MAX_REQUEST_BYTES)
    if serves_ui:
        # Added last, so it runs first: the page's own origin is not a cross-origin request.
        app.add_middleware(SameOriginAsServed)
    return app


def _api_router() -> APIRouter:
    router = APIRouter(prefix=API_PREFIX)
    # interviews comes before content: its GET /topics/{id}/interview must win over the topic route.
    for module in (
        system,
        interviews,
        content,
        workspace,
        attempts,
        learning,
        resolves,
        mocks,
        calendar,
        corpus,
        coverage,
        tutor,
    ):
        router.include_router(module.router)
    return router
