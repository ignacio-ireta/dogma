"""FastAPI dependencies: settings, the clock, the model and the tutor gateway.

Routes take these through ``Depends`` so tests can override the clock (for deterministic dates),
the model (for a fake provider) and the gateway without touching the services.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, Request

from dogma.core.config import Settings
from dogma.tutor.gateway import Gateway, LLMProvider
from dogma.tutor.models import CompletionChunk, CompletionRequest, CompletionResult
from dogma.tutor.providers.local import LocalProvider

Clock = Callable[[], datetime]


class LocalModel:
    """The loopback model. The provider is built inside each call, so a refused URL raises there,
    where the gateway logs it (ENGINEERING_STANDARDS §10). Nothing is built at startup."""

    name = LocalProvider.name

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    def complete(self, req: CompletionRequest) -> CompletionResult:
        return LocalProvider(self._base_url).complete(req)

    def stream(self, req: CompletionRequest) -> Iterator[CompletionChunk]:
        return LocalProvider(self._base_url).stream(req)


def get_settings(request: Request) -> Settings:
    settings = request.app.state.settings
    if not isinstance(settings, Settings):
        raise RuntimeError("create_app was not given Settings")
    return settings


def get_clock() -> Clock:
    return _utc_now


def get_model(settings: Annotated[Settings, Depends(get_settings)]) -> LLMProvider:
    """The model behind the gateway: the loopback local provider (ADR-0008)."""
    return LocalModel(settings.llm_base_url)


def get_gateway(
    settings: Annotated[Settings, Depends(get_settings)],
    model: Annotated[LLMProvider, Depends(get_model)],
) -> Gateway:
    """The gateway the JSON tutor routes call."""
    return Gateway(settings, provider=model)


def _utc_now() -> datetime:
    return datetime.now(UTC)


SettingsDep = Annotated[Settings, Depends(get_settings)]
ClockDep = Annotated[Clock, Depends(get_clock)]
ModelDep = Annotated[LLMProvider, Depends(get_model)]
GatewayDep = Annotated[Gateway, Depends(get_gateway)]
