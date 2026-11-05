"""OpenAPI helpers: error responses are documented as ``application/problem+json``.

FastAPI would otherwise describe validation errors with its own ``HTTPValidationError`` shape,
which the API no longer returns. The contract (ENGINEERING_STANDARDS §8) is what the web client
generates types from, so it must describe what is actually sent.
"""

from __future__ import annotations

from typing import Any, Final

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from dogma.api.errors import PROBLEM_JSON, Problem

PROBLEM_REF: Final = "#/components/schemas/Problem"
_DROPPED_COMPONENTS: Final = ("HTTPValidationError", "ValidationError")


def error_responses(*entries: tuple[int, str]) -> dict[int | str, dict[str, Any]]:
    """Route-level ``responses=`` entries for the statuses a route can answer with."""
    return {status: {"description": description} for status, description in entries}


def build_openapi(app: FastAPI) -> dict[str, Any]:
    schema: dict[str, Any] = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
    )
    components: dict[str, Any] = schema.setdefault("components", {}).setdefault("schemas", {})
    problem_schema = Problem.model_json_schema(ref_template="#/components/schemas/{model}")
    for name, definition in problem_schema.pop("$defs", {}).items():
        components[name] = definition
    components["Problem"] = problem_schema
    for name in _DROPPED_COMPONENTS:
        components.pop(name, None)
    for path_item in schema.get("paths", {}).values():
        for operation in path_item.values():
            _describe_errors(operation.get("responses", {}))
    return schema


def _describe_errors(responses: dict[str, Any]) -> None:
    for status, response in responses.items():
        if str(status).startswith(("4", "5")):
            response["content"] = {PROBLEM_JSON: {"schema": {"$ref": PROBLEM_REF}}}
