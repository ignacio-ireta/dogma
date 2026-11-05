"""RFC 9457 problem bodies, and the table that maps domain codes to HTTP statuses.

Every error the API returns is ``application/problem+json`` and carries the domain ``code``
(ENGINEERING_STANDARDS §4.4, §8). Request content and tracebacks never reach the body.
"""

from __future__ import annotations

import logging
from http import HTTPStatus
from typing import Final, cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from starlette.exceptions import HTTPException as StarletteHTTPException

from dogma.core.errors import DogmaError, ErrorCode
from dogma.db.errors import StorageError, StorageErrorCode

log = logging.getLogger(__name__)

PROBLEM_JSON: Final = "application/problem+json"
_TYPE_PREFIX: Final = "urn:dogma:error:"

# Every ErrorCode and StorageErrorCode member has a row. A test enforces that, so a new code
# is a conscious choice here rather than a silent 500.
STATUS_BY_CODE: Final[dict[str, int]] = {
    # content and catalog
    ErrorCode.CONTENT_SCHEMA_INVALID: 500,
    ErrorCode.CONTENT_NOT_FOUND: 404,
    ErrorCode.CONTENT_ID_MISMATCH: 500,
    ErrorCode.CONTENT_PREREQ_CYCLE: 500,
    ErrorCode.SYLLABUS_INVALID: 500,
    # runner (learner code failing is a report, so these are infrastructure faults)
    ErrorCode.RUNNER_TIMEOUT: 500,
    ErrorCode.RUNNER_OOM: 500,
    ErrorCode.RUNNER_CRASHED: 500,
    ErrorCode.RUNNER_RUNTIME_UNAVAILABLE: 503,
    ErrorCode.HARNESS_REPORT_MISSING: 500,
    # tutor and local model
    ErrorCode.LLM_UNAVAILABLE: 503,
    ErrorCode.LLM_ENDPOINT_REFUSED: 503,
    ErrorCode.LLM_RESPONSE_INVALID: 502,
    ErrorCode.LLM_LOG_FAILED: 500,
    ErrorCode.TUTOR_LOCKED: 403,
    ErrorCode.TUTOR_NO_FAILURE: 409,
    ErrorCode.TUTOR_PROMPT_INVALID: 500,
    ErrorCode.TUTOR_SCORE_REPLY_INVALID: 502,
    ErrorCode.TUTOR_INTERVIEWER_REPLY_INVALID: 502,
    ErrorCode.TUTOR_QUESTION_INVALID: 422,
    ErrorCode.CONFIG_INVALID: 500,
    # reviews
    ErrorCode.REVIEW_TIME_INVALID: 422,
    ErrorCode.REVIEW_LIMIT_INVALID: 422,
    ErrorCode.REVIEW_RATING_INVALID: 422,
    ErrorCode.REVIEW_NOT_FOUND: 404,
    ErrorCode.REVIEW_NOT_DUE: 409,
    ErrorCode.REVIEW_STALE: 409,
    # corpus
    ErrorCode.CORPUS_QUERY_INVALID: 422,
    ErrorCode.CORPUS_INDEX_MISSING: 503,
    ErrorCode.CORPUS_INDEX_INVALID: 503,
    ErrorCode.CORPUS_PATH_INVALID: 500,
    ErrorCode.CORPUS_SOURCE_INVALID: 500,
    ErrorCode.CORPUS_CONVERTER_MISSING: 503,
    ErrorCode.CORPUS_CONVERSION_FAILED: 500,
    ErrorCode.CORPUS_CONVERSION_TIMEOUT: 500,
    # sessions
    ErrorCode.SESSION_TIME_INVALID: 422,
    ErrorCode.SESSION_NOT_FOUND: 404,
    ErrorCode.SESSION_NOTE_INVALID: 422,
    # timed re-solves
    ErrorCode.RESOLVE_NOT_CORE: 422,
    ErrorCode.RESOLVE_NOT_READY: 409,
    ErrorCode.RESOLVE_NOT_SOLVED: 409,
    ErrorCode.RESOLVE_ALREADY_OPEN: 409,
    ErrorCode.RESOLVE_NOT_FOUND: 404,
    ErrorCode.RESOLVE_RECORDING_OFF: 409,
    # interview answers
    ErrorCode.ANSWER_RECORDING_OFF: 409,
    ErrorCode.ANSWER_SCORE_INVALID: 422,
    ErrorCode.ANSWER_TEXT_INVALID: 422,
    ErrorCode.ANSWER_NO_TEXT: 409,
    # mock interviews
    ErrorCode.MOCK_ALREADY_OPEN: 409,
    ErrorCode.MOCK_NOT_OPEN: 409,
    ErrorCode.MOCK_NOT_SCORED: 409,
    ErrorCode.MOCK_NOT_FOUND: 404,
    ErrorCode.MOCK_RECORDING_OFF: 409,
    ErrorCode.MOCK_SCORE_INVALID: 422,
    ErrorCode.CALENDAR_RECORDING_OFF: 409,
    ErrorCode.CALENDAR_REQUEST_INVALID: 422,
    # HTTP layer
    ErrorCode.API_ATTEMPT_NOT_FOUND: 404,
    ErrorCode.API_WORKSPACE_MISSING: 409,
    ErrorCode.API_WORKSPACE_TOO_LARGE: 413,
    ErrorCode.API_WORKSPACE_INVALID: 422,
    ErrorCode.API_REQUEST_TOO_LARGE: 413,
    ErrorCode.API_REQUEST_REFUSED: 403,
    ErrorCode.API_VALIDATION_FAILED: 422,
    ErrorCode.API_NOT_FOUND: 404,
    ErrorCode.API_METHOD_NOT_ALLOWED: 405,
    ErrorCode.API_INTERNAL_ERROR: 500,
    # learner database (a missing or locked file is unavailability, not a bug)
    StorageErrorCode.DB_UNAVAILABLE: 503,
    StorageErrorCode.DB_MIGRATION_FAILED: 503,
    StorageErrorCode.DB_READ_FAILED: 500,
    StorageErrorCode.DB_WRITE_FAILED: 500,
    StorageErrorCode.BACKUP_WRITE_FAILED: 500,
}


class ProblemField(BaseModel):
    """One invalid field in a request body or query."""

    model_config = ConfigDict(frozen=True)

    field: str
    message: str


class Problem(BaseModel):
    """RFC 9457 problem details, with the domain ``code`` and, for validation, ``errors``."""

    model_config = ConfigDict(frozen=True)

    type: str
    title: str
    status: int
    detail: str
    instance: str
    code: str
    errors: list[ProblemField] | None = None


def status_for(code: str) -> int:
    """The HTTP status for a domain code. Unlisted codes are server faults."""
    return STATUS_BY_CODE.get(str(code), 500)


def problem_for_code(
    code: str,
    detail: str,
    *,
    instance: str,
    status: int,
    errors: list[ProblemField] | None = None,
) -> Problem:
    return Problem(
        type=f"{_TYPE_PREFIX}{code}",
        title=_title(status),
        status=status,
        detail=detail,
        instance=instance,
        code=code,
        errors=errors,
    )


def problem_response(problem: Problem, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(
        content=problem.model_dump(mode="json", exclude_none=True),
        status_code=problem.status,
        media_type=PROBLEM_JSON,
        headers=headers,
    )


def install_handlers(app: FastAPI) -> None:
    """Route every failure through the problem format, including validation and routing."""
    app.add_exception_handler(DogmaError, _domain_error)
    app.add_exception_handler(StorageError, _storage_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(Exception, _unexpected_error)


def _domain_error(request: Request, exc: Exception) -> JSONResponse:
    error = cast(DogmaError, exc)
    code = str(error.code)
    problem = problem_for_code(
        code, error.message, instance=request.url.path, status=status_for(code)
    )
    return problem_response(problem)


def _storage_error(request: Request, exc: Exception) -> JSONResponse:
    error = cast(StorageError, exc)
    code = str(error.code)
    log.warning("learner database error %s on %s", code, request.url.path)
    problem = problem_for_code(
        code, error.message, instance=request.url.path, status=status_for(code)
    )
    return problem_response(problem)


def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    validation = cast(RequestValidationError, exc)
    fields = [
        ProblemField(field=_field_name(item["loc"]), message=str(item["msg"]))
        for item in validation.errors()
    ]
    problem = problem_for_code(
        "API_VALIDATION_FAILED",
        "the request does not match the API contract; see errors",
        instance=request.url.path,
        status=422,
        errors=fields,
    )
    return problem_response(problem)


def _http_error(request: Request, exc: Exception) -> JSONResponse:
    http_error = cast(StarletteHTTPException, exc)
    if http_error.status_code == 405:
        code = "API_METHOD_NOT_ALLOWED"
        detail = "this route does not accept that method"
    elif http_error.status_code == 404:
        code = "API_NOT_FOUND"
        detail = "no such route under /api/v1"
    else:
        code = "API_REQUEST_REFUSED"
        detail = str(http_error.detail)
    problem = problem_for_code(
        code, detail, instance=request.url.path, status=http_error.status_code
    )
    headers = (
        {"Allow": http_error.headers["Allow"]}
        if http_error.headers and "Allow" in http_error.headers
        else None
    )
    return problem_response(problem, headers)


def _unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    log.error("unhandled error on %s", request.url.path, exc_info=exc)
    problem = problem_for_code(
        "API_INTERNAL_ERROR",
        "something went wrong on the server; the details are in the server log",
        instance=request.url.path,
        status=500,
    )
    return problem_response(problem)


def _field_name(location: tuple[object, ...]) -> str:
    parts = [str(part) for part in location if part != "body"]
    return ".".join(parts) or "request"


def _title(status: int) -> str:
    try:
        return HTTPStatus(status).phrase
    except ValueError:
        return "Error"
