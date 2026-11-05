"""Pure-ASGI guards that run before routing: loopback Host, allowed Origin, and a body size cap.

They sit outside FastAPI so a refused request never reaches a router or a file. A loopback
server with no authentication (ADR-0001) is still reachable by any web page that the browser
lets talk to 127.0.0.1, so the Host and Origin checks are what keep other sites out (DNS
rebinding and cross-site requests).
"""

from __future__ import annotations

from typing import Final

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from dogma.api.errors import problem_for_code, problem_response

LOOPBACK_NAMES: Final = frozenset({"127.0.0.1", "localhost", "::1"})
MAX_REQUEST_BYTES: Final = 1024 * 1024


class _BodyTooLargeError(Exception):
    """Raised from inside ``receive`` once a request body passes the limit."""


def host_name(value: str) -> str:
    """The bare host from a Host header: ``localhost:8765`` and ``[::1]:8765`` both reduce."""
    if value.startswith("["):
        end = value.find("]")
        return value[1:end] if end > 0 else ""
    return value.split(":", 1)[0]


def _header_values(scope: Scope) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_name, raw_value in scope.get("headers", []):
        values.setdefault(raw_name.decode("latin-1").lower(), raw_value.decode("latin-1"))
    return values


async def _send_problem(
    scope: Scope, receive: Receive, send: Send, *, status: int, code: str, detail: str
) -> None:
    problem = problem_for_code(code, detail, instance=str(scope.get("path", "")), status=status)
    await problem_response(problem)(scope, receive, send)


class HostOriginGuard:
    """Refuse non-loopback Host headers, and Origin headers that are not explicitly allowed."""

    def __init__(self, app: ASGIApp, *, allowed_origins: frozenset[str]) -> None:
        self.app = app
        self.allowed_origins = allowed_origins

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = _header_values(scope)
        if host_name(headers.get("host", "")) not in LOOPBACK_NAMES:
            await _send_problem(
                scope,
                receive,
                send,
                status=403,
                code="API_REQUEST_REFUSED",
                detail="this API answers only to loopback host names",
            )
            return
        origin = headers.get("origin")
        if origin is not None and origin not in self.allowed_origins:
            await _send_problem(
                scope,
                receive,
                send,
                status=403,
                code="API_REQUEST_REFUSED",
                detail="this origin is not allowed to use the local API",
            )
            return
        await self.app(scope, receive, send)


class BodyLimit:
    """Refuse request bodies over ``max_bytes``: by declared length first, then as they arrive."""

    def __init__(self, app: ASGIApp, *, max_bytes: int = MAX_REQUEST_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = _header_values(scope).get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            await self._refuse(scope, receive, send)
            return
        await self._guarded(scope, receive, send)

    async def _guarded(self, scope: Scope, receive: Receive, send: Send) -> None:
        received = 0
        started = False

        async def counting_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLargeError
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, tracking_send)
        except _BodyTooLargeError:
            if started:
                raise
            await self._refuse(scope, receive, send)

    async def _refuse(self, scope: Scope, receive: Receive, send: Send) -> None:
        await _send_problem(
            scope,
            receive,
            send,
            status=413,
            code="API_REQUEST_TOO_LARGE",
            detail=f"the request body is larger than {self.max_bytes} bytes",
        )
