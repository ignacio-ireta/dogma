"""The built web app, served by the API process at ``/`` (ARCHITECTURE §4.8, ROADMAP I-15).

The UI is mounted only when ``web/dist/index.html`` exists, so an API-only checkout behaves as it
did. An unmatched GET outside ``/api`` answers with the build: an existing file, or ``index.html``
so that client routes survive a reload. Unknown ``/api`` paths, disabled API documentation and
missing asset-like names stay problems. Nothing outside the build directory is ever read.

The fallback wraps the exception handler for unmatched routes, so matched routes (including their
405 answers) keep the API's behaviour. ``SameOriginAsServed`` covers the one browser case the Origin
allow-list would otherwise refuse: the served page calling its own server.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Final

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request
from starlette.types import ASGIApp, Receive, Scope, Send

from dogma.api.middleware import LOOPBACK_NAMES, host_name

DEFAULT_WEB_DIST: Final = Path(__file__).resolve().parents[3] / "web" / "dist"
INDEX: Final = "index.html"
_SERVER_SEGMENTS: Final = frozenset({"api", "docs", "redoc"})
_FINGERPRINTED: Final = "assets"
_IMMUTABLE: Final = "public, max-age=31536000, immutable"
_REVALIDATE: Final = "no-cache"
# No runtime CDNs (ENGINEERING_STANDARDS §7): every script, style, font and image is the app's own.
# Styles may be inline because KaTeX and CodeMirror set style attributes.
_CSP: Final = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; font-src 'self'; "
    "img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; "
    "frame-ancestors 'none'; form-action 'self'"
)
_MEDIA_TYPES: Final = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".map": "application/json",
    ".svg": "image/svg+xml",
    ".woff2": "font/woff2",
    ".woff": "font/woff",
    ".png": "image/png",
    ".ico": "image/x-icon",
}
_FALLBACK_MEDIA_TYPE: Final = "application/octet-stream"

ExceptionHandler = Callable[[Request, Exception], Response | Awaitable[Response]]


def mount_spa(app: FastAPI, dist: Path = DEFAULT_WEB_DIST) -> bool:
    """Serve ``dist`` for unmatched GET requests. Returns False and changes nothing without a build.

    Call after ``install_handlers``: it wraps the handler registered for HTTP errors.
    """
    root = dist.resolve()
    if _index_in(root) is None:
        return False
    previous: ExceptionHandler = app.exception_handlers[StarletteHTTPException]

    def handler(request: Request, exc: Exception) -> Response | Awaitable[Response]:
        if (
            isinstance(exc, StarletteHTTPException)
            and exc.status_code == 404
            and request.method in ("GET", "HEAD")
        ):
            found = page_for(request.url.path, root)
            if found is not None:
                return _file_response(found, root)
        return previous(request, exc)

    app.add_exception_handler(StarletteHTTPException, handler)
    return True


def page_for(path: str, root: Path) -> Path | None:
    """The file to answer ``path`` with, or None when the request is a real 404.

    ``root`` is the resolved build directory. A path that resolves outside it is never served.
    """
    relative = path.lstrip("/")
    if relative == "":
        return _index_in(root)
    parts = relative.split("/")
    if parts[0] in _SERVER_SEGMENTS:
        return None
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        return None
    if candidate.is_file():
        return candidate
    if "." in parts[-1]:
        # Looks like a built asset (or favicon) that is not there: say so, do not send HTML.
        return None
    return _index_in(root)


def _index_in(root: Path) -> Path | None:
    """Resolve the SPA entry with the same containment rule used for every asset."""
    index = (root / INDEX).resolve()
    return index if index.is_relative_to(root) and index.is_file() else None


def _file_response(path: Path, root: Path) -> FileResponse:
    fingerprinted = path.relative_to(root).parts[:1] == (_FINGERPRINTED,)
    headers = {"Cache-Control": _IMMUTABLE if fingerprinted else _REVALIDATE}
    if path.suffix == ".html":
        headers["Content-Security-Policy"] = _CSP
    media_type = _MEDIA_TYPES.get(path.suffix.lower(), _FALLBACK_MEDIA_TYPE)
    return FileResponse(path, media_type=media_type, headers=headers)


class SameOriginAsServed:
    """Drop an ``Origin`` that names this server itself, so the allow-list does not refuse it.

    A browser sends ``Origin: http://<host>:<port>`` on a POST from the page this server serves.
    That request is not cross-origin, so the header is removed before the Host and Origin guard
    runs. The guard still checks ``Host``: a non-loopback Host is refused, which is what stops DNS
    rebinding. A page on another site cannot set its Origin to this server's origin.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            scope = _without_own_origin(scope)
        await self.app(scope, receive, send)


def _without_own_origin(scope: Scope) -> Scope:
    headers: list[tuple[bytes, bytes]] = list(scope.get("headers", []))
    host = _first(headers, b"host")
    origin = _first(headers, b"origin")
    if host is None or origin is None:
        return scope
    host_text = host.decode("latin-1")
    if host_name(host_text) not in LOOPBACK_NAMES:
        return scope
    if origin.decode("latin-1") != f"http://{host_text}":
        return scope
    kept = [(name, value) for name, value in headers if name != b"origin"]
    return {**scope, "headers": kept}


def _first(headers: list[tuple[bytes, bytes]], name: bytes) -> bytes | None:
    return next((value for key, value in headers if key == name), None)
