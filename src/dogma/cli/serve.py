"""``dogma serve``: the local API on loopback only (ARCHITECTURE §4.7, ADR-0001).

The API has no authentication, so it listens on loopback and nowhere else. A non-loopback host
is refused before anything is imported or bound. FastAPI and uvicorn are imported inside the
command, so every other CLI command starts without loading the web stack.
"""

from __future__ import annotations

import ipaddress
from typing import Annotated

import typer
from rich.markup import escape

from dogma.cli.context import CliError, out, settings_of

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
_LOOPBACK_NAME = "localhost"


def check_host(host: str) -> None:
    """Accept loopback addresses and ``localhost``; refuse everything else before binding."""
    if host == _LOOPBACK_NAME:
        return
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and address.is_loopback:
        return
    raise CliError(
        f"refusing to listen on {host!r}: dogma serve is local only (loopback: 127.0.0.1, "
        "::1 or localhost). The API has no authentication, so it must not be reachable from "
        "other machines (ADR-0001)."
    )


def serve_command(
    ctx: typer.Context,
    host: Annotated[
        str,
        typer.Option(help="Loopback address to listen on. Other hosts are refused."),
    ] = DEFAULT_HOST,
    port: Annotated[
        int,
        typer.Option(min=1, max=65535, help="TCP port."),
    ] = DEFAULT_PORT,
    allow_origin: Annotated[
        list[str] | None,
        typer.Option(
            "--allow-origin",
            help="A browser origin allowed to call the API (repeatable). "
            "Default: http://127.0.0.1:5173, the web dev server.",
        ),
    ] = None,
) -> None:
    """Serve the local API at http://HOST:PORT/api/v1 until you stop it."""
    check_host(host)
    settings = settings_of(ctx)
    # Imported here, not at module level, to keep the rest of the CLI fast to start.
    import uvicorn  # noqa: PLC0415

    from dogma.api.app import DEFAULT_ALLOWED_ORIGINS, create_app  # noqa: PLC0415

    origins = tuple(allow_origin) if allow_origin else DEFAULT_ALLOWED_ORIGINS
    application = create_app(settings, allowed_origins=origins)
    base = f"http://{host}:{port}"
    console = out()
    console.print(f"dogma API on {escape(base)}/api/v1 (Ctrl+C to stop)", soft_wrap=True)
    console.print(f"OpenAPI contract: {escape(base)}/api/v1/openapi.json", soft_wrap=True)
    uvicorn.run(application, host=host, port=port, log_level="warning", access_log=False)
