"""Serve the built UI against disposable learner data for the browser smoke tests."""

from __future__ import annotations

import argparse
import json
import shutil
import socket
from datetime import UTC, datetime
from pathlib import Path

import uvicorn

from dogma.api.app import create_app
from dogma.api.deps import get_clock, get_model
from dogma.core.config import Settings
from dogma.tutor.providers.fake import FakeProvider

REPO_ROOT = Path(__file__).resolve().parents[3]
NOW = datetime(2026, 10, 26, 9, 0, tzinfo=UTC)


def main() -> None:
    """Serve one newly created scratch repo on a reserved loopback socket."""
    web_dist = REPO_ROOT / "web" / "dist"
    if not (web_dist / "index.html").is_file():
        raise RuntimeError("Build the browser first: cd web && pnpm build")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scratch_root", type=Path)
    root = parser.parse_args().scratch_root
    # The Playwright parent creates and removes this directory, including on server failure.
    shutil.copytree(REPO_ROOT / "curriculum", root / "curriculum")
    topics = (
        Path("content/cml/vector-operations"),
        Path("content/sql/sql-select-filter-and-sort"),
        Path("content/llm/word-and-character-tokenizers"),
    )
    for topic in topics:
        shutil.copytree(REPO_ROOT / topic, root / topic)
    # The mock flow needs the sys-track cases (the planner picks one when none is asked for), and
    # the cases cite their sources, which the catalog checks exist beside the repo's content. All
    # of sources/ is copied so that a case citing a new source does not break the scratch catalog.
    shutil.copytree(REPO_ROOT / "content" / "sys", root / "content" / "sys")
    shutil.copytree(REPO_ROOT / "sources", root / "sources")
    settings = Settings(
        repo_root=root,
        record_attempts=True,
        allow_external=False,
        _env_file=None,
    )
    app = create_app(settings, web_dist=web_dist)
    app.dependency_overrides[get_clock] = lambda: lambda: NOW
    fake_model = FakeProvider()
    app.dependency_overrides[get_model] = lambda: fake_model
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        # The fixture polls this reserved socket; it never attaches to an existing server.
        print(json.dumps({"baseURL": f"http://127.0.0.1:{port}"}), flush=True)  # noqa: T201
        server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False))
        server.run(sockets=[listener])


if __name__ == "__main__":
    main()
