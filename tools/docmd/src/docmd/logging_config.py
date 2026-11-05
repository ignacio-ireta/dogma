"""Logging setup: rich console handler plus optional per-run log file.

Document content is never logged by default (see AppConfig.log_document_content);
log records carry paths, counts, and error codes only.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler

LOG_FORMAT_FILE = "%(asctime)s %(levelname)-8s %(name)s %(message)s"


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure the root ``docmd`` logger. Idempotent: clears prior handlers."""
    root = logging.getLogger("docmd")
    root.setLevel(level.upper())
    root.handlers.clear()
    root.propagate = False

    console = RichHandler(show_path=False, rich_tracebacks=False, markup=False)
    console.setLevel(level.upper())
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT_FILE))
        file_handler.setLevel(logging.DEBUG)
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"docmd.{name}")
