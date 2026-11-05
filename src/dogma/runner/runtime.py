"""Availability checks for optional learner runtimes, without loading native libraries."""

from __future__ import annotations

import importlib.util

from dogma.core.errors import ErrorCode, RunnerError


def ensure_available(runtime: str) -> None:
    """Diagnose an absent optional package before starting the harness."""
    if runtime == "py-torch" and importlib.util.find_spec("torch") is None:
        raise RunnerError(
            ErrorCode.RUNNER_RUNTIME_UNAVAILABLE,
            "This problem needs the optional torch runtime. "
            "Start the command with uv run --frozen --group torch dogma ... "
            "(for the browser, use uv run --frozen --group torch dogma serve).",
        )
