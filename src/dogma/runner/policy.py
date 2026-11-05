"""Fixed macOS Seatbelt policy for the learner's harness process (ROADMAP I-21, ARCHITECTURE §4.2).

The policy denies network access and every file write except inside the attempt directory. It
is applied with ``/usr/bin/sandbox-exec`` and is the only launch path: when Seatbelt is missing
or cannot be applied the attempt fails with an infrastructure error and nothing runs unwrapped.

This contains runaway and careless learner code on one trusted user's machine. It is not a
hostile-code boundary: reads are unrestricted, so same-user files stay readable, and the child
shares the harness's report channel (ARCHITECTURE §4.2).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Final

from dogma.core.errors import ErrorCode, RunnerError

SANDBOX_EXEC: Final = Path("/usr/bin/sandbox-exec")
RUN_DIR_PARAM: Final = "RUN_DIR"
# Seatbelt applies the last matching rule, so the write allowances come after the denial.
# The attempt directory reaches the profile only as a parameter, never by text interpolation.
# ``/dev/null`` is the one system path granted, for data writes only: pytest's logging plugin
# opens it at startup, and it holds no state. Terminals and every other path stay denied.
PROFILE: Final = f"""(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write-data (literal "/dev/null"))
(allow file-write* (subpath (param "{RUN_DIR_PARAM}")))
"""


def _current_platform() -> str:
    return sys.platform


def ensure_supported() -> None:
    """Raise unless this host can apply the fixed policy. Called before any run dir is created."""
    reason = _unsupported_reason(_current_platform(), SANDBOX_EXEC)
    if reason is not None:
        raise RunnerError(
            ErrorCode.RUNNER_CRASHED,
            f"learner code cannot run: {reason}. Execution is refused rather than run without "
            "OS enforcement.",
        )


def sandbox_command(command: list[str], run_dir: Path) -> list[str]:
    """Wrap ``command`` in ``sandbox-exec`` so only ``run_dir`` is writable, and deny network."""
    ensure_supported()
    return [
        str(SANDBOX_EXEC),
        "-D",
        f"{RUN_DIR_PARAM}={_run_dir_parameter(run_dir)}",
        "-p",
        PROFILE,
        *command,
    ]


def _unsupported_reason(platform: str, executable: Path) -> str | None:
    if platform != "darwin":
        return f"this platform ({platform}) has no supported OS sandbox; only macOS is supported"
    if not (executable.is_file() and os.access(executable, os.X_OK)):
        return f"sandbox-exec is missing or not executable at {executable}"
    return None


def _run_dir_parameter(run_dir: Path) -> str:
    resolved = str(run_dir.resolve())
    # Seatbelt matches canonical paths, so the resolved form is what must be allowed. Control
    # characters never belong in a run directory and would blur the one-line -D argument.
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in resolved):
        raise RunnerError(
            ErrorCode.RUNNER_CRASHED,
            "learner code cannot run: the attempt directory name has control characters",
        )
    return resolved
