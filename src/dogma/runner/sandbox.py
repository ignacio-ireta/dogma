"""Process sandbox: runs the harness in its own process group under time and memory limits.

It contains runaway learner code (infinite loops, memory blowups, stray writes). It is not a
security boundary against an adversary (ENGINEERING_STANDARDS §11). Every harness launch is also
wrapped in the fixed macOS Seatbelt policy from ``dogma.runner.policy``: network access and writes
outside the attempt directory are denied, and a wrapper that cannot start is an infrastructure
error, never a retry without the wrapper (ROADMAP I-21).
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import psutil

from dogma.core.errors import ErrorCode, RunnerError
from dogma.runner import policy

REPORT_NAME: Final = "report.json"
STDOUT_NAME: Final = "harness.stdout"
STDERR_NAME: Final = "harness.stderr"
# sandbox-exec exits 65 for a profile it cannot parse and 71 when it cannot apply one. Either,
# with its own message and no report, means the harness never started (see _wrapper_refused).
_WRAPPER_EXIT_CODES: Final = frozenset({65, 71})
_WRAPPER_MESSAGES: Final = ("sandbox_apply:", "sandbox-exec:")
POLL_INTERVAL_S: Final = 0.05
# Bytes the run may add to its directory. ``print`` output is captured in memory by the harness,
# so this catches what reaches disk directly: writes to fd 1 or 2, stderr, and files.
MAX_RUN_DIR_BYTES: Final = 512 * 1024 * 1024
STDERR_TAIL_BYTES: Final = 4000
# Fixed rather than inherited: the runner never reads os.environ (ENGINEERING_STANDARDS §4.6).
SYSTEM_PATH: Final = "/usr/bin:/bin:/usr/sbin:/sbin"
MIB: Final = 1024 * 1024

KillReason = Literal["timeout", "oom", "disk"]


@dataclass(frozen=True)
class SandboxResult:
    """What happened to the harness process. ``kill_reason`` is set when the sandbox killed it."""

    duration_ms: int
    kill_reason: KillReason | None
    exit_code: int
    stderr_tail: str


def scrubbed_env(run_dir: Path) -> dict[str, str]:
    """The only environment the harness sees. ``HOME`` and ``TMPDIR`` stay inside the run dir.

    The child runs with ``-P -s`` rather than ``-I``: ``-I`` also ignores ``PYTHONHASHSEED``, which
    would leave hash randomisation on and break the "same outcome twice" rule. Nothing is lost,
    because this environment is rebuilt from scratch and carries no other ``PYTHON*`` variable.
    """
    return {
        "PATH": SYSTEM_PATH,
        "HOME": str(run_dir),
        # Temporary files (pytest's tmp_path, tempfile, Hypothesis) stay in the writable attempt.
        "TMPDIR": str(run_dir),
        "PYTHONHASHSEED": "0",
        # Imports must not need a writable package or test directory under the policy.
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "4",
        "MPLBACKEND": "Agg",
    }


def execute(
    run_dir: Path,
    *,
    tests: tuple[str, ...],
    timeout_s: int,
    memory_limit_mb: int,
    tests_dir: Path | None = None,
) -> SandboxResult:
    """Run the harness in ``run_dir`` under the Seatbelt policy and enforce the limits.

    ``tests_dir`` is where the test files are read from (the protected bundle). When omitted the
    harness reads them from ``run_dir``, as the standalone CLI always has.

    Raises ``RunnerError`` when the OS sandbox refuses to start the harness. Nothing has run then,
    and the attempt is never retried without the sandbox.
    """
    cmd = [
        sys.executable,
        "-P",  # no cwd or script directory on sys.path: learner files cannot shadow modules
        "-s",  # no user site-packages
        "-m",
        "dogma_harness",
        "--run-dir",
        str(run_dir),
        *(["--tests-dir", str(tests_dir)] if tests_dir is not None else []),
        "--tests",
        *tests,
        "--report",
        str(run_dir / REPORT_NAME),
    ]
    wrapped = policy.sandbox_command(cmd, run_dir)
    started = time.monotonic()
    # Fixtures staged before the spawn are not the run's writes: measure growth from here.
    baseline = _dir_bytes(run_dir)
    proc = _spawn(wrapped, run_dir)
    switch = _KillSwitch(proc.pid)
    watchdog = _Watchdog(proc.pid, run_dir, memory_limit_mb * MIB, baseline, switch)
    watchdog.start()
    try:
        exit_code = _wait(proc, timeout_s, switch)
    finally:
        watchdog.stop()
        watchdog.join()
        _kill_group(proc.pid)  # nothing may outlive the attempt, even a stray child
        if proc.returncode is None:  # the run was aborted (e.g. Ctrl-C) before the wait returned
            proc.wait()
    stderr_tail = read_tail(run_dir / STDERR_NAME)
    if switch.reason is None and _wrapper_refused(exit_code, stderr_tail, run_dir):
        raise RunnerError(
            ErrorCode.RUNNER_CRASHED,
            "the OS sandbox refused to start the test harness; nothing was executed: "
            f"{stderr_tail.splitlines()[-1] if stderr_tail else 'no detail'}",
        )
    return SandboxResult(
        duration_ms=round((time.monotonic() - started) * 1000),
        kill_reason=switch.reason,
        exit_code=exit_code,
        stderr_tail=stderr_tail,
    )


def _wrapper_refused(exit_code: int, stderr_tail: str, run_dir: Path) -> bool:
    """True when sandbox-exec itself failed: its exit status and message, and no harness report.

    A learner can imitate this only by printing the wrapper's text and exiting with the same
    status, which the trusted-user threat model accepts. The cost of a miss is a GradeReport
    error, never execution without the sandbox.
    """
    if exit_code not in _WRAPPER_EXIT_CODES or (run_dir / REPORT_NAME).exists():
        return False
    return any(message in stderr_tail for message in _WRAPPER_MESSAGES)


def read_tail(path: Path, limit_bytes: int = STDERR_TAIL_BYTES) -> str:
    """The last ``limit_bytes`` of a log file, decoded leniently. Empty if it cannot be read."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit_bytes))
            return handle.read().decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def _spawn(cmd: list[str], run_dir: Path) -> subprocess.Popen[bytes]:
    with (run_dir / STDOUT_NAME).open("wb") as out, (run_dir / STDERR_NAME).open("wb") as err:
        try:
            # ``start_new_session`` makes the child its own process-group leader, so the whole
            # tree can be killed with one signal. The argv is fixed, with no shell.
            return subprocess.Popen(  # noqa: S603
                cmd,
                cwd=run_dir,
                env=scrubbed_env(run_dir),
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
        except OSError as exc:
            raise RunnerError(
                ErrorCode.RUNNER_CRASHED, f"cannot start the test harness: {exc}"
            ) from exc


def _wait(proc: subprocess.Popen[bytes], timeout_s: int, switch: _KillSwitch) -> int:
    try:
        return proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        switch.trip("timeout")
        return proc.wait()


def _kill_group(pgid: int) -> None:
    with contextlib.suppress(ProcessLookupError):  # the group is already gone
        os.killpg(pgid, signal.SIGKILL)


class _KillSwitch:
    """Kills the process group at most once and remembers why. The first reason wins."""

    def __init__(self, pgid: int) -> None:
        self._pgid = pgid
        self._lock = threading.Lock()
        self._reason: KillReason | None = None

    @property
    def reason(self) -> KillReason | None:
        with self._lock:
            return self._reason

    def trip(self, reason: KillReason) -> None:
        with self._lock:
            if self._reason is None:
                self._reason = reason
                _kill_group(self._pgid)


class _Watchdog(threading.Thread):
    """Polls the harness tree's RSS and the bytes the run writes, every ``POLL_INTERVAL_S``."""

    def __init__(
        self,
        pid: int,
        run_dir: Path,
        memory_limit_bytes: int,
        baseline_bytes: int,
        switch: _KillSwitch,
    ) -> None:
        super().__init__(name="dogma-runner-watchdog", daemon=True)
        self._pid = pid
        self._run_dir = run_dir
        self._memory_limit_bytes = memory_limit_bytes
        self._baseline_bytes = baseline_bytes
        self._switch = switch
        self._done = threading.Event()

    def run(self) -> None:
        try:
            root = psutil.Process(self._pid)
        except psutil.Error:
            return
        while not self._done.wait(POLL_INTERVAL_S):
            if _tree_rss(root) > self._memory_limit_bytes:
                self._switch.trip("oom")
                return
            if _dir_bytes(self._run_dir) - self._baseline_bytes > MAX_RUN_DIR_BYTES:
                self._switch.trip("disk")
                return

    def stop(self) -> None:
        self._done.set()


def _tree_rss(root: psutil.Process) -> int:
    """Resident memory of ``root`` and all its descendants, in bytes."""
    try:
        procs = [root, *root.children(recursive=True)]
    except psutil.Error:
        return 0
    total = 0
    for proc in procs:
        try:
            total += proc.memory_info().rss
        except psutil.Error:
            continue  # exited between the listing and the read
    return total


def _dir_bytes(root: Path) -> int:
    """Total size of regular files under ``root``. Unreadable entries count as zero."""
    total = 0
    pending = [str(root)]
    while pending:
        current = pending.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(entry.path)
                else:
                    total += entry.stat(follow_symlinks=False).st_size
            except OSError:
                continue
    return total
