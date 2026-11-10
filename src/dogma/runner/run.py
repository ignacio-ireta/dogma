"""One attempt end to end: static checks, sandboxed harness run, GradeReport.

See ARCHITECTURE §4.2-4.3.

Learner failures are data: every outcome below returns a ``GradeReport``. Only broken problem
directories and broken infrastructure raise ``RunnerError``.
"""

from __future__ import annotations

import logging
import math
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from dogma.core.config import Settings, get_settings
from dogma.core.errors import ErrorCode, RunnerError
from dogma.grading.models import GradeReport, RunStatus, StaticFinding, TestResult
from dogma.grading.static import check_submission
from dogma.runner import policy, sandbox
from dogma.runner.meta import load_meta
from dogma.runner.retention import mark_completed
from dogma.runner.runtime import ensure_available
from dogma.runner.staging import staged_tests

log = logging.getLogger(__name__)

PUBLIC_TESTS = "test_public.py"
HIDDEN_TESTS = "test_hidden.py"
_ATTEMPT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_HARNESS_STATUS: dict[str, RunStatus] = {
    "passed": RunStatus.PASSED,
    "failed": RunStatus.FAILED,
    "error": RunStatus.ERROR,
}
_MIB = 1024 * 1024


class _HarnessReport(BaseModel):
    """The JSON the harness writes. Validated here because the learner's code shares its dir."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    status: Literal["passed", "failed", "error"]
    tests: tuple[TestResult, ...] = ()
    stdout: str = ""
    collection_error: str | None = None


def effective_timeout_s(problem_timeout_s: int | None, cfg: Settings) -> int:
    """The wall-clock limit for one run: the problem's own limit, or the default, scaled.

    ``run_timeout_scale`` is 1 on the learner's machine. A slower machine raises it so that a
    limit calibrated elsewhere still means "this run is stuck", not "this machine is slow".
    """
    base = problem_timeout_s if problem_timeout_s is not None else cfg.run_timeout_s
    return math.ceil(base * cfg.run_timeout_scale)


@dataclass(frozen=True)
class _Identity:
    """The ids every GradeReport carries, gathered once per attempt."""

    attempt_id: str
    problem_id: str
    problem_version: int
    runtime: str


def _report(
    ident: _Identity,
    *,
    status: RunStatus,
    tests: tuple[TestResult, ...] = (),
    static: tuple[StaticFinding, ...] = (),
    stdout: str = "",
    duration_ms: int = 0,
    message: str = "",
) -> GradeReport:
    return GradeReport(
        attempt_id=ident.attempt_id,
        problem_id=ident.problem_id,
        problem_version=ident.problem_version,
        runtime=ident.runtime,
        status=status,
        tests=tests,
        static=static,
        stdout=stdout,
        duration_ms=duration_ms,
        message=message,
    )


def new_attempt_id() -> str:
    """UTC timestamp plus a short random suffix; sorts by time and is unique in practice."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def run(
    problem_dir: Path,
    submission: Path,
    *,
    include_hidden: bool,
    settings: Settings | None = None,
    attempt_id: str | None = None,
    memory_limit_mb: int | None = None,
) -> GradeReport:
    """Grade ``submission`` against a problem.

    Static checks run first; a finding returns ``rejected`` and nothing executes. Otherwise the
    submission runs under ``var/runs/<attempt_id>/`` (``settings.runs_dir``) with the public tests,
    and the hidden ones too when ``include_hidden``. Learner failures never raise.

    Raises ``RunnerError`` for a broken problem directory or infrastructure failures.
    """
    cfg = settings if settings is not None else get_settings()
    meta = load_meta(problem_dir)
    attempt = attempt_id if attempt_id is not None else new_attempt_id()
    if not _ATTEMPT_ID.fullmatch(attempt):
        raise ValueError(f"attempt id must be a short path-safe token, got {attempt!r}")
    tests = (PUBLIC_TESTS, HIDDEN_TESTS) if include_hidden else (PUBLIC_TESTS,)
    _check_problem_files(problem_dir, tests)
    ident = _Identity(
        attempt_id=attempt,
        problem_id=meta.id,
        problem_version=meta.version,
        runtime=meta.runtime,
    )

    findings = check_submission(
        submission,
        language=meta.language,
        entry=meta.entry,
        banned_imports=meta.banned_imports,
    )
    if findings:
        return _report(
            ident,
            status=RunStatus.REJECTED,
            static=tuple(findings),
            message="rejected before running: fix the findings listed",
        )

    ensure_available(meta.runtime)
    policy.ensure_supported()
    run_dir = _create_run_dir(cfg.runs_dir, attempt)
    timeout_s = effective_timeout_s(meta.timeout_s, cfg)
    limit_mb = memory_limit_mb if memory_limit_mb is not None else cfg.run_memory_mb
    with staged_tests(run_dir, submission, problem_dir, meta, tests) as tests_dir:
        result = sandbox.execute(
            run_dir,
            tests=tests,
            tests_dir=tests_dir,
            timeout_s=timeout_s,
            memory_limit_mb=limit_mb,
        )
    report = _grade(ident, run_dir, result, timeout_s=timeout_s, memory_limit_mb=limit_mb)
    try:  # after grading, child cleanup and bundle removal; an unmarked run is simply kept
        mark_completed(run_dir, completed_at=datetime.now(UTC))
    except OSError as exc:
        log.warning("cannot mark %s as completed: %s", run_dir, exc)
    return report


def _check_problem_files(problem_dir: Path, tests: tuple[str, ...]) -> None:
    for name in tests:
        if not (problem_dir / name).is_file():
            raise RunnerError(ErrorCode.CONTENT_NOT_FOUND, f"{problem_dir} has no {name}")


def _create_run_dir(runs_dir: Path, attempt: str) -> Path:
    run_dir = runs_dir / attempt
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise RunnerError(
            ErrorCode.RUNNER_CRASHED, f"attempt directory already exists: {run_dir}"
        ) from exc
    except OSError as exc:
        raise RunnerError(ErrorCode.RUNNER_CRASHED, f"cannot create {run_dir}: {exc}") from exc
    return run_dir


def _grade(
    ident: _Identity,
    run_dir: Path,
    result: sandbox.SandboxResult,
    *,
    timeout_s: int,
    memory_limit_mb: int,
) -> GradeReport:
    """Map what the sandbox observed to a report. A kill reason outranks any report on disk."""
    duration = result.duration_ms
    if result.kill_reason == "timeout":
        return _report(
            ident,
            status=RunStatus.TIMEOUT,
            duration_ms=duration,
            message=f"the run exceeded its {timeout_s} s time limit",
        )
    if result.kill_reason == "oom":
        return _report(
            ident,
            status=RunStatus.OOM,
            duration_ms=duration,
            message=f"the run exceeded its {memory_limit_mb} MB memory limit",
        )
    if result.kill_reason == "disk":
        return _report(
            ident,
            status=RunStatus.ERROR,
            duration_ms=duration,
            message=f"the run wrote more than {sandbox.MAX_RUN_DIR_BYTES // _MIB} MB to disk",
        )
    report_path = run_dir / sandbox.REPORT_NAME
    if not report_path.is_file():
        detail = f":\n{result.stderr_tail}" if result.stderr_tail else ""
        return _report(
            ident,
            status=RunStatus.ERROR,
            duration_ms=duration,
            message=f"the test harness exited without a report{detail}",
        )
    try:
        parsed = _HarnessReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        detail = f":\n{result.stderr_tail}" if result.stderr_tail else ""
        return _report(
            ident,
            status=RunStatus.ERROR,
            duration_ms=duration,
            message=f"the test harness wrote an unreadable report{detail}",
        )
    return _report(
        ident,
        status=_HARNESS_STATUS[parsed.status],
        tests=parsed.tests,
        stdout=parsed.stdout,
        duration_ms=duration,
        message=parsed.collection_error or "",
    )
