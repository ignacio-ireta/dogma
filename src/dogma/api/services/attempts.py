"""Grade the workspace file, record it the way the CLI does, and read recorded attempts back.

Recording rule (ARCHITECTURE §4.7): the API only grades the learner's own workspace file, so it
records by default, subject to ``settings.record_attempts``. A storage failure keeps the grade:
the report is still returned, with ``recorded`` false and a warning.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Literal

from dogma.api.deps import Clock
from dogma.api.ids import is_attempt_id
from dogma.api.schemas.attempts import AttemptCreated, AttemptRecord, LearnerReport, LearnerTest
from dogma.api.services import resolves
from dogma.api.services.content import problem_ref
from dogma.content.resolve import workspace_file
from dogma.core.config import Settings
from dogma.core.errors import DogmaError, ErrorCode
from dogma.db.errors import StorageError, StorageErrorCode
from dogma.db.models import Attempt, AttemptMode
from dogma.db.session import database_path, open_session
from dogma.db.store import record_attempt
from dogma.grading.display import findings_in_words, learner_message, status_headline
from dogma.grading.models import GradeReport
from dogma.learning.resolves import ClosedWindow
from dogma.runner import attempt as runner_attempt

log = logging.getLogger(__name__)


def create(
    settings: Settings,
    problem_id: str,
    mode: Literal["run", "submit"],
    *,
    clock: Clock,
) -> AttemptCreated:
    ref = problem_ref(settings, problem_id)
    source = workspace_file(settings, ref)
    if not source.is_file():
        raise DogmaError(
            ErrorCode.API_WORKSPACE_MISSING,
            f"no workspace file for {ref.id}: save your solution first (PUT /workspace)",
        )
    attempt_mode = AttemptMode(mode)
    started_at = clock()
    report = runner_attempt.grade_attempt(
        ref.dir,
        source,
        include_hidden=attempt_mode is AttemptMode.SUBMIT,
        settings=settings,
    )
    submitted_at = clock()
    recorded, warning, closed = _record(
        settings,
        report,
        mode=attempt_mode,
        core=ref.meta.core,
        started_at=started_at,
        submitted_at=submitted_at,
    )
    return AttemptCreated(
        mode=mode,
        recorded=recorded,
        record_warning=warning,
        report=learner_report(report),
        timed_resolve=resolves.verdict_out(closed, now=submitted_at),
    )


def learner_report(report: GradeReport) -> LearnerReport:
    """The report the API returns. Hidden-test text is reduced to one safe line (Step 0 safety).

    The raw GradeReport is never returned: its hidden messages can name inputs and values.
    """
    return LearnerReport(
        attempt_id=report.attempt_id,
        problem_id=report.problem_id,
        problem_version=report.problem_version,
        runtime=report.runtime,
        status=report.status,
        headline=status_headline(report.status),
        tests=[
            LearnerTest(
                name=test.name,
                file=test.file,
                outcome=test.outcome,
                message=learner_message(test),
                duration_ms=test.duration_ms,
            )
            for test in report.tests
        ],
        static=list(report.static),
        findings=findings_in_words(report),
        stdout=report.stdout,
        duration_ms=report.duration_ms,
        message=report.message,
    )


def read(settings: Settings, attempt_id: str) -> AttemptRecord:
    if not is_attempt_id(attempt_id) or not database_path(settings).is_file():
        raise _not_found(attempt_id)
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        row = session.get(Attempt, attempt_id)
        if row is None:
            raise _not_found(attempt_id)
        return AttemptRecord(
            id=row.id,
            problem_id=row.problem_id,
            problem_version=row.problem_version,
            mode=row.mode,
            status=row.status,
            public_passed=row.public_passed,
            public_total=row.public_total,
            hidden_passed=row.hidden_passed,
            hidden_total=row.hidden_total,
            hints_used=row.hints_used,
            started_at=row.started_at,
            submitted_at=row.submitted_at,
            duration_ms=row.duration_ms,
        )


def _record(
    settings: Settings,
    report: GradeReport,
    *,
    mode: AttemptMode,
    core: bool,
    started_at: datetime,
    submitted_at: datetime,
) -> tuple[bool, str | None, ClosedWindow | None]:
    if not settings.record_attempts:
        return False, "attempt recording is turned off", None
    try:
        closed = record_attempt(
            settings,
            report,
            mode=mode,
            core=core,
            started_at=started_at,
            submitted_at=submitted_at,
        )
    except StorageError as exc:
        log.warning("attempt %s not recorded: %s", report.attempt_id, exc.code)
        return False, f"[{exc.code}] attempt not recorded: {exc.message}", None
    return True, None, closed


def _not_found(attempt_id: str) -> DogmaError:
    return DogmaError(ErrorCode.API_ATTEMPT_NOT_FOUND, f"no recorded attempt {attempt_id!r}")
