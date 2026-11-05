"""Start, check, abandon and hint-count a timed re-solve (ARCHITECTURE §4.4, §4.7).

These are the use cases the CLI (and later the API) call. The checks and the window write share
one transaction. Restoring the workspace happens after that commit, and keeps the learner's old
file under ``var/workspace-history/`` exactly like ``dogma open --fresh``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from dogma.content.loader import load_topic
from dogma.content.resolve import ProblemRef, resolve_problem, set_aside_workspace, workspace_file
from dogma.core.config import Settings
from dogma.core.errors import ContentError, DogmaError, ErrorCode
from dogma.core.fsio import atomic_write
from dogma.db.errors import StorageErrorCode
from dogma.db.progress import Snapshot
from dogma.db.progress import first_passes as snapshot_first_passes
from dogma.db.resolve_store import (
    abandon_window,
    count_hint,
    first_passes,
    open_window,
    start_window,
)
from dogma.db.session import database_path, open_session
from dogma.learning.planner import READY_STATUSES
from dogma.learning.resolves import (
    ReSolveSchedule,
    TimedResult,
    WindowRecord,
    build_schedules,
    timed_results,
    upcoming,
)


@dataclass(frozen=True)
class StartedWindow:
    window: WindowRecord
    problem: ProblemRef
    workspace: Path
    kept: Path | None  # the learner's previous file, if there was one

    @property
    def budget(self) -> timedelta:
        return timedelta(seconds=self.window.budget_s)


def start(settings: Settings, problem: str, *, now: datetime) -> StartedWindow:
    """Open a timed re-solve of a solved core problem, and restore its starter in the workspace.

    Refused for a problem that is not core, not verified or reviewed, never passed, while another
    window is open, or while recording is switched off (a window that could not be closed by a
    submit would be a trap).
    """
    ref = resolve_problem(settings, problem)
    if not ref.meta.core:
        raise DogmaError(
            ErrorCode.RESOLVE_NOT_CORE,
            f"Timed re-solves are for core problems. Practise {ref.id} untimed with "
            f"dogma open {ref.id}.",
        )
    if ref.meta.status not in READY_STATUSES:
        raise DogmaError(
            ErrorCode.RESOLVE_NOT_READY,
            f"{ref.id} is not verified or reviewed yet, so it cannot be timed.",
        )
    if not settings.record_attempts:
        raise DogmaError(
            ErrorCode.RESOLVE_RECORDING_OFF,
            "Recording is switched off (DOGMA_RECORD_ATTEMPTS=false), so a timed re-solve "
            "could not be closed by a submit. Turn recording on to time a re-solve.",
        )
    if not ref.starter.is_file():
        raise DogmaError(ErrorCode.CONTENT_NOT_FOUND, f"{ref.starter} is missing")
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as db:
        if ref.id not in first_passes(db):
            raise DogmaError(
                ErrorCode.RESOLVE_NOT_SOLVED,
                f"Pass {ref.id} with dogma submit first. A timed re-solve starts from a solve.",
            )
        window = start_window(
            db,
            problem_id=ref.id,
            problem_version=ref.meta.version,
            budget_s=ref.meta.minutes * 60,
            now=now,
        )
    dest = workspace_file(settings, ref)
    kept = set_aside_workspace(settings, ref, at=now)
    atomic_write(dest, ref.starter.read_text(encoding="utf-8"))
    return StartedWindow(window=window, problem=ref, workspace=dest, kept=kept)


def status(settings: Settings, *, now: datetime) -> WindowRecord | None:
    """The open window, if there is one. With no database yet there is none, and none is created."""
    if not database_path(settings).is_file():
        return None
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as db:
        return open_window(db, now=now)


def abandon(settings: Settings, problem: str | None, *, now: datetime) -> WindowRecord:
    """Abandon the open window, on ``problem`` when one is named. No penalty."""
    problem_id = None if problem is None else resolve_problem(settings, problem).id
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as db:
        return abandon_window(db, now=now, problem_id=problem_id)


def core_problem_ids(settings: Settings, topic_ids: Iterable[str]) -> frozenset[str]:
    """The core problems of the given topics, read from their packages. Missing topics add none."""
    found: set[str] = set()
    for topic_id in topic_ids:
        topic_dir = settings.content_dir / topic_id
        if not (topic_dir / "topic.yaml").is_file():
            continue
        try:
            package = load_topic(topic_dir)
        except ContentError:
            continue
        for problem in package.problems:
            if problem.meta is not None and problem.meta.core:
                found.add(f"{topic_id}/{problem.dir.name}")
    return frozenset(found)


@dataclass(frozen=True)
class Readiness:
    """The next due timed re-solves and the last timed results, on core problems only."""

    upcoming: tuple[ReSolveSchedule, ...]
    recent: tuple[TimedResult, ...]


def readiness(
    settings: Settings,
    topic_ids: Iterable[str],
    snapshot: Snapshot,
    now: datetime,
) -> Readiness:
    """What ``dogma progress`` and the API's progress both show about timed re-solves.

    ``topic_ids`` are the topics of the core path. A problem that is not core is never scheduled.
    """
    core_ids = core_problem_ids(settings, topic_ids)
    schedules = {
        problem_id: schedule
        for problem_id, schedule in build_schedules(
            snapshot_first_passes(snapshot), snapshot.windows
        ).items()
        if problem_id in core_ids
    }
    recent = timed_results(window for window in snapshot.windows if window.problem_id in core_ids)
    return Readiness(upcoming=upcoming(schedules), recent=recent)


def open_window_for(settings: Settings, problem_id: str, *, now: datetime) -> WindowRecord | None:
    """The open window if it is on ``problem_id``. Hints ask before they count against it."""
    window = status(settings, now=now)
    return window if window is not None and window.problem_id == problem_id else None


def record_hint(settings: Settings, problem_id: str, *, now: datetime) -> bool:
    """Count one hint against the open window on ``problem_id``. False when there is none."""
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as db:
        return count_hint(db, problem_id=problem_id, now=now)
