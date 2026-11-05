"""Run-artifact retention: completion markers and explicit, conservative pruning (§4.2).

A graded run is marked completed by the parent after grading and child cleanup. Only marked,
old, unreferenced run directories can be removed, and only by an explicit command. Nothing here
reads the database: the caller passes the protected ids and paths (ENGINEERING_STANDARDS §4.7).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from dogma.core.errors import ErrorCode, RunnerError
from dogma.core.fsio import atomic_write

MARKER_NAME: Final = ".dogma-completed.json"
MARKER_SCHEMA_VERSION: Final = 1
# The same rule the runner applies to attempt ids. Temporary names start with a dot and never match.
_ATTEMPT_ID: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_MAX_MARKER_BYTES: Final = 4096


@dataclass(frozen=True)
class _Candidate:
    path: Path
    name: str
    identity: tuple[int, int]  # (st_dev, st_ino) of the directory when it was inspected
    completed_at: datetime


def _is_aware(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() is not None


def is_eligible(
    attempt_id: str, completed_at: datetime | None, *, before: datetime, protected: bool
) -> bool:
    """The rule for one directory: an attempt-shaped name, a valid completion time, old enough,
    and not referenced by the database. A missing marker (active, interrupted or legacy) is never
    eligible."""
    if protected or completed_at is None or not _ATTEMPT_ID.fullmatch(attempt_id):
        return False
    return completed_at < before


def mark_completed(run_dir: Path, *, completed_at: datetime) -> None:
    """Record that ``run_dir`` finished grading and its child is gone. Atomic; UTC on disk.

    Raises ``OSError`` if the directory is missing or is a link. The marker records lifecycle only,
    never mastery or whether the learner recorded the attempt.
    """
    if not _is_aware(completed_at):
        raise ValueError("completed_at must be timezone-aware")
    attempt_id = run_dir.name
    if not _ATTEMPT_ID.fullmatch(attempt_id):
        raise ValueError(f"run directory name is not an attempt id: {attempt_id!r}")
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise OSError(f"{run_dir} is not a run directory")
    payload = {
        "schema_version": MARKER_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "completed_at": completed_at.astimezone(UTC).isoformat(timespec="microseconds"),
    }
    atomic_write(run_dir / MARKER_NAME, json.dumps(payload, sort_keys=True) + "\n")


def _read_marker(run_dir: Path, attempt_id: str) -> datetime | None:
    """The completion time in a well-formed marker, or None.

    Links, non-regular files and oversized files are refused. Any unreadable or unexpected content
    counts as no marker, so the directory is kept.
    """
    raw = _read_small_file(run_dir / MARKER_NAME)
    if raw is None:
        return None
    try:
        payload: object = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return _completion_time(payload, attempt_id)


def _read_small_file(path: Path) -> bytes | None:
    """The bytes of a regular, non-link file no larger than the marker limit, else None."""
    try:
        info = os.lstat(path)
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_MARKER_BYTES:
        return None
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as handle:
            data = handle.read(_MAX_MARKER_BYTES + 1)
    except OSError:
        return None
    return data if len(data) <= _MAX_MARKER_BYTES else None


def _completion_time(payload: object, attempt_id: str) -> datetime | None:
    """Validate a parsed marker: exact schema version, matching attempt id, aware timestamp."""
    if not isinstance(payload, dict):
        return None
    schema = payload.get("schema_version")
    if type(schema) is not int or schema != MARKER_SCHEMA_VERSION:
        return None
    if payload.get("attempt_id") != attempt_id:
        return None
    text = payload.get("completed_at")
    if not isinstance(text, str):
        return None
    try:
        completed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return completed.astimezone(UTC) if _is_aware(completed) else None


def _real_directory(path: Path) -> os.stat_result | None:
    """The lstat of ``path`` when it is an ordinary directory (not a link), else None."""
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    if not stat.S_ISDIR(info.st_mode):
        return None
    return info


def _resolved_set(paths: frozenset[Path]) -> frozenset[Path]:
    return frozenset(Path(os.path.realpath(p)) for p in paths)


def _candidates(
    runs_dir: Path,
    *,
    protected_ids: frozenset[str],
    protected_paths: frozenset[Path],
    before: datetime,
) -> list[_Candidate]:
    """Scan the immediate children of ``runs_dir`` and keep the ones that pass every rule."""
    try:
        root = os.lstat(runs_dir)
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise RunnerError(ErrorCode.RUNNER_CRASHED, f"cannot inspect {runs_dir}: {exc}") from exc
    if not stat.S_ISDIR(root.st_mode):
        raise RunnerError(
            ErrorCode.RUNNER_CRASHED, f"{runs_dir} must be an ordinary directory, not a link"
        )
    try:
        with os.scandir(runs_dir) as entries:
            names = sorted(entry.name for entry in entries)
    except OSError as exc:
        raise RunnerError(ErrorCode.RUNNER_CRASHED, f"cannot list {runs_dir}: {exc}") from exc

    protected_real = _resolved_set(protected_paths)
    found: list[_Candidate] = []
    for name in names:
        if not _ATTEMPT_ID.fullmatch(name):
            continue
        path = runs_dir / name
        try:
            info = _real_directory(path)
        except OSError as exc:
            raise RunnerError(ErrorCode.RUNNER_CRASHED, f"cannot inspect {path}: {exc}") from exc
        if info is None:
            continue
        protected = name in protected_ids or Path(os.path.realpath(path)) in protected_real
        completed = _read_marker(path, name)
        if completed is None or not is_eligible(
            name, completed, before=before, protected=protected
        ):
            continue
        found.append(_Candidate(path, name, (info.st_dev, info.st_ino), completed))
    return found


def _still_eligible(
    candidate: _Candidate,
    *,
    protected_ids: frozenset[str],
    protected_real: frozenset[Path],
    before: datetime,
) -> bool:
    """Re-check a candidate right before removal: same directory, not a link, same marker."""
    try:
        info = _real_directory(candidate.path)
    except OSError:
        return False
    if info is None or (info.st_dev, info.st_ino) != candidate.identity:
        return False
    protected = (
        candidate.name in protected_ids or Path(os.path.realpath(candidate.path)) in protected_real
    )
    completed = _read_marker(candidate.path, candidate.name)
    return completed == candidate.completed_at and is_eligible(
        candidate.name, completed, before=before, protected=protected
    )


def prune_runs(
    runs_dir: Path,
    *,
    protected_ids: frozenset[str],
    protected_paths: frozenset[Path],
    before: datetime,
    apply: bool = False,
) -> tuple[Path, ...]:
    """Find (and with ``apply``, remove) completed, old, unreferenced run directories.

    Dry run returns the candidates and touches nothing. With ``apply`` it returns the directories
    it removed. Removal uses ``shutil.rmtree``, which unlinks nested links rather than following
    them; apply fails closed when that function cannot avoid symlink attacks on this platform.
    Any listing or removal error raises ``RunnerError``, so protection is never silently empty.
    """
    if not _is_aware(before):
        raise ValueError("the cutoff must be timezone-aware")
    if apply and not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
        raise RunnerError(
            ErrorCode.RUNNER_CRASHED,
            "this platform cannot remove run directories without following links; not pruning",
        )
    candidates = _candidates(
        runs_dir,
        protected_ids=protected_ids,
        protected_paths=protected_paths,
        before=before,
    )
    if not apply:
        return tuple(c.path for c in candidates)

    protected_real = _resolved_set(protected_paths)
    removed: list[Path] = []
    for candidate in candidates:
        if not _still_eligible(
            candidate, protected_ids=protected_ids, protected_real=protected_real, before=before
        ):
            continue
        try:
            shutil.rmtree(candidate.path)
        except OSError as exc:
            done = len(removed)
            raise RunnerError(
                ErrorCode.RUNNER_CRASHED,
                f"cannot remove {candidate.path} ({done} removed before this): {exc}",
            ) from exc
        removed.append(candidate.path)
    return tuple(removed)
