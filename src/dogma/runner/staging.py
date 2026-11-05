"""Protected test bundle: tests the learner process can read but cannot change (ROADMAP I-21).

An attempt uses two locations:

- the attempt directory (``var/runs/<id>``), the only place the sandbox allows writes. It keeps the
  submission and fixtures so the preserved artifact looks as it did, and never holds test files;
- a sibling bundle (``var/runs/.<id>.tests-<hex>``) outside the write allowance. It holds the
  selected tests, a snapshot of the submission and the fixtures. Pytest imports and reads from
  here, so ``from submission import ...`` and ``Path(__file__).with_name(...)`` still resolve
  (CONTENT_SPEC §6.4).

The bundle is sealed read-only as defence in depth. Seatbelt is the real write barrier.
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

from dogma.core.errors import ErrorCode, RunnerError
from dogma.runner.meta import ProblemMeta

log = logging.getLogger(__name__)

BUNDLE_MARKER: Final = ".tests-"
_SUBMISSION_FILE: Final = {"python": "submission.py", "sql": "submission.sql"}
_DIR_MODE: Final = 0o555
_FILE_MODE: Final = 0o444
_OWNER_DIR_MODE: Final = 0o755


@contextmanager
def staged_tests(
    run_dir: Path,
    submission: Path,
    problem_dir: Path,
    meta: ProblemMeta,
    tests: tuple[str, ...],
) -> Iterator[Path]:
    """Stage an attempt and yield the protected bundle that holds the tests to run.

    The bundle is removed when the block exits, whether it returns or raises. The attempt
    directory is kept either way. Fixture symlinks are refused before anything is copied.
    """
    fixtures = _checked_fixtures(problem_dir)
    bundle = run_dir.parent / f".{run_dir.name}{BUNDLE_MARKER}{uuid.uuid4().hex[:12]}"
    try:
        _stage_run_dir(run_dir, submission, meta, fixtures)
        _stage_bundle(
            bundle,
            run_dir=run_dir,
            problem_dir=problem_dir,
            meta=meta,
            tests=tests,
            fixtures=fixtures,
        )
        yield bundle
    finally:
        _remove_bundle(bundle)


def _checked_fixtures(problem_dir: Path) -> Path | None:
    """The fixtures directory, or None. Symlinks anywhere in it are refused, never followed.

    Refusing every symlink is stricter than refusing only escaping ones, and content has none.
    """
    fixtures = problem_dir / "fixtures"
    if fixtures.is_symlink():
        raise _symlink_error(fixtures)
    if not fixtures.is_dir():
        return None
    for current, dirnames, filenames in os.walk(fixtures, followlinks=False):
        for name in (*dirnames, *filenames):
            path = Path(current, name)
            if path.is_symlink():
                raise _symlink_error(path)
    return fixtures


def _symlink_error(path: Path) -> RunnerError:
    return RunnerError(
        ErrorCode.CONTENT_SCHEMA_INVALID,
        f"{path} is a symlink; problem fixtures must be plain files and directories",
    )


def _stage_run_dir(
    run_dir: Path, submission: Path, meta: ProblemMeta, fixtures: Path | None
) -> None:
    """Copy the submission and fixtures into the writable, retained attempt directory."""
    try:
        shutil.copyfile(submission, run_dir / _SUBMISSION_FILE[meta.language])
        if fixtures is not None:
            shutil.copytree(fixtures, run_dir / "fixtures")
    except OSError as exc:
        raise RunnerError(ErrorCode.RUNNER_CRASHED, f"cannot stage {run_dir}: {exc}") from exc


def _stage_bundle(
    bundle: Path,
    *,
    run_dir: Path,
    problem_dir: Path,
    meta: ProblemMeta,
    tests: tuple[str, ...],
    fixtures: Path | None,
) -> None:
    """Populate the sealed bundle. The submission snapshot comes from the attempt's own copy."""
    try:
        bundle.mkdir(parents=True, exist_ok=False)
        name = _SUBMISSION_FILE[meta.language]
        shutil.copyfile(run_dir / name, bundle / name)
        for test in tests:
            shutil.copyfile(problem_dir / test, bundle / test)
        if fixtures is not None:
            shutil.copytree(fixtures, bundle / "fixtures")
        _seal(bundle)
    except OSError as exc:
        _remove_bundle(bundle)
        raise RunnerError(ErrorCode.RUNNER_CRASHED, f"cannot stage {bundle}: {exc}") from exc


def _seal(root: Path) -> None:
    """Make everything under ``root`` read-only. The bundle holds no symlinks (see staging)."""
    deepest_first = sorted(root.rglob("*"), key=lambda path: len(path.parts), reverse=True)
    for path in deepest_first:
        path.chmod(_DIR_MODE if path.is_dir() else _FILE_MODE)
    root.chmod(_DIR_MODE)


def _unseal(root: Path) -> None:
    """Restore owner write on every directory so the bundle can be deleted."""
    for path in [root, *root.rglob("*")]:
        if path.is_dir() and not path.is_symlink():
            path.chmod(_OWNER_DIR_MODE)


def _remove_bundle(bundle: Path) -> None:
    """Delete the bundle. A cleanup failure is logged and never replaces the attempt's outcome."""
    if not bundle.exists() and not bundle.is_symlink():
        return
    try:
        _unseal(bundle)
        shutil.rmtree(bundle)
    except OSError as exc:
        log.warning("could not remove protected test bundle %s: %s", bundle, exc)
