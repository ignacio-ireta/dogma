"""Run-artifact retention: completion markers, eligibility, and bounded removal on scratch dirs."""

from __future__ import annotations

import importlib
import json
import os
import shutil
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from dogma.core.config import Settings
from dogma.core.errors import ErrorCode, RunnerError
from dogma.grading.models import RunStatus
from dogma.runner import retention
from dogma.runner import sandbox as sandbox_module
from dogma.runner.retention import MARKER_NAME, is_eligible, mark_completed, prune_runs
from dogma.runner.sandbox import SandboxResult

REPO_ROOT = Path(__file__).resolve().parents[3]
PY_ADD = REPO_ROOT / "tests" / "fixtures" / "problems" / "py-add"
NOW = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)
BEFORE = NOW - timedelta(days=30)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(repo_root=tmp_path / "repo")


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    root.mkdir()
    return root


def _dir(runs: Path, name: str) -> Path:
    path = runs / name
    path.mkdir(parents=True)
    (path / "report.json").write_text("{}", encoding="utf-8")
    return path


def _mark(path: Path, completed_at: datetime) -> None:
    mark_completed(path, completed_at=completed_at)


def _prune(runs: Path, *, apply: bool = False, **kwargs: Any) -> tuple[Path, ...]:
    options: dict[str, Any] = {"protected_ids": frozenset(), "protected_paths": frozenset()}
    options.update(kwargs)
    return prune_runs(runs, before=BEFORE, apply=apply, **options)


# --- mark_completed -------------------------------------------------------------------------


def test_mark_completed_writes_the_lifecycle_marker(runs: Path) -> None:
    run = _dir(runs, "20261001T000000Z-aaaaaaaa")
    mark_completed(run, completed_at=NOW)

    payload = json.loads((run / MARKER_NAME).read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": 1,
        "attempt_id": "20261001T000000Z-aaaaaaaa",
        "completed_at": NOW.isoformat(timespec="microseconds"),
    }


def test_mark_completed_converts_the_time_to_utc(runs: Path) -> None:
    run = _dir(runs, "t1")
    local = NOW.astimezone(timezone(timedelta(hours=-3)))
    mark_completed(run, completed_at=local)
    payload = json.loads((run / MARKER_NAME).read_text(encoding="utf-8"))
    assert payload["completed_at"].endswith("+00:00")
    assert datetime.fromisoformat(payload["completed_at"]) == NOW


def test_mark_completed_refuses_a_naive_time(runs: Path) -> None:
    run = _dir(runs, "t2")
    with pytest.raises(ValueError):
        mark_completed(run, completed_at=datetime(2026, 10, 27, 12, 0))
    assert not (run / MARKER_NAME).exists()


def test_mark_completed_does_not_create_a_missing_directory(runs: Path) -> None:
    missing = runs / "t3"
    with pytest.raises(OSError):
        mark_completed(missing, completed_at=NOW)
    assert not missing.exists()


# --- eligibility ----------------------------------------------------------------------------


def test_only_an_old_completed_unreferenced_run_is_a_candidate(runs: Path) -> None:
    old = _dir(runs, "old-run")
    _mark(old, NOW - timedelta(days=45))
    recent = _dir(runs, "recent-run")
    _mark(recent, NOW - timedelta(days=1))
    future = _dir(runs, "future-run")
    _mark(future, NOW + timedelta(days=5))
    _dir(runs, "active-run")  # no marker: still running, or interrupted, or legacy

    assert _prune(runs) == (old,)


@pytest.mark.parametrize(
    "marker",
    [
        "{not json",
        "[]",
        json.dumps(
            {"schema_version": 2, "attempt_id": "x", "completed_at": "2026-01-01T00:00:00+00:00"}
        ),
        json.dumps(
            {"schema_version": True, "attempt_id": "x", "completed_at": "2026-01-01T00:00:00+00:00"}
        ),
        json.dumps(
            {
                "schema_version": 1,
                "attempt_id": "other",
                "completed_at": "2026-01-01T00:00:00+00:00",
            }
        ),
        json.dumps({"schema_version": 1, "attempt_id": "x", "completed_at": "2026-01-01T00:00:00"}),
        json.dumps({"schema_version": 1, "attempt_id": "x", "completed_at": "yesterday"}),
        json.dumps({"schema_version": 1, "attempt_id": "x"}),
    ],
    ids=[
        "not-json",
        "not-object",
        "future-schema",
        "bool-schema",
        "attempt-mismatch",
        "naive-time",
        "unparseable-time",
        "missing-time",
    ],
)
def test_an_unknown_or_malformed_marker_keeps_the_run(runs: Path, marker: str) -> None:
    run = _dir(runs, "x")
    (run / MARKER_NAME).write_text(marker, encoding="utf-8")
    assert _prune(runs) == ()


def test_a_marker_that_is_a_symlink_is_ignored(runs: Path, tmp_path: Path) -> None:
    run = _dir(runs, "linked-marker")
    real = tmp_path / "elsewhere.json"
    real.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "attempt_id": "linked-marker",
                "completed_at": (NOW - timedelta(days=60)).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    (run / MARKER_NAME).symlink_to(real)
    assert _prune(runs) == ()


def test_children_that_are_not_attempt_directories_are_ignored(runs: Path) -> None:
    (runs / "stray.txt").write_text("x", encoding="utf-8")
    (runs / ".hidden-bundle").mkdir()
    (runs / "with space").mkdir()
    assert _prune(runs) == ()


def test_a_file_named_like_an_attempt_is_not_a_candidate(runs: Path) -> None:
    (runs / "a-file").write_text("x", encoding="utf-8")
    assert _prune(runs) == ()


def test_directory_modification_time_does_not_decide_eligibility(runs: Path) -> None:
    old_mtime = datetime(2000, 1, 1, tzinfo=UTC).timestamp()
    recent_marked = _dir(runs, "recent-but-old-mtime")
    _mark(recent_marked, NOW - timedelta(days=2))
    os.utime(recent_marked, (old_mtime, old_mtime))

    old_marked = _dir(runs, "old-but-new-mtime")
    _mark(old_marked, NOW - timedelta(days=90))
    now_ts = NOW.timestamp()
    os.utime(old_marked, (now_ts, now_ts))

    assert _prune(runs) == (old_marked,)


def test_is_eligible_applies_the_rules_to_one_run() -> None:
    old = NOW - timedelta(days=40)
    assert is_eligible("20261001T000000Z-aaaaaaaa", old, before=BEFORE, protected=False)
    assert not is_eligible("20261001T000000Z-aaaaaaaa", None, before=BEFORE, protected=False)
    assert not is_eligible("20261001T000000Z-aaaaaaaa", old, before=BEFORE, protected=True)
    assert not is_eligible("20261001T000000Z-aaaaaaaa", BEFORE, before=BEFORE, protected=False)
    assert not is_eligible(".bundle", old, before=BEFORE, protected=False)


# --- protection -----------------------------------------------------------------------------


def test_protected_ids_and_paths_are_never_candidates(runs: Path) -> None:
    by_id = _dir(runs, "20250101T000000Z-bbbbbbbb")
    _mark(by_id, NOW - timedelta(days=400))
    by_path = _dir(runs, "20250101T000000Z-cccccccc")
    _mark(by_path, NOW - timedelta(days=400))
    free = _dir(runs, "20250101T000000Z-dddddddd")
    _mark(free, NOW - timedelta(days=400))

    result = _prune(
        runs,
        protected_ids=frozenset({"20250101T000000Z-bbbbbbbb"}),
        protected_paths=frozenset({by_path}),
    )
    assert result == (free,)


def test_a_protected_path_spelled_through_a_symlink_still_protects(
    runs: Path, tmp_path: Path
) -> None:
    kept = _dir(runs, "kept-run")
    _mark(kept, NOW - timedelta(days=90))
    alias = tmp_path / "alias"
    alias.symlink_to(runs, target_is_directory=True)

    assert _prune(runs, protected_paths=frozenset({alias / "kept-run"})) == ()


# --- apply and dry run ----------------------------------------------------------------------


def test_dry_run_lists_candidates_and_deletes_nothing(runs: Path) -> None:
    old = _dir(runs, "old-run")
    _mark(old, NOW - timedelta(days=60))
    assert _prune(runs, apply=False) == (old,)
    assert old.is_dir()


def test_apply_removes_only_the_candidates(runs: Path) -> None:
    old = _dir(runs, "old-run")
    _mark(old, NOW - timedelta(days=60))
    recent = _dir(runs, "recent-run")
    _mark(recent, NOW - timedelta(days=2))
    legacy = _dir(runs, "legacy-run")

    removed = _prune(runs, apply=True)
    assert removed == (old,)
    assert not old.exists()
    assert recent.is_dir()
    assert legacy.is_dir()


def test_a_missing_runs_root_has_no_candidates_and_is_not_created(tmp_path: Path) -> None:
    missing = tmp_path / "no-runs"
    assert _prune(missing, apply=True) == ()
    assert not missing.exists()


def test_a_naive_cutoff_is_rejected(runs: Path) -> None:
    with pytest.raises(ValueError):
        prune_runs(
            runs,
            protected_ids=frozenset(),
            protected_paths=frozenset(),
            before=datetime(2026, 9, 1),
        )


def test_a_symlink_candidate_is_skipped_and_its_target_survives(runs: Path, tmp_path: Path) -> None:
    sentinel = tmp_path / "sentinel-dir"
    sentinel.mkdir()
    (sentinel / "keep.txt").write_text("keep", encoding="utf-8")
    _mark(sentinel, NOW - timedelta(days=90))  # a marker-shaped target must still not qualify
    link = runs / "20250101T000000Z-linkrun"
    link.symlink_to(sentinel, target_is_directory=True)

    assert _prune(runs, apply=True) == ()
    assert link.is_symlink()
    assert (sentinel / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_a_nested_symlink_is_unlinked_and_its_target_survives(runs: Path, tmp_path: Path) -> None:
    victim = tmp_path / "outside.txt"
    victim.write_text("outside", encoding="utf-8")
    run = _dir(runs, "old-with-link")
    _mark(run, NOW - timedelta(days=90))
    (run / "escape").symlink_to(victim)

    assert _prune(runs, apply=True) == (run,)
    assert not run.exists()
    assert victim.read_text(encoding="utf-8") == "outside"


def test_a_symlinked_runs_root_fails_closed(runs: Path, tmp_path: Path) -> None:
    real = tmp_path / "real-runs"
    real.mkdir()
    child = real / "old-run"
    child.mkdir()
    _mark(child, NOW - timedelta(days=90))
    link_root = tmp_path / "linked-runs"
    link_root.symlink_to(real, target_is_directory=True)

    with pytest.raises(RunnerError) as info:
        _prune(link_root, apply=True)
    assert info.value.code == ErrorCode.RUNNER_CRASHED
    assert child.is_dir()


def test_a_candidate_replaced_by_a_symlink_before_removal_is_not_followed(
    runs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    victim = _dir(runs, "old-run")
    _mark(victim, NOW - timedelta(days=90))
    sentinel = tmp_path / "sentinel"
    sentinel.mkdir()
    (sentinel / "precious.txt").write_text("precious", encoding="utf-8")

    original = retention._candidates

    def swap_after_scan(*args: Any, **kwargs: Any) -> Any:
        found = original(*args, **kwargs)
        shutil.move(str(victim), str(tmp_path / "moved-aside"))
        victim.symlink_to(sentinel, target_is_directory=True)
        return found

    monkeypatch.setattr(retention, "_candidates", swap_after_scan)
    assert _prune(runs, apply=True) == ()
    assert (sentinel / "precious.txt").read_text(encoding="utf-8") == "precious"


def test_apply_refuses_when_rmtree_cannot_avoid_symlink_attacks(
    runs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = _dir(runs, "old-run")
    _mark(old, NOW - timedelta(days=90))
    monkeypatch.setattr(shutil.rmtree, "avoids_symlink_attacks", False)

    with pytest.raises(RunnerError) as info:
        _prune(runs, apply=True)
    assert info.value.code == ErrorCode.RUNNER_CRASHED
    assert old.is_dir()
    assert _prune(runs) == (old,), "a dry run does not need the safer removal"


def test_a_failed_removal_is_a_runner_error(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    old = _dir(runs, "old-run")
    _mark(old, NOW - timedelta(days=90))

    def refuse(*args: object, **kwargs: object) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(retention.shutil, "rmtree", refuse)
    with pytest.raises(RunnerError) as info:
        _prune(runs, apply=True)
    assert info.value.code == ErrorCode.RUNNER_CRASHED


def test_a_failed_listing_is_a_runner_error_not_an_empty_result(
    runs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unreadable(*args: object, **kwargs: object) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(retention.os, "scandir", unreadable)
    with pytest.raises(RunnerError) as info:
        _prune(runs, apply=True)
    assert info.value.code == ErrorCode.RUNNER_CRASHED


# --- the run() seam -------------------------------------------------------------------------


def _fake_harness(passed: bool = True) -> Any:
    def fake_execute(run_dir: Path, **_: object) -> SandboxResult:
        status = "passed" if passed else "failed"
        (run_dir / sandbox_module.REPORT_NAME).write_text(
            json.dumps({"status": status, "tests": [], "stdout": ""}), encoding="utf-8"
        )
        return SandboxResult(duration_ms=3, kill_reason=None, exit_code=0, stderr_tail="")

    return fake_execute


def _graded_submission(tmp_path: Path) -> Path:
    path = tmp_path / "work" / "submission.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    return path


def test_a_graded_run_is_marked_completed_after_grading(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_module = importlib.import_module("dogma.runner.run")
    monkeypatch.setattr(sandbox_module, "execute", _fake_harness(passed=True))
    before = datetime.now(UTC)

    report = run_module.run(
        PY_ADD,
        _graded_submission(tmp_path),
        include_hidden=False,
        settings=settings,
        attempt_id="graded-1",
    )

    assert report.status is RunStatus.PASSED
    marker = json.loads((settings.runs_dir / "graded-1" / MARKER_NAME).read_text(encoding="utf-8"))
    assert marker["attempt_id"] == "graded-1"
    completed = datetime.fromisoformat(marker["completed_at"])
    assert before <= completed <= datetime.now(UTC)


def test_an_interrupted_run_stays_unmarked(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_module = importlib.import_module("dogma.runner.run")

    def interrupted(*args: object, **kwargs: object) -> SandboxResult:
        raise RunnerError(ErrorCode.RUNNER_CRASHED, "the process group could not be signalled")

    monkeypatch.setattr(sandbox_module, "execute", interrupted)
    with pytest.raises(RunnerError):
        run_module.run(
            PY_ADD,
            _graded_submission(tmp_path),
            include_hidden=False,
            settings=settings,
            attempt_id="interrupted-1",
        )
    assert (settings.runs_dir / "interrupted-1").is_dir()
    assert not (settings.runs_dir / "interrupted-1" / MARKER_NAME).exists()


def test_a_marker_write_failure_keeps_the_grade(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_module = importlib.import_module("dogma.runner.run")
    monkeypatch.setattr(sandbox_module, "execute", _fake_harness(passed=True))

    def no_space(*args: object, **kwargs: object) -> None:
        raise OSError("no space left on device")

    monkeypatch.setattr(run_module, "mark_completed", no_space)
    report = run_module.run(
        PY_ADD,
        _graded_submission(tmp_path),
        include_hidden=False,
        settings=settings,
        attempt_id="unmarked-1",
    )
    assert report.status is RunStatus.PASSED
