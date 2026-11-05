"""Single-writer lockfiles: acquisition, staleness, and release semantics."""

import json
import os
import socket
import threading
from pathlib import Path

import pytest

from docmd.errors import RepoLockedError
from docmd.tracking.lock import REGISTRY_LOCK_NAME, StateLock, lock_path


def write_lock(out: Path, name: str, payload: dict[str, object] | str) -> Path:
    path = lock_path(out, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload)
    path.write_text(text, encoding="utf-8")
    return path


def test_lock_path_layout(tmp_path: Path) -> None:
    assert lock_path(tmp_path, "notes") == tmp_path / ".docmd" / "locks" / "notes.lock"
    assert lock_path(tmp_path, REGISTRY_LOCK_NAME).name == "_registry.lock"


def test_lock_acquire_writes_holder_and_release_removes(tmp_path: Path) -> None:
    with StateLock(tmp_path, "notes", run_id="r1") as lock:
        holder = json.loads(lock.path.read_text(encoding="utf-8"))
        assert holder["pid"] == os.getpid()
        assert holder["hostname"] == socket.gethostname()
        assert holder["run_id"] == "r1"
    assert not lock.path.exists()


def test_lock_released_when_body_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError), StateLock(tmp_path, "notes", run_id="r1"):
        raise RuntimeError("boom")
    assert not lock_path(tmp_path, "notes").exists()


def test_lock_held_by_live_local_pid_raises(tmp_path: Path) -> None:
    write_lock(tmp_path, "notes", {"pid": os.getpid(), "hostname": socket.gethostname()})
    with pytest.raises(RepoLockedError) as excinfo:
        StateLock(tmp_path, "notes", run_id="r2").acquire()
    assert excinfo.value.error_code == "REPO_LOCKED"
    assert excinfo.value.exit_code == 2


def test_lock_with_dead_local_pid_is_stale_and_replaced(tmp_path: Path) -> None:
    write_lock(tmp_path, "notes", {"pid": 99999999, "hostname": socket.gethostname()})
    with StateLock(tmp_path, "notes", run_id="r2") as lock:
        holder = json.loads(lock.path.read_text(encoding="utf-8"))
        assert holder["pid"] == os.getpid()


def test_lock_from_foreign_host_counts_as_live(tmp_path: Path) -> None:
    write_lock(tmp_path, "notes", {"pid": 99999999, "hostname": "some-other-host"})
    with pytest.raises(RepoLockedError):
        StateLock(tmp_path, "notes", run_id="r2").acquire()


def test_lock_with_malformed_payload_is_stale(tmp_path: Path) -> None:
    write_lock(tmp_path, "notes", "not json at all")
    with StateLock(tmp_path, "notes", run_id="r2"):
        pass
    write_lock(tmp_path, "notes", {"hostname": socket.gethostname()})  # no pid
    with StateLock(tmp_path, "notes", run_id="r3"):
        pass


def test_lock_names_are_independent(tmp_path: Path) -> None:
    write_lock(tmp_path, "other", {"pid": os.getpid(), "hostname": socket.gethostname()})
    with StateLock(tmp_path, "notes", run_id="r1"):
        pass  # a held lock for another name does not block this one


def test_wait_s_retries_until_holder_releases(tmp_path: Path) -> None:
    # The registry upsert relies on this: a colliding sync waits out the
    # other run's short critical section instead of failing.
    holder = StateLock(tmp_path, "notes", run_id="r1")
    holder.acquire()
    timer = threading.Timer(0.2, holder.release)
    timer.start()
    try:
        with StateLock(tmp_path, "notes", run_id="r2", wait_s=5.0) as lock:
            assert json.loads(lock.path.read_text(encoding="utf-8"))["run_id"] == "r2"
    finally:
        timer.join()


def test_release_is_idempotent_and_safe_unacquired(tmp_path: Path) -> None:
    lock = StateLock(tmp_path, "notes", run_id="r1")
    lock.release()  # never acquired: no-op
    lock.acquire()
    lock.release()
    lock.release()
    assert not lock.path.exists()


def test_release_leaves_a_lock_that_is_no_longer_ours(tmp_path: Path) -> None:
    # If our lock was (wrongly) replaced by another holder, release must not
    # unlink that holder's lock out from under it.
    lock = StateLock(tmp_path, "notes", run_id="r1")
    lock.acquire()
    write_lock(tmp_path, "notes", {"pid": os.getpid() + 1, "hostname": socket.gethostname()})
    lock.release()
    assert lock.path.exists()
    assert json.loads(lock.path.read_text(encoding="utf-8"))["pid"] == os.getpid() + 1
    lock.path.unlink()


def test_stale_claim_loser_does_not_touch_winners_fresh_lock(tmp_path: Path) -> None:
    # Two waiters judge the same stale lock; the rename-based claim means the
    # loser's takeover is a no-op instead of deleting the winner's fresh lock.
    write_lock(tmp_path, "notes", {"pid": 99999999, "hostname": socket.gethostname()})
    winner = StateLock(tmp_path, "notes", run_id="winner")
    winner.acquire()  # replaces the stale lock
    loser = StateLock(tmp_path, "notes", run_id="loser")
    # The loser saw the stale file, but by claim time the winner holds a live
    # lock: _claim_stale must re-verify and restore, never steal.
    loser._claim_stale()
    holder = json.loads(lock_path(tmp_path, "notes").read_text(encoding="utf-8"))
    assert holder["run_id"] == "winner"
    with pytest.raises(RepoLockedError):
        loser.acquire()
    winner.release()
