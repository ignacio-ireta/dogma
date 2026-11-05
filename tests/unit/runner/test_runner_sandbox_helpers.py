"""Pure helpers of the sandbox: environment, log tails, directory size, the kill switch."""

from __future__ import annotations

import psutil

from dogma.runner import sandbox


def test_environment_is_scrubbed_to_the_documented_keys(tmp_path):
    env = sandbox.scrubbed_env(tmp_path)
    assert set(env) == {
        "PATH",
        "HOME",
        "TMPDIR",
        "PYTHONHASHSEED",
        "PYTHONDONTWRITEBYTECODE",
        "OMP_NUM_THREADS",
        "MPLBACKEND",
    }
    assert env["PYTHONDONTWRITEBYTECODE"] == "1", "imports must not need a writable package dir"
    assert env["HOME"] == str(tmp_path)
    assert env["TMPDIR"] == str(tmp_path), "temporary files must stay inside the run directory"
    assert "PYTHONPATH" not in env


def test_path_is_the_system_default_not_the_callers(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", "/some/user/bin")
    assert sandbox.scrubbed_env(tmp_path)["PATH"] == sandbox.SYSTEM_PATH


def test_read_tail_returns_only_the_end_of_a_long_log(tmp_path):
    log = tmp_path / "harness.stderr"
    log.write_text("a" * 10_000 + "THE END", encoding="utf-8")
    tail = sandbox.read_tail(log, limit_bytes=100)
    assert tail.endswith("THE END")
    assert len(tail) <= 100


def test_read_tail_of_a_missing_file_is_empty(tmp_path):
    assert sandbox.read_tail(tmp_path / "absent.log") == ""


def test_dir_bytes_counts_nested_regular_files(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"x" * 100)
    (tmp_path / "sub" / "deeper").mkdir(parents=True)
    (tmp_path / "sub" / "deeper" / "b.bin").write_bytes(b"y" * 250)
    assert sandbox._dir_bytes(tmp_path) == 350


def test_dir_bytes_of_a_missing_directory_is_zero(tmp_path):
    assert sandbox._dir_bytes(tmp_path / "nope") == 0


def test_kill_switch_keeps_the_first_reason_and_kills_once(monkeypatch):
    killed: list[int] = []
    monkeypatch.setattr(sandbox, "_kill_group", killed.append)
    switch = sandbox._KillSwitch(4242)
    assert switch.reason is None
    switch.trip("oom")
    switch.trip("timeout")
    assert switch.reason == "oom"
    assert killed == [4242]


def test_tree_rss_of_this_process_is_positive():
    assert sandbox._tree_rss(psutil.Process()) > 0


def test_unreadable_subdirectory_is_skipped_when_measuring_size(tmp_path):
    (tmp_path / "readable.txt").write_bytes(b"x" * 10)
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "hidden.bin").write_bytes(b"y" * 1000)
    locked.chmod(0)
    try:
        assert sandbox._dir_bytes(tmp_path) == 10
    finally:
        locked.chmod(0o755)
