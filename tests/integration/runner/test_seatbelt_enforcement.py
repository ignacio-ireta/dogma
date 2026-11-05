"""Real macOS Seatbelt enforcement: the denials are checked against actual bytes and sockets.

These run on a supported host (Darwin with /usr/bin/sandbox-exec). They are not skipped there:
an unsupported host gets the unit-level fail-closed tests instead.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from dogma.grading.models import RunStatus
from dogma.runner import policy, run, sandbox

_SUPPORTED = sys.platform == "darwin" and policy.SANDBOX_EXEC.is_file()
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _SUPPORTED, reason="requires Darwin with /usr/bin/sandbox-exec"),
]
DENIED = "PermissionError"


def _runs_root(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    root.mkdir()
    return root


def _sandboxed(script: str, run_dir: Path) -> subprocess.CompletedProcess[str]:
    """Run ``script`` the way the runner does: scrubbed env, cwd in the run dir, wrapped argv."""
    return subprocess.run(
        policy.sandbox_command([sys.executable, "-P", "-s", "-c", script], run_dir),
        cwd=run_dir,
        env=sandbox.scrubbed_env(run_dir),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_learner_process_writes_inside_its_run_dir(tmp_path):
    run_dir = _runs_root(tmp_path) / "attempt"
    run_dir.mkdir()
    target = run_dir / "ok.txt"
    result = _sandboxed(f"open({str(target)!r}, 'wb').write(b'inside')", run_dir)
    assert result.returncode == 0, result.stderr
    assert target.read_bytes() == b"inside"


def test_sibling_file_cannot_be_rewritten_created_renamed_chmodded_removed_or_mkdired(tmp_path):
    runs = _runs_root(tmp_path)
    run_dir = runs / "attempt"
    run_dir.mkdir()
    sentinel = runs / "sentinel.txt"
    sentinel.write_bytes(b"original")
    script = f"""
import json, os
sentinel = {str(sentinel)!r}
def attempt(action):
    try:
        action()
    except OSError as exc:
        return type(exc).__name__
    return "allowed"
out = {{
    "rewrite": attempt(lambda: open(sentinel, "wb").write(b"hacked")),
    "create": attempt(lambda: open(sentinel + ".new", "wb").write(b"x")),
    "rename": attempt(lambda: os.rename(sentinel, sentinel + ".moved")),
    "chmod": attempt(lambda: os.chmod(sentinel, 0o777)),
    "unlink": attempt(lambda: os.unlink(sentinel)),
    "mkdir": attempt(lambda: os.mkdir(sentinel + ".dir")),
}}
print(json.dumps(out))
"""
    result = _sandboxed(script, run_dir)
    assert result.returncode == 0, result.stderr
    outcomes = json.loads(result.stdout)
    assert outcomes == dict.fromkeys(outcomes, DENIED), outcomes
    assert sentinel.read_bytes() == b"original"
    assert sorted(p.name for p in runs.iterdir()) == ["attempt", "sentinel.txt"]


def test_symlink_inside_the_run_dir_cannot_write_through_to_the_sibling(tmp_path):
    runs = _runs_root(tmp_path)
    run_dir = runs / "attempt"
    run_dir.mkdir()
    sentinel = runs / "sentinel.txt"
    sentinel.write_bytes(b"original")
    (run_dir / "link.txt").symlink_to(sentinel)
    script = f"""
import json
try:
    open({str(run_dir / "link.txt")!r}, "wb").write(b"via link")
    print(json.dumps("allowed"))
except OSError as exc:
    print(json.dumps(type(exc).__name__))
"""
    result = _sandboxed(script, run_dir)
    assert json.loads(result.stdout) == DENIED, result.stderr
    assert sentinel.read_bytes() == b"original"


def test_a_child_process_inherits_the_denial(tmp_path):
    runs = _runs_root(tmp_path)
    run_dir = runs / "attempt"
    run_dir.mkdir()
    sentinel = runs / "sentinel.txt"
    sentinel.write_bytes(b"original")
    inner = f"open({str(sentinel)!r}, 'wb').write(b'child')"
    script = f"""
import subprocess, sys
child = subprocess.run([sys.executable, "-P", "-s", "-c", {inner!r}], capture_output=True)
print(child.returncode)
open({str(run_dir / "child-ok.txt")!r}, "wb").write(b"parent wrote inside")
"""
    result = _sandboxed(script, run_dir)
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip()) != 0, "the child reached the sibling file"
    assert sentinel.read_bytes() == b"original"
    assert (run_dir / "child-ok.txt").read_bytes() == b"parent wrote inside"


def test_dev_null_is_writable_but_no_other_system_path_is(tmp_path):
    run_dir = _runs_root(tmp_path) / "attempt"
    run_dir.mkdir()
    script = """
import json, os
out = {}
with open(os.devnull, "w") as sink:
    sink.write("discarded")
out["devnull"] = "ok"
for name, path in [("tty", "/dev/tty"), ("etc", "/etc/hosts-dogma-probe")]:
    try:
        open(path, "w").close()
        out[name] = "allowed"
    except OSError as exc:
        out[name] = type(exc).__name__
print(json.dumps(out))
"""
    result = _sandboxed(script, run_dir)
    assert json.loads(result.stdout) == {"devnull": "ok", "tty": DENIED, "etc": DENIED}


def test_loopback_connections_are_denied_while_the_host_connects(tmp_path):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(4)
    listener.settimeout(5)
    port = listener.getsockname()[1]
    run_dir = _runs_root(tmp_path) / "attempt"
    run_dir.mkdir()
    try:
        host_client = socket.create_connection(("127.0.0.1", port), timeout=5)
        accepted, _ = listener.accept()  # the ordinary parent path works, so the probe is real
        accepted.close()
        host_client.close()
        script = f"""
import json, socket
probe = socket.socket()
probe.settimeout(5)
try:
    probe.connect(("127.0.0.1", {port}))
    print(json.dumps("connected"))
except OSError as exc:
    print(json.dumps(type(exc).__name__))
"""
        result = _sandboxed(script, run_dir)
        assert json.loads(result.stdout) == DENIED, result.stderr
        listener.settimeout(0.5)
        with pytest.raises(TimeoutError):
            listener.accept()  # no learner connection ever reached the listener
    finally:
        listener.close()


def test_confinement_holds_for_run_dirs_with_unusual_characters(tmp_path):
    runs = _runs_root(tmp_path)
    run_dir = runs / 'odd "quoted" (paren) ünï; $HOME'
    run_dir.mkdir()
    sentinel = runs / "sentinel.txt"
    sentinel.write_bytes(b"original")
    script = f"""
import json
out = {{}}
try:
    open({str(run_dir / "in.txt")!r}, "wb").write(b"in")
    out["inside"] = "ok"
except OSError as exc:
    out["inside"] = type(exc).__name__
try:
    open({str(sentinel)!r}, "wb").write(b"hacked")
    out["sibling"] = "allowed"
except OSError as exc:
    out["sibling"] = type(exc).__name__
print(json.dumps(out))
"""
    result = _sandboxed(script, run_dir)
    assert json.loads(result.stdout) == {"inside": "ok", "sibling": DENIED}, result.stderr
    assert (run_dir / "in.txt").read_bytes() == b"in"
    assert sentinel.read_bytes() == b"original"


def test_tmp_path_tempfile_and_hypothesis_work_under_enforcement(settings, tmp_path, submit):
    problem = tmp_path / "problem"
    problem.mkdir()
    (problem / "meta.yaml").write_text(
        "id: fixtures/tmp-probe/p01\nversion: 1\nlanguage: python\nruntime: py-numpy\n"
        "entry: add\ntimeout_s: 60\n",
        encoding="utf-8",
    )
    (problem / "test_public.py").write_text(
        "import tempfile\n"
        "from hypothesis import given, settings, strategies as st\n"
        "from submission import add\n\n"
        "def test_pytest_tmp_path_is_writable(tmp_path):\n"
        "    (tmp_path / 'note.txt').write_text('ok')\n"
        "    assert (tmp_path / 'note.txt').read_text() == 'ok'\n\n"
        "def test_tempfile_lands_in_the_attempt_and_is_writable():\n"
        "    import os\n"
        "    assert os.path.realpath(tempfile.gettempdir()) == os.path.realpath(os.getcwd())\n"
        "    scratch = tempfile.mkdtemp()\n"
        "    with open(os.path.join(scratch, 'x'), 'w') as handle:\n"
        "        handle.write('x')\n\n"
        "@settings(max_examples=5, deadline=None)\n"
        "@given(st.integers(min_value=-10, max_value=10))\n"
        "def test_hypothesis_runs_with_its_database(value):\n"
        "    assert add(value, 0) == value\n",
        encoding="utf-8",
    )
    sub = submit("def add(a, b):\n    return a + b\n")
    report = run(problem, sub, include_hidden=False, settings=settings, attempt_id="tmp-probe")
    assert report.status is RunStatus.PASSED, report.message or [t.message for t in report.tests]
    assert report.counts("public") == (3, 3)
