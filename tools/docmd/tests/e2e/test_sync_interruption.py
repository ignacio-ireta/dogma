"""Subprocess SIGINT test for sync: graceful interruption and re-run resume. Marked slow."""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COPIES = 150


@pytest.fixture
def big_repo(fixtures_dir: Path, tmp_path: Path) -> Path:
    src = tmp_path / "bigrepo"
    src.mkdir()
    for i in range(COPIES):
        shutil.copy(fixtures_dir / "pdf" / "basic.pdf", src / f"doc_{i:03d}.pdf")
        shutil.copy(fixtures_dir / "docx" / "basic.docx", src / f"doc_{i:03d}.docx")
    return src


def test_sync_sigint_graceful_stop_then_rerun_completes(big_repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "kb"
    repo_out = out / big_repo.name
    cmd = [
        sys.executable,
        "-m",
        "docmd.cli",
        "sync",
        str(big_repo),
        "--out",
        str(out),
        "--no-progress",
    ]
    env = os.environ | {"PYTHONPATH": str(PROJECT_ROOT / "src")}
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Interrupt as soon as a few files are converted but well before the end.
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        done = len(list(repo_out.rglob("*.md"))) if repo_out.exists() else 0
        if done >= 2:
            break
        time.sleep(0.01)
    else:
        proc.kill()
        pytest.fail("sync never produced output in time")
    proc.send_signal(signal.SIGINT)
    stdout, stderr = proc.communicate(timeout=60)

    assert proc.returncode == 130, f"stdout:\n{stdout}\nstderr:\n{stderr}"
    assert "re-run the same docmd sync command" in stdout
    assert "Interrupted" in stderr or "INTERRUPTED" in stdout

    # The snapshot exists and only reflects completed work: every entry's
    # live output is on disk, and not all planned files made it in.
    snapshot_path = out / ".docmd" / "repos" / big_repo.name / "snapshot.json"
    assert snapshot_path.is_file()
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert 0 < len(snapshot["files"]) < 2 * COPIES, "should be stopped mid-run"
    for entry in snapshot["files"].values():
        assert (repo_out / entry["output"]).is_file()

    # changes.json records the interruption.
    run_dir = max((out / "runs").iterdir(), key=lambda p: p.name)
    changes = json.loads((run_dir / "changes.json").read_text(encoding="utf-8"))
    assert changes["counts"]["added"] == 2 * COPIES
    assert changes["counts"]["interrupted"] > 0
    completed = sum(1 for c in changes["changes"] if c["status"] == "completed")
    assert completed == len(snapshot["files"])

    # Re-running the same sync IS resume: it completes the rest.
    rerun = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert rerun.returncode == 0, f"stdout:\n{rerun.stdout}\nstderr:\n{rerun.stderr}"

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert len(snapshot["files"]) == 2 * COPIES
    for source in (p.name for p in big_repo.iterdir()):
        assert source in snapshot["files"]
        assert (repo_out / f"{source}.md").is_file()

    registry = json.loads((out / "registry.json").read_text(encoding="utf-8"))
    entry = registry["repos"][big_repo.name]
    assert entry["file_count"] == 2 * COPIES
    assert entry["run_count"] == 2
