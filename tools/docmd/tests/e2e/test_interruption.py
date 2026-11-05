"""Subprocess SIGINT test: graceful interruption and resume. Marked slow."""

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
def big_tree(fixtures_dir: Path, tmp_path: Path) -> Path:
    src = tmp_path / "many"
    src.mkdir()
    for i in range(COPIES):
        shutil.copy(fixtures_dir / "pdf" / "basic.pdf", src / f"doc_{i:03d}.pdf")
        shutil.copy(fixtures_dir / "docx" / "basic.docx", src / f"doc_{i:03d}.docx")
    return src


def test_sigint_graceful_stop_then_resume(big_tree: Path, tmp_path: Path) -> None:
    out = tmp_path / "md"
    cmd = [
        sys.executable,
        "-m",
        "docmd.cli",
        "scan",
        str(big_tree),
        "--out",
        str(out),
        "--workers",
        "1",
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
        done = len(list(out.rglob("*.md"))) if out.exists() else 0
        if done >= 2:
            break
        time.sleep(0.01)
    else:
        proc.kill()
        pytest.fail("run never produced output in time")
    proc.send_signal(signal.SIGINT)
    stdout, stderr = proc.communicate(timeout=30)

    assert proc.returncode == 130, f"stdout:\n{stdout}\nstderr:\n{stderr}"
    assert "Resume with:" in stdout
    assert "Interrupted" in stderr or "INTERRUPTED" in stdout

    run_dir = max((out / "runs").iterdir(), key=lambda p: p.name)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    statuses = {rec["status"] for rec in manifest["files"].values()}
    assert "interrupted" in statuses or manifest["summary"]["interrupted"] > 0
    completed_before = sum(
        1 for r in manifest["files"].values() if r["status"] in ("success", "warning")
    )
    assert 0 < completed_before < 2 * COPIES, "should be stopped mid-run"

    # Resume completes the rest.
    resume = subprocess.run(
        [*cmd, "--resume"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert resume.returncode == 0, resume.stderr
    md_files = list(out.rglob("*.md"))
    assert len(md_files) == 2 * COPIES
