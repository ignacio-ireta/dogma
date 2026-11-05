"""The full quality gate must audit newly generated production assets."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_check_rejects_a_build_that_introduces_private_text(tmp_path: Path) -> None:
    (tmp_path / "Makefile").write_text((REPO / "Makefile").read_text())
    (tmp_path / "web").mkdir()
    (tmp_path / "private").mkdir()
    (tmp_path / "private/denylist.txt").write_text("zqxcorp\n")
    commands = tmp_path / "bin"
    commands.mkdir()
    script = """#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

if Path(sys.argv[0]).name == "pnpm" and sys.argv[1:] == ["build"]:
    output = Path("dist/assets/app.js")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(os.environ["TASK_PRIVATE_VALUE"])
elif sys.argv[-2:] == ["python", "scripts/audit_privacy.py"]:
    raise SystemExit(subprocess.call([
        sys.executable, os.environ["TASK_PRIVACY_SCRIPT"], "--root", os.environ["TASK_GATE_ROOT"]
    ]))
"""
    for name in ("uv", "pnpm"):
        path = commands / name
        path.write_text(script)
        path.chmod(0o755)
    result = subprocess.run(
        ["make", "--no-print-directory", "check"],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{commands}{os.pathsep}{os.environ['PATH']}",
            "TASK_PRIVACY_SCRIPT": str(REPO / "scripts/audit_privacy.py"),
            "TASK_GATE_ROOT": str(tmp_path),
            "TASK_PRIVATE_VALUE": "ZQXCORP",
        },
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode != 0, "the privacy audit must run after production assets are built"
    assert "web/dist/assets/app.js" in result.stdout
    assert "private rule 1" in result.stdout
