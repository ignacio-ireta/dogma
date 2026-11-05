from __future__ import annotations

from pathlib import Path

from dogma.core.errors import DogmaError, ErrorCode
from dogma.core.fsio import atomic_write


def test_atomic_write_creates_parents_and_leaves_no_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "file.txt"
    atomic_write(target, "hello")
    atomic_write(target, "world")
    assert target.read_text() == "world"
    assert sorted(p.name for p in target.parent.iterdir()) == ["file.txt"]


def test_errors_carry_stable_code() -> None:
    err = DogmaError(ErrorCode.RUNNER_TIMEOUT, "took too long")
    assert err.code == "RUNNER_TIMEOUT"
    assert "took too long" in str(err)
