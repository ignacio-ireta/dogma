"""Atomic write semantics: parents created, replace-in-place, temp colocated."""

from pathlib import Path

import pytest

from docmd.fsio import atomic_write_text


def test_creates_parent_directories(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "out.md"
    atomic_write_text(target, "hello")
    assert target.read_text() == "hello"


def test_replaces_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "out.md"
    target.write_text("old")
    atomic_write_text(target, "new")
    assert target.read_text() == "new"


def test_no_temp_files_left_behind(tmp_path: Path) -> None:
    target = tmp_path / "out.md"
    atomic_write_text(target, "content")
    assert [p.name for p in tmp_path.iterdir()] == ["out.md"]


def test_failure_cleans_up_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "out.md"

    def boom(self: Path, other: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", boom)
    with pytest.raises(OSError, match="replace failed"):
        atomic_write_text(target, "content")
    assert list(tmp_path.iterdir()) == []
