"""The grade seam: the one call the CLI and the API make to grade an attempt."""

from __future__ import annotations

from pathlib import Path

import pytest

from dogma.core.config import Settings
from dogma.runner import attempt


def test_grade_attempt_hands_its_arguments_to_the_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[Path, Path, bool, Settings]] = []
    sentinel = object()

    def fake_run(
        problem_dir: Path, submission: Path, *, include_hidden: bool, settings: Settings
    ) -> object:
        calls.append((problem_dir, submission, include_hidden, settings))
        return sentinel

    monkeypatch.setattr(attempt, "run", fake_run)
    settings = Settings(repo_root=tmp_path)

    result = attempt.grade_attempt(
        tmp_path / "p", tmp_path / "s.py", include_hidden=True, settings=settings
    )

    assert result is sentinel
    assert calls == [(tmp_path / "p", tmp_path / "s.py", True, settings)]
