from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from dogma.core.config import Settings
from dogma.runner.run import effective_timeout_s


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(repo_root=tmp_path, **overrides)  # type: ignore[arg-type]


def test_a_problem_limit_wins_over_the_default(tmp_path: Path) -> None:
    cfg = _settings(tmp_path, run_timeout_s=30)

    assert effective_timeout_s(60, cfg) == 60
    assert effective_timeout_s(None, cfg) == 30


def test_the_scale_lengthens_both_the_default_and_a_problem_limit(tmp_path: Path) -> None:
    cfg = _settings(tmp_path, run_timeout_s=30, run_timeout_scale=3.0)

    assert effective_timeout_s(None, cfg) == 90
    assert effective_timeout_s(60, cfg) == 180


def test_a_fractional_scale_rounds_the_limit_up(tmp_path: Path) -> None:
    cfg = _settings(tmp_path, run_timeout_scale=1.5)

    assert effective_timeout_s(7, cfg) == 11


def test_the_scale_cannot_shorten_a_limit(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        _settings(tmp_path, run_timeout_scale=0.5)


def test_the_scale_is_read_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOGMA_RUN_TIMEOUT_SCALE", "2")

    assert _settings(tmp_path).run_timeout_scale == 2.0
