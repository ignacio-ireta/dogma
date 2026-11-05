"""Tutor fixtures: a synthetic repo under tmp_path, never the real var/ or private/."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.unit.tutor.support import write_problem_repo

from dogma.core.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return write_problem_repo(tmp_path / "repo")
