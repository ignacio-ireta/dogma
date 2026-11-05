"""Settings rooted in a temporary repo. Nothing under tests/unit/db touches the real var/."""

from __future__ import annotations

from pathlib import Path

import pytest

from dogma.core.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(repo_root=tmp_path / "repo")
