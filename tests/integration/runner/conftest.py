"""Shared fixtures for the subprocess-based runner tests.

Every run writes into a temporary repo root, never into the real ``var/``.
"""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from dogma.core.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "problems"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(repo_root=tmp_path / "repo")


@pytest.fixture
def submit(tmp_path: Path) -> Callable[..., Path]:
    """Write a learner file somewhere unique and return its path."""

    def _write(source: str, name: str = "submission.py") -> Path:
        path = tmp_path / "work" / uuid.uuid4().hex / name
        path.parent.mkdir(parents=True)
        path.write_text(source, encoding="utf-8")
        return path

    return _write


@pytest.fixture
def problem_copy(tmp_path: Path) -> Callable[..., Path]:
    """Copy a fixture problem and override keys in its meta.yaml (e.g. a short timeout)."""

    def _copy(base: Path, **overrides: Any) -> Path:
        dest = tmp_path / "problems" / uuid.uuid4().hex / base.name
        shutil.copytree(base, dest)
        meta = yaml.safe_load((dest / "meta.yaml").read_text(encoding="utf-8"))
        meta.update(overrides)
        (dest / "meta.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")
        return dest

    return _copy
