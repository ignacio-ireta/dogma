"""Review tests only use a scratch copy of the example topic and learner DB."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from dogma.core.config import Settings
from dogma.db.errors import StorageErrorCode
from dogma.db.models import Mastery, TopicState
from dogma.db.session import open_session

NOW = datetime(2026, 10, 24, 12, tzinfo=UTC)
TOPIC = "cml/vector-operations"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    root = tmp_path / "repo"
    source = Path(__file__).resolve().parents[3] / "content" / TOPIC
    shutil.copytree(source, root / "content" / TOPIC)
    return Settings(repo_root=root)


@pytest.fixture
def solved(settings: Settings) -> Settings:
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        session.add(TopicState(topic_id=TOPIC, mastery=Mastery.SOLVED, solved_at=NOW))
    return settings


@pytest.fixture
def old_solve(settings: Settings) -> Settings:
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        session.add(
            TopicState(topic_id=TOPIC, mastery=Mastery.SOLVED, solved_at=NOW - timedelta(days=3))
        )
    return settings
