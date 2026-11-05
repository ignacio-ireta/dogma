"""A synthetic repository for session tests: a four-topic path with known prerequisites.

It never reads the real ``content/`` or ``curriculum/``, so these tests do not change when authored
content does. Topics: ALPHA is ready with no prerequisites; BETA requires ALPHA; GAMMA is a draft;
DELTA is ready with no prerequisites. The path lists them in that order.
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
import yaml
from tests.unit.db.support import make_report

import dogma.cli.close as close_module
import dogma.cli.today as today_module
from dogma.core.config import Settings
from dogma.db.models import AttemptMode
from dogma.db.store import record_attempt
from dogma.grading.models import RunStatus

ALPHA = "tst/alpha"
BETA = "tst/beta"
GAMMA = "tst/gamma"
DELTA = "tst/delta"
PATH = (ALPHA, BETA, GAMMA, DELTA)
REPO = Path(__file__).resolve().parents[3]
NOON = datetime(2026, 10, 24, 12, tzinfo=UTC)


@pytest.fixture(autouse=True)
def utc_timezone() -> Iterator[None]:
    """Pin the process zone to UTC, so 'the local date' in these tests never depends on the host."""
    original = os.environ.get("TZ")
    os.environ["TZ"] = "UTC"
    time.tzset()
    yield
    if original is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = original
    time.tzset()


def core_problem(topic_id: str) -> str:
    return f"{topic_id}/p01-core"


def build_repo(root: Path) -> Settings:
    """Write a small curriculum and content tree under ``root`` and return settings for it."""
    paths = root / "curriculum" / "paths"
    paths.mkdir(parents=True)
    # The planner names the weakest mock part, so the scoring rubrics must exist in the repo.
    shutil.copytree(REPO / "curriculum" / "rubrics", root / "curriculum" / "rubrics")
    core = {
        "id": "test",
        "title": "Synthetic path",
        "start": "2026-10-24",
        "end": "2026-12-31",
        "sessions_per_day": 3,
        "session_minutes": 45,
        "days_per_week": 6,
        "weeks": [{"week": 1, "starts": "2026-10-24", "theme": "Synthetic", "topics": list(PATH)}],
    }
    (paths / "core.yaml").write_text(yaml.safe_dump(core, sort_keys=False), encoding="utf-8")
    write_topic(root, ALPHA)
    write_topic(root, BETA, requires=(ALPHA,))
    write_topic(root, GAMMA, status="draft")
    write_topic(root, DELTA)
    return Settings(repo_root=root)


def write_topic(
    root: Path,
    topic_id: str,
    *,
    status: str = "reviewed",
    requires: tuple[str, ...] = (),
    core_status: str = "reviewed",
) -> None:
    folder = root / "content" / topic_id
    problem = folder / "problems" / "p01-core"
    problem.mkdir(parents=True)
    slug = topic_id.split("/")[1]
    topic = {
        "id": topic_id,
        "title": slug.replace("-", " ").capitalize(),
        "tier": "A",
        "requires": list(requires),
        "aliases": [slug],
        "objectives": [f"Explain {slug} in one sentence", f"Compute {slug} on an example"],
        "minutes": {"lesson": 12, "practice": 20, "min_viable": 15},
        "interview_angle": "A warm-up question.",
        "sources": ["book: Synthetic reader, chapter 1"],
        "status": status,
    }
    cards = {
        "cards": [
            {"id": f"{topic_id}#1", "type": "qa", "front": f"What is {slug}?", "back": "A test."},
            {"id": f"{topic_id}#2", "type": "qa", "front": f"Why {slug}?", "back": "For tests."},
        ]
    }
    meta = {
        "id": core_problem(topic_id),
        "version": 1,
        "title": f"{slug} core",
        "language": "python",
        "runtime": "py-numpy",
        "entry": "solve",
        "difficulty": "easy",
        "minutes": 20,
        "core": True,
        "tolerance": {"atol": 1e-9, "rtol": 1e-9},
        "banned_imports": [],
        "timeout_s": 30,
        "status": core_status,
    }
    (folder / "topic.yaml").write_text(yaml.safe_dump(topic, sort_keys=False), encoding="utf-8")
    (folder / "cards.yaml").write_text(yaml.safe_dump(cards, sort_keys=False), encoding="utf-8")
    (problem / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")


def solve(settings: Settings, topic_id: str, *, when: datetime) -> None:
    """Record a passing core submit: the only thing that marks a topic solved."""
    report = make_report(
        RunStatus.PASSED,
        public=(3, 3),
        hidden=(2, 2),
        problem_id=core_problem(topic_id),
        attempt_id=uuid4().hex,
    )
    record_attempt(
        settings,
        report,
        mode=AttemptMode.SUBMIT,
        core=True,
        started_at=when - timedelta(seconds=1),
        submitted_at=when,
    )


def fix_clock(monkeypatch: pytest.MonkeyPatch, instant: datetime) -> None:
    """Make the session commands read ``instant`` as the current time (they call datetime.now)."""

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> datetime:  # type: ignore[override]
            return instant.astimezone(tz) if tz is not None else instant.replace(tzinfo=None)  # type: ignore[arg-type]

    monkeypatch.setattr(today_module, "datetime", FixedDatetime)
    monkeypatch.setattr(close_module, "datetime", FixedDatetime)
