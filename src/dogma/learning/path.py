"""The core path's calendar and each topic's status on disk (ROADMAP I-05, I-14).

Shared by the CLI (``dogma path``, ``dogma progress``) and the API (``/today``, ``/progress``), so
both answer from the same rules. A topic is ``missing`` when no package exists yet, ``invalid`` when
its package does not load, and otherwise shows the status in its topic.yaml. Only verified and
reviewed topics are ready to study (CONTENT_SPEC §2, ENGINEERING_STANDARDS §5.7).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from dogma.content.loader import load_topic
from dogma.content.schemas import PathWeek
from dogma.core.config import Settings
from dogma.core.errors import ContentError

READY = frozenset({"verified", "reviewed"})


@dataclass(frozen=True)
class TopicRow:
    topic_id: str
    title: str
    status: str
    problems: tuple[str, ...]


def current_week(weeks: tuple[PathWeek, ...], today: date) -> PathWeek:
    """The last week that has started by ``today``; the first week before the path starts."""
    current = weeks[0]
    for item in weeks:
        if item.starts <= today:
            current = item
    return current


def topic_row(settings: Settings, topic_id: str) -> TopicRow:
    """One topic's status and problems, read from its package on disk."""
    topic_dir = settings.content_dir / topic_id
    if not (topic_dir / "topic.yaml").is_file():
        return TopicRow(topic_id, "", "missing", ())
    try:
        package = load_topic(topic_dir)
    except ContentError:
        return TopicRow(topic_id, "", "invalid", ())
    return TopicRow(
        topic_id=topic_id,
        title=package.meta.title,
        status=package.meta.status,
        problems=tuple(problem.dir.name for problem in package.problems),
    )
