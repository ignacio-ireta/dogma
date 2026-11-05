"""Plain helpers for the API tests: report builders, a frozen clock, a scratch repo copy."""

from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dogma.core.config import Settings
from dogma.grading.models import GradeReport, RunStatus
from dogma.grading.models import TestResult as Result
from dogma.tutor.models import CompletionChunk, CompletionRequest, ProviderError
from dogma.tutor.providers.fake import FakeProvider

REPO_ROOT = Path(__file__).resolve().parents[3]
TOPIC = "cml/vector-operations"
PROBLEM = "cml/vector-operations/p01-vector-projection"
NOW = datetime(2026, 10, 26, 9, 0, tzinfo=UTC)
ORIGIN = "http://127.0.0.1:5173"


@dataclass
class FrozenClock:
    """Stands in for ``dogma.api.deps.get_clock``: every call returns ``now``."""

    now: datetime = NOW

    def __call__(self) -> datetime:
        return self.now


def report(
    status: RunStatus = RunStatus.PASSED,
    *,
    include_hidden: bool = False,
    attempt_id: str | None = None,
    message: str = "",
) -> GradeReport:
    """A GradeReport for ``PROBLEM``; a passing submit has public and hidden results."""
    tests: list[Result] = [Result(name="test_pub_0", file="public", outcome="pass")]
    if include_hidden:
        tests.append(Result(name="test_hid_0", file="hidden", outcome="pass"))
    if status is not RunStatus.PASSED:
        tests = [Result(name="test_pub_0", file="public", outcome="fail", message="wrong")]
    return GradeReport(
        attempt_id=attempt_id or f"t{uuid.uuid4().hex[:12]}",
        problem_id=PROBLEM,
        problem_version=2,
        runtime="py-numpy",
        status=status,
        tests=tuple(tests),
        message=message,
    )


def copy_repo(tmp: Path) -> Path:
    """A scratch repo with the real curriculum, templates and the example topic only."""
    root = tmp / "repo"
    root.mkdir()
    shutil.copy(REPO_ROOT / "pyproject.toml", root / "pyproject.toml")
    shutil.copytree(REPO_ROOT / "curriculum", root / "curriculum")
    shutil.copytree(REPO_ROOT / "templates", root / "templates")
    shutil.copytree(REPO_ROOT / "content" / TOPIC, root / "content" / TOPIC)
    (root / "workspace").mkdir()
    return root


class FakeGrader:
    """Replaces ``runner.attempt.grade_attempt``: a passing report unless told otherwise."""

    def __init__(self) -> None:
        self.calls: list[tuple[Path, bool]] = []
        self.status = None  # set to a RunStatus to script a non-passing verdict

    def __call__(
        self,
        problem_dir: Path,
        submission: Path,
        *,
        include_hidden: bool,
        settings: Settings,
    ) -> GradeReport:
        self.calls.append((submission, include_hidden))
        if self.status is None:
            return report(include_hidden=include_hidden)
        return report(self.status, include_hidden=include_hidden)


class ChunkedFakeProvider(FakeProvider):
    """Streams each scripted reply in pieces of ``size`` characters, so a test sees several deltas.

    A ``ProviderError`` in the script is raised mid-stream, as a real model failure would be.
    """

    def __init__(self, replies: Iterable[str | ProviderError] = (), *, size: int = 9) -> None:
        super().__init__(replies)
        self._size = size

    def stream(self, req: CompletionRequest) -> Iterator[CompletionChunk]:
        text = self._next(req)
        for start in range(0, len(text), self._size):
            yield CompletionChunk(text=text[start : start + self._size])


def parse_events(body: str) -> list[tuple[str, dict[str, Any]]]:
    """The ``(event, data)`` pairs of an SSE body. Each block holds one event and one data line."""
    events: list[tuple[str, dict[str, Any]]] = []
    for block in body.strip("\n").split("\n\n"):
        lines = block.split("\n")
        assert len(lines) == 2, block
        assert lines[0].startswith("event: "), block
        assert lines[1].startswith("data: "), block
        events.append((lines[0][len("event: ") :], json.loads(lines[1][len("data: ") :])))
    return events
