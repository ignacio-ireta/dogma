"""Tutor test helpers: a synthetic problem repo with sentinel secrets, and report builders.

Sentinels mark content that must never reach a prompt: the reference solution, the hidden tests,
the learner profile under private/, and hidden-test values. Plain helpers; fixtures are in conftest.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from dogma.core.config import Settings
from dogma.corpus.models import Chunk, SearchHit
from dogma.db.models import AttemptMode
from dogma.db.store import record_attempt
from dogma.grading.models import GradeReport, RunStatus
from dogma.grading.models import TestResult as Result
from dogma.tutor.evidence import Evidence, History, LessonSection, redact_report
from dogma.tutor.hints import parse_reviewed_hints

PROBLEM_ID = "tst/alpha/p01-double"
SOLUTION_SENTINEL = "SOLUTION_SENTINEL_7f3a"
HIDDEN_SENTINEL = "HIDDEN_SENTINEL_91c2"
PROFILE_SENTINEL = "PROFILE_SENTINEL_5b8e"
HIDDEN_VALUE = "HIDDEN_VALUE_4402"
LEARNER_CODE = "def double_all(xs):\n    return [x + x for x in xs[0]]\n"

PROBLEM_TEXT = """# Double all

Implement `double_all(xs)`, which returns every element of `xs` doubled.

## Contract

- `xs`: a list of numbers. Return a new list and do not mutate `xs`.
"""

LESSON_TEXT = """---
title: Alpha lesson
minutes: 5
min_viable: "Read Doubling."
---

# Alpha

## Doubling

Doubling a number is multiplication by 2. Map the operation over each element of the list.

## Unrelated

Pandas indexes rows by label, not by position.
"""

HINTS_TEXT = """## Hint 1: concept

Doubling is multiplication by two, applied to every element, so the work is a map over the list.

## Hint 2: formula / approach

Each output value is `y = x + x`, one addition per element. Build a new list, leave `xs` as it is.

## Hint 3: code shape

Loop over `xs`, double each value, collect the results in a new list and return it. The new value
for one element is `x * 2`.
"""

META_TEXT = """id: tst/alpha/p01-double
version: 1
title: Double all
language: python
runtime: py-numpy
entry: double_all
difficulty: easy
minutes: 10
core: true
tolerance: {atol: 1.0e-9, rtol: 1.0e-9}
banned_imports: []
timeout_s: 30
status: reviewed
"""


def write_problem_repo(root: Path) -> Settings:
    """A temporary repo with one problem, a lesson, a workspace file and a private profile."""
    problem = root / "content" / "tst" / "alpha" / "problems" / "p01-double"
    problem.mkdir(parents=True)
    (root / "content" / "tst" / "alpha" / "lesson.md").write_text(LESSON_TEXT, encoding="utf-8")
    (problem / "meta.yaml").write_text(META_TEXT, encoding="utf-8")
    (problem / "problem.md").write_text(PROBLEM_TEXT, encoding="utf-8")
    (problem / "hints.md").write_text(HINTS_TEXT, encoding="utf-8")
    (problem / "starter.py").write_text("def double_all(xs):\n    ...\n", encoding="utf-8")
    (problem / "solution.py").write_text(
        f"def double_all(xs):\n    return [{SOLUTION_SENTINEL} * x for x in xs]\n",
        encoding="utf-8",
    )
    (problem / "test_public.py").write_text("# public tests\n", encoding="utf-8")
    (problem / "test_hidden.py").write_text(f"# {HIDDEN_SENTINEL}\n", encoding="utf-8")
    (root / "private").mkdir()
    (root / "private" / "learner.yaml").write_text(f"{PROFILE_SENTINEL}: true\n", encoding="utf-8")
    workspace = root / "workspace" / "tst" / "alpha"
    workspace.mkdir(parents=True)
    (workspace / "p01-double.py").write_text(LEARNER_CODE, encoding="utf-8")
    return Settings(repo_root=root)


def record_attempt_with_report(
    settings: Settings,
    *,
    attempt_id: str,
    status: RunStatus,
    mode: AttemptMode = AttemptMode.SUBMIT,
    tests: tuple[Result, ...] = (),
    problem_id: str = PROBLEM_ID,
    when: datetime | None = None,
) -> GradeReport:
    """Store an attempt the way the CLI does, plus the harness report file the runner leaves."""
    stamp = when if when is not None else datetime(2026, 10, 24, 12, 0, tzinfo=UTC)
    report = GradeReport(
        attempt_id=attempt_id,
        problem_id=problem_id,
        problem_version=1,
        runtime="py-numpy",
        status=status,
        tests=tests,
    )
    record_attempt(settings, report, mode=mode, core=True, started_at=stamp, submitted_at=stamp)
    if status in (RunStatus.PASSED, RunStatus.FAILED, RunStatus.ERROR):
        run_dir = settings.runs_dir / attempt_id
        run_dir.mkdir(parents=True, exist_ok=True)
        harness = {
            "status": "passed" if status is RunStatus.PASSED else status.value,
            "tests": [test.model_dump() for test in tests],
            "stdout": "",
        }
        (run_dir / "report.json").write_text(json.dumps(harness), encoding="utf-8")
    return report


def failing_tests() -> tuple[Result, ...]:
    """Two public and two hidden results; the hidden message carries a value that must not leak."""
    return (
        Result(name="test_small", file="public", outcome="pass"),
        Result(
            name="test_negative",
            file="public",
            outcome="fail",
            message="AssertionError: expected [-2] got [0]",
        ),
        Result(name="test_hidden_large", file="hidden", outcome="pass"),
        Result(
            name="test_hidden_mutation",
            file="hidden",
            outcome="fail",
            message=(
                f"AssertionError: value {HIDDEN_VALUE}\n"
                "Falsifying example: test_x(xs=[1, 2])\n"
                "submission.py:2: in double_all\n"
            ),
        ),
    )


def sample_evidence(*, failed_submits: int = 2) -> Evidence:
    """Fixed evidence for snapshot tests: no disk, no database. Mirrors one real problem."""
    report = GradeReport(
        attempt_id="20261024T120000Z-sample01",
        problem_id=PROBLEM_ID,
        problem_version=1,
        runtime="py-numpy",
        status=RunStatus.FAILED,
        tests=failing_tests(),
    )
    hit = SearchHit(
        chunk=Chunk(
            source_id="sources:notes/doubling.md",
            path="sources/notes/doubling.md",
            page=3,
            headings=("Notes", "Doubling"),
            ordinal=0,
            text="Doubling a value means multiplying it by two, element by element.",
        ),
        citation="[sources:notes/doubling.md p.3](sources/notes/doubling.md)",
        score=-1.25,
    )
    return Evidence(
        problem_id=PROBLEM_ID,
        title="Double all",
        topic_id="tst/alpha",
        language="python",
        problem_text=PROBLEM_TEXT,
        lesson_title="Alpha lesson",
        lesson_sections=(
            LessonSection(
                "Doubling",
                "Doubling a number is multiplication by 2. Map the operation over each element"
                " of the list.",
            ),
        ),
        workspace_name="workspace/tst/alpha/p01-double.py",
        workspace_text=LEARNER_CODE,
        history=History(
            known=True,
            failed_submits=failed_submits,
            latest=redact_report(report),
            latest_submitted_at="2026-10-24T12:00:00Z",
        ),
        corpus_hits=(hit,),
        reviewed=parse_reviewed_hints(HINTS_TEXT),
        notes=(),
    )
