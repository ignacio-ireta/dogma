"""Evidence for one tutor call: what the learner is working on, and nothing else (ENGINEERING §10).

Reads content/ (the problem, its reviewed hints.md and its lesson), workspace/ (the learner's own
file), the learner database (recorded attempts) and the corpus index. It never reads private/,
solution files or hidden tests. Hidden-test detail is reduced to error types and learner-file line
numbers here, so no later stage can leak it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select

from dogma.content.markdown import parse_outline, split_front_matter
from dogma.content.schemas import ProblemMeta
from dogma.core.config import Settings
from dogma.core.errors import DogmaError
from dogma.corpus.index import search as corpus_search
from dogma.corpus.models import SearchHit
from dogma.db.errors import StorageError, StorageErrorCode
from dogma.db.models import Attempt
from dogma.db.session import open_session
from dogma.grading.models import GradeReport, RunStatus, TestResult
from dogma.tutor.hints import load_reviewed_hints
from dogma.tutor.models import ReviewedHints

FAILED_STATUSES = frozenset({RunStatus.FAILED, RunStatus.ERROR, RunStatus.TIMEOUT, RunStatus.OOM})
_FAILED_VALUES = frozenset(status.value for status in FAILED_STATUSES)
UNLOCK_FAILED_SUBMITS = 2
LESSON_SECTIONS = 3
# Candidate corpus passages, best first. context.py keeps at most one per document.
CORPUS_CANDIDATES = 12
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_ERROR_TYPE = re.compile(r"\b([A-Za-z_]\w*(?:Error|Exception|Warning))\b")
_LEARNER_LINE = re.compile(r'submission\.(?:py|sql)(?::|",\s*line\s*)(\d+)')
_WORD = re.compile(r"[a-z][a-z0-9]{2,}")
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "each",
        "are",
        "its",
        "not",
        "you",
        "your",
        "must",
        "use",
        "using",
        "given",
        "one",
        "two",
        "any",
        "all",
        "has",
        "have",
    }
)
_HIDDEN_WITHHELD = "details withheld (hidden test)"


@dataclass(frozen=True)
class LessonSection:
    heading: str
    text: str


@dataclass(frozen=True)
class History:
    """Recorded attempts for one problem. ``known`` is False when the learner database failed."""

    known: bool
    failed_submits: int
    latest: GradeReport | None
    latest_submitted_at: str | None


@dataclass(frozen=True)
class Evidence:
    problem_id: str
    title: str
    topic_id: str
    language: str
    problem_text: str
    lesson_title: str | None
    lesson_sections: tuple[LessonSection, ...]
    workspace_name: str
    workspace_text: str | None
    history: History
    corpus_hits: tuple[SearchHit, ...]
    reviewed: ReviewedHints
    notes: tuple[str, ...]


@dataclass(frozen=True)
class _AttemptRow:
    id: str
    mode: str
    status: str
    problem_version: int
    duration_ms: int
    submitted_at_iso: str


class _HarnessFile(BaseModel):
    """The harness's report.json. Validated because the learner's code shares its directory."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    tests: tuple[TestResult, ...] = ()


def redact_hidden_message(message: str) -> str:
    """Keep the error type and learner-file line numbers of a hidden-test message, nothing else."""
    types = list(dict.fromkeys(_ERROR_TYPE.findall(message)))
    lines = list(dict.fromkeys(_LEARNER_LINE.findall(message)))
    parts: list[str] = []
    if types:
        parts.append("error type: " + ", ".join(types))
    if lines:
        parts.append("learner file line(s): " + ", ".join(lines))
    return "; ".join(parts) if parts else _HIDDEN_WITHHELD


def redact_report(report: GradeReport) -> GradeReport:
    """A copy safe to show a model: hidden messages reduced, collection errors reduced."""
    tests = tuple(_redact_test(test) for test in report.tests)
    message = report.message
    if report.status is RunStatus.ERROR and message:
        message = redact_hidden_message(message)
    return report.model_copy(update={"tests": tests, "message": message})


def gather_evidence(
    settings: Settings, *, problem_dir: Path, problem_id: str, meta: ProblemMeta
) -> Evidence:
    """Collect everything a tutor call may see for one problem. Missing parts become notes."""
    notes: list[str] = []
    track, topic, _ = problem_id.split("/")
    topic_id = f"{track}/{topic}"
    problem_text = _read_text(problem_dir / "problem.md") or ""
    if not problem_text:
        notes.append("problem.md is missing")
    lesson_title, sections = _lesson(problem_dir.parent.parent / "lesson.md", topic_id, notes)
    workspace_name, workspace_text = _workspace(settings, problem_id, meta, notes)
    history = _history(settings, problem_id, meta, notes)
    hits = _corpus(settings, topic_id, meta.title, notes)
    return Evidence(
        problem_id=problem_id,
        title=meta.title,
        topic_id=topic_id,
        language=meta.language,
        problem_text=problem_text,
        lesson_title=lesson_title,
        lesson_sections=_select_sections(sections, f"{meta.title} {problem_text}"),
        workspace_name=workspace_name,
        workspace_text=workspace_text,
        history=history,
        corpus_hits=hits,
        reviewed=load_reviewed_hints(problem_dir),
        notes=tuple(notes),
    )


def _redact_test(test: TestResult) -> TestResult:
    if test.file != "hidden" or not test.message:
        return test
    return test.model_copy(update={"message": redact_hidden_message(test.message)})


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _lesson(
    path: Path, topic_id: str, notes: list[str]
) -> tuple[str | None, tuple[LessonSection, ...]]:
    text = _read_text(path)
    if text is None:
        notes.append(f"no lesson found for {topic_id}")
        return None, ()
    try:
        front, body = split_front_matter(text)
    except ValueError:
        notes.append(f"lesson front matter for {topic_id} is malformed")
        return None, ()
    title = _title(front)
    outline = parse_outline(body)
    return title, tuple(LessonSection(s.heading, s.text.strip()) for s in outline.sections)


def _title(front: str | None) -> str | None:
    if front is None:
        return None
    try:
        data = yaml.safe_load(front)
    except yaml.YAMLError:
        return None
    title = data.get("title") if isinstance(data, dict) else None
    return title if isinstance(title, str) else None


def _select_sections(sections: tuple[LessonSection, ...], text: str) -> tuple[LessonSection, ...]:
    """The lesson sections that share the most words with the problem; document order is kept."""
    if not sections:
        return ()
    wanted = _terms(text)
    scored = [
        (len(wanted & _terms(f"{section.heading} {section.text}")), index, section)
        for index, section in enumerate(sections)
    ]
    ranked = sorted((item for item in scored if item[0] > 0), key=lambda item: (-item[0], item[1]))
    chosen = ranked[:LESSON_SECTIONS] or scored[:1]
    return tuple(section for _, _, section in sorted(chosen, key=lambda item: item[1]))


def _terms(text: str) -> frozenset[str]:
    return frozenset(word for word in _WORD.findall(text.lower()) if word not in _STOPWORDS)


def _workspace(
    settings: Settings, problem_id: str, meta: ProblemMeta, notes: list[str]
) -> tuple[str, str | None]:
    track, topic, slug = problem_id.split("/")
    suffix = ".sql" if meta.language == "sql" else ".py"
    path = settings.workspace_dir / track / topic / f"{slug}{suffix}"
    text = _read_text(path)
    if text is None:
        notes.append("no workspace file yet: run dogma open first")
    return path.relative_to(settings.repo_root).as_posix(), text


def _history(settings: Settings, problem_id: str, meta: ProblemMeta, notes: list[str]) -> History:
    try:
        rows = _attempt_rows(settings, problem_id)
    except StorageError as exc:
        notes.append(
            f"learner history unavailable ({exc.code}); the walkthrough needs confirmation"
        )
        return History(known=False, failed_submits=0, latest=None, latest_submitted_at=None)
    failed = sum(1 for row in rows if row.mode == "submit" and row.status in _FAILED_VALUES)
    if not rows:
        return History(known=True, failed_submits=failed, latest=None, latest_submitted_at=None)
    latest = rows[0]
    return History(
        known=True,
        failed_submits=failed,
        latest=_report(settings, latest, meta),
        latest_submitted_at=latest.submitted_at_iso,
    )


def _attempt_rows(settings: Settings, problem_id: str) -> tuple[_AttemptRow, ...]:
    """Recorded attempts for the problem, newest first. A read failure is a StorageError."""
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        found = session.scalars(
            select(Attempt)
            .where(Attempt.problem_id == problem_id)
            .order_by(Attempt.submitted_at.desc(), Attempt.id.desc())
        ).all()
        return tuple(
            _AttemptRow(
                id=row.id,
                mode=row.mode,
                status=row.status,
                problem_version=row.problem_version,
                duration_ms=row.duration_ms,
                submitted_at_iso=row.submitted_at.isoformat(timespec="seconds"),
            )
            for row in found
        )


def _report(settings: Settings, row: _AttemptRow, meta: ProblemMeta) -> GradeReport:
    """Rebuild the recorded GradeReport from its row and the run's report.json, then redact it."""
    try:
        status = RunStatus(row.status)
    except ValueError:
        status = RunStatus.ERROR
    report = GradeReport(
        attempt_id=row.id,
        problem_id=meta.id,
        problem_version=row.problem_version,
        runtime=meta.runtime,
        status=status,
        tests=_harness_tests(settings, row.id),
        duration_ms=row.duration_ms,
    )
    return redact_report(report)


def _harness_tests(settings: Settings, attempt_id: str) -> tuple[TestResult, ...]:
    if not _RUN_ID.fullmatch(attempt_id):
        return ()
    text = _read_text(settings.runs_dir / attempt_id / "report.json")
    if text is None:
        return ()
    try:
        return _HarnessFile.model_validate_json(text).tests
    except ValueError:
        return ()


def _corpus(
    settings: Settings, topic_id: str, title: str, notes: list[str]
) -> tuple[SearchHit, ...]:
    """Corpus candidates for the problem's topic, best first. The topic's own lesson is excluded.

    The candidates are not yet limited per document: context assembly does that (I-22).
    """
    terms = list(
        dict.fromkeys(word for word in _WORD.findall(title.lower()) if word not in _STOPWORDS)
    )
    if not terms:
        return ()
    try:
        hits = corpus_search(
            settings, " ".join(terms[:4]), topics=(topic_id,), k=CORPUS_CANDIDATES + 2
        )
    except DogmaError as exc:
        notes.append(f"corpus unavailable ({exc.code}); run dogma corpus sync for other sources")
        return ()
    lesson_id = f"content:{topic_id}/lesson"
    return tuple(hit for hit in hits if hit.chunk.source_id != lesson_id)[:CORPUS_CANDIDATES]
