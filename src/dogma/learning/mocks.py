"""Mock interviews: start, reveal, score, debrief and abandon (ARCHITECTURE §4.4, ADR-0007).

A mock is a timed run through three things: a live coding problem the learner has passed, a
theory question from a topic they have retained, and a system-design case from the ``sys`` track.
The learner scores the 30 items of the ``mock`` rubric themselves. The score is the source of
truth, and it is stored as one ``answer`` row linked to the mock.

Choosing is deterministic. A case covering the weakest part over the last three scored mocks comes
first, then the least recently used one, then the smallest id. The same rule orders the coding
problems and theory topics by recency. At most one mock is open. Recording off refuses every write
before anything is touched, so nothing is stored.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal
from uuid import uuid4

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from dogma.content.interview import InterviewQuestion, parse_questions
from dogma.content.loader import discover_topic_dirs, load_problem, load_topic
from dogma.content.rubrics import Band, Rubric, check_scores, load_rubric_version
from dogma.content.schemas import Reveal
from dogma.core.config import Settings
from dogma.core.errors import ContentError, DogmaError, ErrorCode
from dogma.db.answer_models import Answer, AnswerKind
from dogma.db.errors import StorageErrorCode
from dogma.db.mock_models import MockSession, MockStatus
from dogma.db.models import Mastery
from dogma.db.progress import Snapshot, first_passes, read_snapshot
from dogma.db.session import database_path, open_session
from dogma.learning.mock_rules import (
    PartScore,
    band_for,
    part_scores,
    reveal_for,
    shares_keyword,
    strongest_part,
    total_of,
    weakest_over,
    weakest_part,
)
from dogma.learning.planner import READY_STATUSES
from dogma.tutor.gateway import Gateway
from dogma.tutor.interviewer import clean_question, pick_reveal

MOCK_RUBRIC_ID: Final = "mock"
MOCK_RUBRIC_VERSION: Final = 1
RECENT_MOCKS: Final = 3  # the weakest-part rule looks at this many scored mocks
ITEM_NOTE_LIMIT: Final = 200
NOTHING_ON_THAT: Final = "The interviewer has nothing on that."
_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
_RETAINED: Final = frozenset({Mastery.RETAINED, Mastery.INTERVIEW_READY})


@dataclass(frozen=True)
class MockPlan:
    """What a started mock will work through. Any part is None when nothing qualified yet."""

    session_id: str
    started_at: datetime
    case_id: str | None
    case_title: str | None
    coding_problem_id: str | None
    coding_title: str | None
    theory_topic_id: str | None
    theory_title: str | None
    theory_question: int | None
    theory_question_title: str | None
    part_minutes: tuple[tuple[str, str], ...]  # (part title, minutes), in rubric order
    focus: PartScore | None  # the weakest part over the last scored mocks, if there are any


@dataclass(frozen=True)
class RevealResult:
    text: str
    matched: bool
    exhibit: str | None  # repo-relative path of the exhibit that comes with the reveal
    via: Literal["model", "keyword"] = "keyword"  # which path chose it (I-18d interviewer)


@dataclass(frozen=True)
class CaseMaterial:
    """What the interviewer may use for the open mock: the brief and reveals, never debrief.md."""

    case_id: str | None
    brief: str | None
    reveals: tuple[Reveal, ...]


@dataclass(frozen=True)
class MockReport:
    """A scored mock: part totals, the band, and the highest and lowest part by percentage."""

    session_id: str
    parts: tuple[PartScore, ...]
    total: int
    maximum: int
    band: Band
    strongest: PartScore
    weakest: PartScore


@dataclass(frozen=True)
class MockDebrief:
    """The last scored mock, with the case's debrief text (hidden until the mock is scored)."""

    report: MockReport
    case_id: str | None
    case_debrief: str | None


@dataclass(frozen=True)
class MockStatusView:
    open: MockPlan | None
    last_scored: MockReport | None


@dataclass(frozen=True)
class _Case:
    case_id: str
    title: str
    covers: frozenset[str]  # the H2 headings of debrief.md, lower-cased


def start_mock(settings: Settings, *, now: datetime, case_id: str | None = None) -> MockPlan:
    """Open a mock and choose its case, coding problem and theory question. Nothing else changes."""
    _require_recording(settings)
    when = _aware(now)
    rubric = _rubric(settings)
    snapshot = read_snapshot(settings, now=when)
    cases = _cases(settings)
    chosen_case = _chosen_case(cases, case_id)
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as db:
        if _open_row(db) is not None:
            raise DogmaError(
                ErrorCode.MOCK_ALREADY_OPEN,
                "A mock is already open. Finish it with dogma mock score or dogma mock "
                "abandon first.",
            )
        history = list(db.scalars(select(MockSession).order_by(MockSession.started_at)))
        weak = weakest_over(rubric, _recent_scores(db, rubric))
        case = _pick_case(cases, history, weak, chosen_case)
        coding = _pick_coding(settings, snapshot, history)
        theory = _pick_theory(settings, snapshot, history)
        row = MockSession(
            id=uuid4().hex,
            started_at=when,
            case_id=case.case_id if case else None,
            coding_problem_id=coding,
            theory_topic_id=theory[0] if theory else None,
            theory_question=theory[1] if theory else None,
            status=MockStatus.OPEN.value,
            part_minutes={part.id: part.minutes for part in rubric.parts},
        )
        db.add(row)
        db.flush()
        return _plan(settings, row, rubric, weak)


def reveal(settings: Settings, question: str) -> RevealResult:
    """What the interviewer releases for ``question``. Reads only: nothing is stored."""
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as db:
        row = _open_row(db)
        if row is None:
            raise DogmaError(ErrorCode.MOCK_NOT_OPEN, _no_open_message())
        case_id = row.case_id
    if case_id is None:
        return RevealResult(text=NOTHING_ON_THAT, matched=False, exhibit=None)
    reveals = _case_reveals(settings, case_id)
    hit = reveal_for(question, reveals)
    if hit is None:
        return RevealResult(text=NOTHING_ON_THAT, matched=False, exhibit=None)
    exhibit = f"content/{case_id}/{hit.exhibit}" if hit.exhibit else None
    return RevealResult(text=hit.answer, matched=True, exhibit=exhibit)


def open_case_material(settings: Settings) -> CaseMaterial:
    """The open mock's case brief and reveals. Reads only; debrief.md is never loaded here."""
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as db:
        row = _open_row(db)
        if row is None:
            raise DogmaError(ErrorCode.MOCK_NOT_OPEN, _no_open_message())
        case_id = row.case_id
    if case_id is None:
        return CaseMaterial(case_id=None, brief=None, reveals=())
    try:
        package = load_topic(settings.content_dir / case_id)
    except ContentError:
        return CaseMaterial(case_id=case_id, brief=None, reveals=())
    case = package.case
    brief = case.brief if case is not None else None
    reveals = case.reveals if case is not None and case.reveals is not None else ()
    return CaseMaterial(case_id=case_id, brief=brief, reveals=reveals)


def keyword_reveal(material: CaseMaterial, question: str) -> RevealResult:
    """The deterministic keyword match (I-18c). Used when the model cannot answer."""
    if material.case_id is None:
        return RevealResult(text=NOTHING_ON_THAT, matched=False, exhibit=None)
    hit = reveal_for(question, material.reveals)
    if hit is None:
        return RevealResult(text=NOTHING_ON_THAT, matched=False, exhibit=None)
    return _revealed(material.case_id, hit, via="keyword")


def ask_interviewer(settings: Settings, gateway: Gateway, question: str) -> RevealResult:
    """Let the model pick one reveal for the question, then keep it only if the question earns it.

    The model chooses a number. The text printed is the reveal's own answer from reveals.yaml.
    A pick whose question shares no keyword with the learner's question is treated as nothing.
    Raises the tutor's errors when the model is down or its reply is unusable: the caller then
    falls back to keyword_reveal.
    """
    clean = clean_question(question)
    material = open_case_material(settings)
    if material.case_id is None or not material.reveals or material.brief is None:
        return keyword_reveal(material, clean)
    picked = pick_reveal(
        gateway,
        brief=material.brief,
        when_asked=[reveal.when_asked for reveal in material.reveals],
        question=clean,
    )
    nothing = RevealResult(text=NOTHING_ON_THAT, matched=False, exhibit=None, via="model")
    if picked.number is None:
        return nothing
    chosen = material.reveals[picked.number - 1]
    if not shares_keyword(clean, chosen.when_asked):
        return nothing
    return _revealed(material.case_id, chosen, via="model")


def _revealed(case_id: str, hit: Reveal, *, via: Literal["model", "keyword"]) -> RevealResult:
    exhibit = f"content/{case_id}/{hit.exhibit}" if hit.exhibit else None
    return RevealResult(text=hit.answer, matched=True, exhibit=exhibit, via=via)


def score_mock(
    settings: Settings,
    scores: Mapping[str, object],
    *,
    now: datetime,
    notes: Mapping[str, str] | None = None,
) -> MockReport:
    """Store the 30 item scores of the open mock as one ``mock`` answer, and close it as scored.

    Every item needs a whole number from 1 to 4 and nothing else is allowed. A bad score stores
    nothing and leaves the mock open.
    """
    _require_recording(settings)
    when = _aware(now)
    rubric = _rubric(settings)
    clean_notes = _clean_notes(rubric, notes)
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as db:
        row = _open_row(db)
        if row is None:
            raise DogmaError(ErrorCode.MOCK_NOT_OPEN, _no_open_message())
        problems = check_scores(rubric, scores)
        if problems:
            raise DogmaError(ErrorCode.MOCK_SCORE_INVALID, "; ".join(problems) + ".")
        stored = {key: _whole_score(scores[key]) for key in rubric.criteria_ids}
        db.add(
            Answer(
                id=uuid4().hex,
                topic_id=None,
                kind=AnswerKind.MOCK.value,
                prompt_ref=row.case_id or "mock",
                text=clean_notes,
                rubric_id=rubric.id,
                rubric_version=rubric.version,
                scores=stored,
                notes=None,
                created_at=when,
                mock_session_id=row.id,
            )
        )
        row.status = MockStatus.SCORED.value
        row.ended_at = when
        session_id = row.id
    return _report(session_id, rubric, stored)


def debrief(settings: Settings, session_id: str | None = None) -> MockDebrief:
    """A scored mock's debrief: part totals, band, strength, weakness, and the case's debrief.

    Without ``session_id`` this is the last scored mock (the CLI's ``dogma mock debrief``). With
    one, it is that mock, which must be scored; any other id is not found.
    """
    rubric = _rubric(settings)
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as db:
        query = select(MockSession).where(MockSession.status == MockStatus.SCORED.value)
        if session_id is not None:
            query = query.where(MockSession.id == session_id)
        row = db.scalars(
            query.order_by(MockSession.ended_at.desc(), MockSession.id.desc()).limit(1)
        ).first()
        if row is None:
            raise DogmaError(
                ErrorCode.MOCK_NOT_FOUND,
                "No scored mock has that id."
                if session_id is not None
                else "No mock has been scored yet. Start one with dogma mock start.",
            )
        answer = db.scalars(select(Answer).where(Answer.mock_session_id == row.id)).one()
        stored = dict(answer.scores)
        session_id, case_id = row.id, row.case_id
    return MockDebrief(
        report=_report(session_id, rubric, stored),
        case_id=case_id,
        case_debrief=_case_debrief_text(settings, case_id),
    )


def abandon_mock(settings: Settings, *, now: datetime) -> str:
    """Close the open mock without scores. Returns its id. The next mock may start at once."""
    _require_recording(settings)
    when = _aware(now)
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as db:
        row = _open_row(db)
        if row is None:
            raise DogmaError(ErrorCode.MOCK_NOT_OPEN, _no_open_message())
        row.status = MockStatus.ABANDONED.value
        row.ended_at = when
        return row.id


def mock_status(settings: Settings) -> MockStatusView:
    """The open mock, if any, and the last scored one. Reads only; a missing database is empty."""
    if not database_path(settings).is_file():
        return MockStatusView(open=None, last_scored=None)
    rubric = _rubric(settings)
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as db:
        row = _open_row(db)
        open_plan = None if row is None else _plan(settings, row, rubric, _focus(db, rubric))
        last = db.scalars(
            select(MockSession)
            .where(MockSession.status == MockStatus.SCORED.value)
            .order_by(MockSession.ended_at.desc(), MockSession.id.desc())
            .limit(1)
        ).first()
        last_report = None
        if last is not None:
            answer = db.scalars(select(Answer).where(Answer.mock_session_id == last.id)).one()
            last_report = _report(last.id, rubric, dict(answer.scores))
    return MockStatusView(open=open_plan, last_scored=last_report)


def recent_weak_part(settings: Settings) -> PartScore | None:
    """The weakest part over the last three scored mocks. None until one mock is scored."""
    if not database_path(settings).is_file():
        return None
    rubric = _rubric(settings)
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as db:
        return _focus(db, rubric)


# --- choosing ---------------------------------------------------------------------------------


def _chosen_case(cases: Sequence[_Case], case_id: str | None) -> _Case | None:
    """The case the learner asked for, which must exist; None when they did not ask."""
    if case_id is None:
        return None
    for case in cases:
        if case.case_id == case_id:
            return case
    raise ContentError(
        ErrorCode.CONTENT_NOT_FOUND,
        f"no case {case_id}: a case is a sys-track topic with kind case and a debrief.",
    )


def _pick_case(
    cases: Sequence[_Case],
    history: Sequence[MockSession],
    weak: PartScore | None,
    asked: _Case | None,
) -> _Case | None:
    """A case covering the weakest recent part first, then the least recently used, then by id."""
    if asked is not None:
        return asked
    if not cases:
        return None
    last_used = _last_used(history, "case_id")

    def order(case: _Case) -> tuple[bool, datetime, str]:
        misses_weak = weak is not None and weak.part_id not in case.covers
        return (misses_weak, last_used.get(case.case_id, _EPOCH), case.case_id)

    return min(cases, key=order)


def _pick_coding(
    settings: Settings, snapshot: Snapshot, history: Sequence[MockSession]
) -> str | None:
    """The passed core problem used longest ago by a mock, then by id. None when nothing passed."""
    candidates = sorted(
        problem_id for problem_id in first_passes(snapshot) if _is_ready_core(settings, problem_id)
    )
    if not candidates:
        return None
    last_used = _last_used(history, "coding_problem_id")
    return min(candidates, key=lambda problem_id: (last_used.get(problem_id, _EPOCH), problem_id))


def _pick_theory(
    settings: Settings, snapshot: Snapshot, history: Sequence[MockSession]
) -> tuple[str, int] | None:
    """A theory question from a retained topic (solved ones only when none are retained)."""
    mastery = {topic_id: Mastery(value) for topic_id, value in snapshot.mastery.items()}
    retained = sorted(topic for topic, state in mastery.items() if state in _RETAINED)
    pool = retained or sorted(topic for topic, state in mastery.items() if state is Mastery.SOLVED)
    options = [(topic, _questions(settings, topic)) for topic in pool]
    options = [(topic, questions) for topic, questions in options if questions]
    if not options:
        return None
    last_used = _last_used(history, "theory_topic_id")
    topic, questions = min(options, key=lambda item: (last_used.get(item[0], _EPOCH), item[0]))
    asked = sum(1 for row in history if row.theory_topic_id == topic)
    return topic, questions[asked % len(questions)].number


def _last_used(history: Sequence[MockSession], column: str) -> dict[str, datetime]:
    """For each value of ``column``, the latest start time among mocks that used it."""
    last: dict[str, datetime] = {}
    for row in history:  # oldest first, so the newest start wins
        value = getattr(row, column)
        if value is not None:
            last[value] = row.started_at
    return last


def _cases(settings: Settings) -> list[_Case]:
    """Every sys-track case that is ready to use: verified or reviewed, with a debrief."""
    found: list[_Case] = []
    for topic_dir in discover_topic_dirs(settings.content_dir):
        if not _is_case(topic_dir):
            continue
        try:
            package = load_topic(topic_dir)
        except ContentError:
            continue
        case = package.case
        if (
            case is None
            or case.debrief is None
            or case.brief is None
            or package.meta.status not in READY_STATUSES
        ):
            continue
        covers = frozenset(section.heading.lower() for section in case.debrief.sections)
        found.append(
            _Case(
                case_id=package.meta.id,
                title=package.meta.title,
                covers=covers,
            )
        )
    return sorted(found, key=lambda case: case.case_id)


def _is_case(topic_dir: Path) -> bool:
    try:
        data = yaml.safe_load((topic_dir / "topic.yaml").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return False
    return isinstance(data, dict) and data.get("kind") == "case"


def _is_ready_core(settings: Settings, problem_id: str) -> bool:
    parts = problem_id.split("/")
    if len(parts) != 3:
        return False
    problem_dir = settings.content_dir.joinpath(parts[0], parts[1], "problems", parts[2])
    try:
        package = load_problem(problem_dir)
    except ContentError:
        return False
    meta = package.meta
    return meta is not None and meta.core and meta.status in READY_STATUSES


def _questions(settings: Settings, topic_id: str) -> tuple[InterviewQuestion, ...]:
    path = settings.content_dir / topic_id / "interview.md"
    if not path.is_file():
        return ()
    return parse_questions(path.read_text(encoding="utf-8"))


# --- rows, plans and reports ------------------------------------------------------------------


def _open_row(db: Session) -> MockSession | None:
    return db.scalars(
        select(MockSession).where(MockSession.status == MockStatus.OPEN.value)
    ).first()


def _recent_scores(db: Session, rubric: Rubric) -> list[Mapping[str, int]]:
    rows = db.scalars(
        select(Answer)
        .join(MockSession, Answer.mock_session_id == MockSession.id)
        .where(MockSession.status == MockStatus.SCORED.value)
        .order_by(MockSession.ended_at.desc(), MockSession.id.desc())
        .limit(RECENT_MOCKS)
    )
    return [dict(row.scores) for row in rows]


def _focus(db: Session, rubric: Rubric) -> PartScore | None:
    return weakest_over(rubric, _recent_scores(db, rubric))


def _plan(
    settings: Settings, row: MockSession, rubric: Rubric, focus: PartScore | None
) -> MockPlan:
    theory_title = None
    question_title = None
    if row.theory_topic_id is not None:
        theory_title = _topic_title(settings, row.theory_topic_id)
        for question in _questions(settings, row.theory_topic_id):
            if question.number == row.theory_question:
                question_title = question.title
    return MockPlan(
        session_id=row.id,
        started_at=row.started_at,
        case_id=row.case_id,
        case_title=None if row.case_id is None else _topic_title(settings, row.case_id),
        coding_problem_id=row.coding_problem_id,
        coding_title=None
        if row.coding_problem_id is None
        else _problem_title(settings, row.coding_problem_id),
        theory_topic_id=row.theory_topic_id,
        theory_title=theory_title,
        theory_question=row.theory_question,
        theory_question_title=question_title,
        part_minutes=tuple((part.title, part.minutes) for part in rubric.parts),
        focus=focus,
    )


def _report(session_id: str, rubric: Rubric, stored: Mapping[str, int]) -> MockReport:
    parts = part_scores(rubric, stored)
    total = total_of(parts)
    return MockReport(
        session_id=session_id,
        parts=parts,
        total=total,
        maximum=rubric.maximum,
        band=band_for(rubric, total),
        strongest=strongest_part(parts),
        weakest=weakest_part(parts),
    )


def _whole_score(value: object) -> int:
    """A checked score as an int. check_scores has already refused anything else."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise DogmaError(ErrorCode.MOCK_SCORE_INVALID, "every score must be a whole number.")
    return value


def _clean_notes(rubric: Rubric, notes: Mapping[str, str] | None) -> str | None:
    if not notes:
        return None
    lines: list[str] = []
    for key in rubric.criteria_ids:
        text = (notes.get(key) or "").strip()
        if len(text) > ITEM_NOTE_LIMIT:
            raise DogmaError(
                ErrorCode.MOCK_SCORE_INVALID,
                f"a note on {key} is over {ITEM_NOTE_LIMIT} characters.",
            )
        if text:
            lines.append(f"{key}: {text}")
    return "\n".join(lines) or None


def _case_reveals(settings: Settings, case_id: str) -> tuple[Reveal, ...]:
    try:
        package = load_topic(settings.content_dir / case_id)
    except ContentError:
        return ()
    if package.case is None or package.case.reveals is None:
        return ()
    return package.case.reveals


def _case_debrief_text(settings: Settings, case_id: str | None) -> str | None:
    if case_id is None:
        return None
    path = settings.content_dir / case_id / "debrief.md"
    return path.read_text(encoding="utf-8") if path.is_file() else None


def _topic_title(settings: Settings, topic_id: str) -> str:
    path = settings.content_dir / topic_id / "topic.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return topic_id
    return str(data.get("title", topic_id)) if isinstance(data, dict) else topic_id


def _problem_title(settings: Settings, problem_id: str) -> str:
    parts = problem_id.split("/")
    if len(parts) != 3:
        return problem_id
    try:
        package = load_problem(
            settings.content_dir.joinpath(parts[0], parts[1], "problems", parts[2])
        )
    except ContentError:
        return problem_id
    return package.meta.title if package.meta is not None else problem_id


def _rubric(settings: Settings) -> Rubric:
    return load_rubric_version(settings.curriculum_dir, MOCK_RUBRIC_ID, MOCK_RUBRIC_VERSION)


def _require_recording(settings: Settings) -> None:
    if not settings.record_attempts:
        raise DogmaError(
            ErrorCode.MOCK_RECORDING_OFF,
            "Recording is switched off (DOGMA_RECORD_ATTEMPTS=false), so nothing was stored.",
        )


def _no_open_message() -> str:
    return "No mock is open. Start one with dogma mock start."


def _aware(now: datetime) -> datetime:
    if now.tzinfo is None or now.utcoffset() is None:
        raise DogmaError(ErrorCode.SESSION_TIME_INVALID, "Mock times must be timezone-aware.")
    return now.astimezone(UTC)
