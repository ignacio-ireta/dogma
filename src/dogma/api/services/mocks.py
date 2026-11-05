"""Mock interviews over HTTP: the same use cases as ``dogma mock``, mapped to the wire shapes.

Every rule (one open mock, the choice of case, the reveal match, the band) lives in
``dogma.learning.mocks`` and ``dogma.learning.mock_rules``. Reads and abandon check that the
database exists first, so asking about a mock never creates a learner database.
"""

from __future__ import annotations

from datetime import datetime

from dogma.api.ids import is_session_id
from dogma.api.schemas.mocks import (
    AbandonedOut,
    CaseOut,
    CodingOut,
    FocusOut,
    MockBandOut,
    MockDebriefOut,
    MockPlanOut,
    MockReportOut,
    MockStartRequest,
    MockStatusOut,
    PartScoreOut,
    PartTimingOut,
    RevealOut,
    TheoryOut,
)
from dogma.content.loader import load_topic
from dogma.core.config import Settings
from dogma.core.errors import ContentError, DogmaError, ErrorCode
from dogma.db.session import database_path
from dogma.learning import mocks
from dogma.learning.mock_rules import PartScore


def status(settings: Settings) -> MockStatusOut:
    """The open mock and the last scored one. Empty, not an error, before any mock exists."""
    view = mocks.mock_status(settings)
    return MockStatusOut(
        open=None if view.open is None else plan_out(settings, view.open),
        last_scored=None if view.last_scored is None else report_out(view.last_scored),
    )


def start(settings: Settings, body: MockStartRequest, *, now: datetime) -> MockPlanOut:
    return plan_out(settings, mocks.start_mock(settings, now=now, case_id=body.case_id))


def reveal(settings: Settings, question: str) -> RevealOut:
    _require_open(settings)
    result = mocks.reveal(settings, question)
    return RevealOut(text=result.text, matched=result.matched, exhibit=result.exhibit)


def score(settings: Settings, scores: dict[str, int], *, now: datetime) -> MockReportOut:
    return report_out(mocks.score_mock(settings, scores, now=now))


def debrief(settings: Settings, session_id: str) -> MockDebriefOut:
    if not is_session_id(session_id) or not database_path(settings).is_file():
        raise DogmaError(ErrorCode.MOCK_NOT_FOUND, "No scored mock has that id.")
    found = mocks.debrief(settings, session_id)
    return MockDebriefOut(
        report=report_out(found.report),
        case_id=found.case_id,
        case_debrief=found.case_debrief,
    )


def abandon(settings: Settings, *, now: datetime) -> AbandonedOut:
    _require_open(settings)
    session_id = mocks.abandon_mock(settings, now=now)
    return AbandonedOut(session_id=session_id, status="abandoned")


def plan_out(settings: Settings, plan: mocks.MockPlan) -> MockPlanOut:
    return MockPlanOut(
        session_id=plan.session_id,
        started_at=plan.started_at,
        case=_case_out(settings, plan),
        coding=None
        if plan.coding_problem_id is None
        else CodingOut(problem_id=plan.coding_problem_id, title=plan.coding_title or ""),
        theory=None
        if plan.theory_topic_id is None
        else TheoryOut(
            topic_id=plan.theory_topic_id,
            title=plan.theory_title or "",
            question=plan.theory_question,
            question_title=plan.theory_question_title,
        ),
        parts=[PartTimingOut(title=title, minutes=minutes) for title, minutes in plan.part_minutes],
        focus=None if plan.focus is None else _focus_out(plan.focus),
    )


def report_out(report: mocks.MockReport) -> MockReportOut:
    return MockReportOut(
        session_id=report.session_id,
        parts=[part_out(part) for part in report.parts],
        total=report.total,
        maximum=report.maximum,
        band=MockBandOut(
            label=report.band.label,
            low=report.band.low,
            high=report.band.high,
            text=report.band.text,
        ),
        strongest=part_out(report.strongest),
        weakest=part_out(report.weakest),
    )


def part_out(part: PartScore) -> PartScoreOut:
    return PartScoreOut(
        part_id=part.part_id,
        title=part.title,
        minutes=part.minutes,
        score=part.score,
        maximum=part.maximum,
        percent=round(part.percent * 100, 1),
    )


def _focus_out(part: PartScore) -> FocusOut:
    return FocusOut(part_id=part.part_id, title=part.title, percent=round(part.percent * 100, 1))


def _case_out(settings: Settings, plan: mocks.MockPlan) -> CaseOut | None:
    if plan.case_id is None:
        return None
    return CaseOut(
        id=plan.case_id,
        title=plan.case_title or plan.case_id,
        brief=_brief(settings, plan.case_id),
    )


def _brief(settings: Settings, case_id: str) -> str | None:
    """The candidate-facing brief. A case that no longer loads has no brief to show."""
    try:
        package = load_topic(settings.content_dir / case_id)
    except ContentError:
        return None
    return None if package.case is None else package.case.brief


def _require_open(settings: Settings) -> None:
    if not database_path(settings).is_file():
        raise DogmaError(
            ErrorCode.MOCK_NOT_OPEN, "No mock is open. Start one with dogma mock start."
        )
