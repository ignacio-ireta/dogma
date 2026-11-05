"""Mock interviews: start, reveal, score, debrief, abandon, and the one-open-mock rule.

Every test runs on a scratch copy of the curriculum, the example topic and the sys cases. Nothing
here reads or writes the learner's real var/ directory.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select
from tests.unit.db.support import PROBLEM, make_report

from dogma.content.rubrics import Rubric, load_rubric_version
from dogma.core.config import Settings
from dogma.core.errors import DogmaError, ErrorCode
from dogma.db.answer_models import Answer
from dogma.db.errors import StorageErrorCode
from dogma.db.mock_models import MockSession
from dogma.db.models import AttemptMode, Mastery, TopicState
from dogma.db.session import database_path, open_session
from dogma.db.store import record_attempt
from dogma.learning.mocks import (
    abandon_mock,
    debrief,
    recent_weak_part,
    reveal,
    score_mock,
    start_mock,
)

REPO = Path(__file__).resolve().parents[3]
TOPIC = "cml/vector-operations"
CHURN = "sys/churn-prediction-case"
FRAUD = "sys/fraud-detection-case"
RASS = "sys/rass-case-framework"
T0 = datetime(2026, 11, 9, 9, tzinfo=UTC)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    root = tmp_path / "repo"
    shutil.copytree(REPO / "curriculum", root / "curriculum")
    shutil.copytree(REPO / "content" / "cml" / "vector-operations", root / "content" / TOPIC)
    shutil.copytree(REPO / "content" / "sys", root / "content" / "sys")
    return Settings(repo_root=root, _env_file=None)


@pytest.fixture
def rubric(settings: Settings) -> Rubric:
    return load_rubric_version(settings.curriculum_dir, "mock", 1)


@pytest.fixture
def ready(settings: Settings) -> Settings:
    """A learner who has passed the example core problem and retained its topic."""
    record_attempt(
        settings,
        make_report(public=(3, 3), hidden=(2, 2), attempt_id="a" * 32),
        mode=AttemptMode.SUBMIT,
        core=True,
        started_at=T0 - timedelta(days=20, seconds=1),
        submitted_at=T0 - timedelta(days=20),
    )
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        state = session.get(TopicState, TOPIC)  # the passing submit above made it solved
        assert state is not None
        state.mastery = Mastery.RETAINED.value
        state.retained_at = T0 - timedelta(days=10)
    return settings


def _all(rubric: Rubric, score: int = 3, **parts: int) -> dict[str, int]:
    """Every item at ``score``, except the parts named in ``parts`` (by part id) at their value."""
    scores = dict.fromkeys(rubric.criteria_ids, score)
    for part in rubric.parts:
        if part.id in parts:
            for item in part.items:
                scores[item.id] = parts[part.id]
    return scores


@dataclass(frozen=True)
class _MockRow:
    id: str
    status: str
    started_at: datetime
    ended_at: datetime | None


def _mock_rows(settings: Settings) -> list[_MockRow]:
    """Plain copies of the mock rows, so the assertions do not depend on a live session."""
    if not database_path(settings).is_file():
        return []
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        return [
            _MockRow(row.id, row.status, row.started_at, row.ended_at)
            for row in session.scalars(select(MockSession).order_by(MockSession.started_at))
        ]


def _answer_count(settings: Settings) -> int:
    if not database_path(settings).is_file():
        return 0
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        return session.scalar(select(func.count()).select_from(Answer)) or 0


def _code(error: DogmaError) -> ErrorCode:
    return error.code


# --- start ----------------------------------------------------------------------------------


def test_start_picks_a_case_a_passed_core_problem_and_a_retained_theory_question(
    ready: Settings,
) -> None:
    plan = start_mock(ready, now=T0)

    assert plan.case_id == CHURN  # no history: ties go to the smallest id
    assert plan.coding_problem_id == PROBLEM
    assert plan.theory_topic_id == TOPIC
    assert plan.theory_question == 1
    assert [name for name, _ in plan.part_minutes] == [
        "Live coding",
        "Case, reframe",
        "Case, questions",
        "Case, structure",
        "Case, synthesis",
        "Overall communication",
    ]
    assert dict(plan.part_minutes)["Live coding"] == "10-15"
    [row] = _mock_rows(ready)
    assert row.status == "open"
    assert row.ended_at is None
    assert row.started_at == T0


def test_only_one_mock_can_be_open_at_a_time(ready: Settings) -> None:
    start_mock(ready, now=T0)

    with pytest.raises(DogmaError) as caught:
        start_mock(ready, now=T0 + timedelta(minutes=1))

    assert _code(caught.value) is ErrorCode.MOCK_ALREADY_OPEN
    assert len(_mock_rows(ready)) == 1


def test_recording_off_refuses_to_start_and_writes_nothing(ready: Settings) -> None:
    off = Settings(repo_root=ready.repo_root, _env_file=None, record_attempts=False)

    with pytest.raises(DogmaError) as caught:
        start_mock(off, now=T0)

    assert _code(caught.value) is ErrorCode.MOCK_RECORDING_OFF
    assert _mock_rows(ready) == []


def test_the_least_recently_used_case_comes_next_after_an_abandoned_mock(ready: Settings) -> None:
    first = start_mock(ready, now=T0)
    abandon_mock(ready, now=T0 + timedelta(minutes=5))

    second = start_mock(ready, now=T0 + timedelta(minutes=10))

    assert first.case_id == CHURN
    assert second.case_id == FRAUD


def test_a_case_covering_the_weakest_recent_part_is_preferred(
    ready: Settings, rubric: Rubric
) -> None:
    # Only churn and RASS keep their "Ask" section, so only they cover that part. Removing it from
    # every other case keeps this rule independent of how many cases the sys track holds.
    for case_dir in (ready.content_dir / "sys").iterdir():
        if f"sys/{case_dir.name}" in {CHURN, RASS}:
            continue
        debrief_path = case_dir / "debrief.md"
        text = debrief_path.read_text(encoding="utf-8")
        debrief_path.write_text(
            "\n".join(line for line in text.splitlines() if not line.startswith("## Ask")),
            encoding="utf-8",
        )
    for _ in range(3):  # three scored mocks, all weak on the ask part
        start_mock(ready, now=T0)
        score_mock(ready, _all(rubric, 4, ask=1), now=T0 + timedelta(minutes=1))

    plan = start_mock(ready, now=T0 + timedelta(hours=1))

    assert plan.case_id in {CHURN, RASS}
    assert plan.case_id != FRAUD


def test_a_learner_without_a_passed_core_problem_still_starts_without_one(
    settings: Settings,
) -> None:
    plan = start_mock(settings, now=T0)

    assert plan.coding_problem_id is None
    assert plan.theory_topic_id is None
    assert plan.case_id == CHURN


# --- reveal ---------------------------------------------------------------------------------


def test_reveal_releases_the_matching_fact_only_during_an_open_mock(ready: Settings) -> None:
    with pytest.raises(DogmaError) as caught:
        reveal(ready, "What does the retention process do?")
    assert _code(caught.value) is ErrorCode.MOCK_NOT_OPEN

    start_mock(ready, now=T0, case_id=CHURN)

    found = reveal(ready, "What does the current retention process do?")
    nothing = reveal(ready, "What is the weather in Lisbon?")

    assert "20 percent discount" in found.text
    assert nothing.text == "The interviewer has nothing on that."
    assert _answer_count(ready) == 0


# --- score ----------------------------------------------------------------------------------


def test_score_stores_thirty_items_as_a_linked_mock_answer(ready: Settings, rubric: Rubric) -> None:
    plan = start_mock(ready, now=T0, case_id=CHURN)

    report = score_mock(ready, _all(rubric, 3), now=T0 + timedelta(minutes=45))

    assert report.total == 90
    assert report.maximum == 120
    with open_session(ready, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        [answer] = session.scalars(select(Answer))
        mock = session.get(MockSession, plan.session_id)
        assert mock is not None
        found = (
            answer.mock_session_id,
            answer.kind,
            answer.topic_id,
            answer.rubric_id,
            answer.rubric_version,
            dict(answer.scores),
            mock.status,
            mock.ended_at,
        )
    assert found == (
        plan.session_id,
        "mock",
        None,
        "mock",
        1,
        _all(rubric, 3),
        "scored",
        T0 + timedelta(minutes=45),
    )


def test_a_bad_score_stores_nothing_and_keeps_the_mock_open(
    ready: Settings, rubric: Rubric
) -> None:
    start_mock(ready, now=T0)
    scores = _all(rubric, 3)
    scores["coding.speed"] = 5
    del scores["ask.focused"]
    scores["made.up"] = 2

    with pytest.raises(DogmaError) as caught:
        score_mock(ready, scores, now=T0 + timedelta(minutes=1))

    assert _code(caught.value) is ErrorCode.MOCK_SCORE_INVALID
    assert "ask.focused is missing" in caught.value.message
    assert _answer_count(ready) == 0
    [row] = _mock_rows(ready)
    assert row.status == "open"


def test_scoring_without_an_open_mock_is_refused(ready: Settings, rubric: Rubric) -> None:
    with pytest.raises(DogmaError) as caught:
        score_mock(ready, _all(rubric), now=T0)

    assert _code(caught.value) is ErrorCode.MOCK_NOT_OPEN


# --- debrief --------------------------------------------------------------------------------


def test_debrief_gives_part_totals_band_strength_weakness_and_the_case_debrief(
    ready: Settings, rubric: Rubric
) -> None:
    start_mock(ready, now=T0, case_id=CHURN)
    score_mock(ready, _all(rubric, 4, structure=2, synthesize=4), now=T0 + timedelta(minutes=40))

    view = debrief(ready)

    by_title = {part.title: part for part in view.report.parts}
    # Structure has seven items, all scored 2 here: 14 of 28.
    assert by_title["Case, structure"].score == 14
    assert by_title["Case, structure"].maximum == 28
    assert view.report.total == 120 - 14
    assert view.report.maximum == 120
    assert view.report.weakest.part_id == "structure"
    assert view.report.strongest.part_id in {
        "coding",
        "reframe",
        "ask",
        "synthesize",
        "communication",
    }
    assert view.report.band.label
    assert view.case_debrief is not None
    assert "## Synthesize" in view.case_debrief


def test_debrief_without_any_scored_mock_is_not_found(ready: Settings) -> None:
    start_mock(ready, now=T0)

    with pytest.raises(DogmaError) as caught:
        debrief(ready)

    assert _code(caught.value) is ErrorCode.MOCK_NOT_FOUND


def test_debrief_shows_the_last_scored_mock_not_an_open_one(
    ready: Settings, rubric: Rubric
) -> None:
    start_mock(ready, now=T0)
    score_mock(ready, _all(rubric, 2), now=T0 + timedelta(minutes=1))
    start_mock(ready, now=T0 + timedelta(minutes=2))

    assert debrief(ready).report.total == 60


# --- abandon and the weakest recent part ----------------------------------------------------


def test_abandon_closes_the_mock_without_scores_and_frees_the_slot(ready: Settings) -> None:
    start_mock(ready, now=T0)

    abandon_mock(ready, now=T0 + timedelta(minutes=3))

    [row] = _mock_rows(ready)
    assert row.status == "abandoned"
    assert _answer_count(ready) == 0
    assert start_mock(ready, now=T0 + timedelta(minutes=4)).session_id != row.id


def test_abandon_with_no_open_mock_is_refused(ready: Settings) -> None:
    with pytest.raises(DogmaError) as caught:
        abandon_mock(ready, now=T0)

    assert _code(caught.value) is ErrorCode.MOCK_NOT_OPEN


def test_the_weakest_part_looks_at_the_last_three_scored_mocks_only(
    ready: Settings, rubric: Rubric
) -> None:
    assert recent_weak_part(ready) is None
    for index, scores in enumerate(
        [_all(rubric, 1, coding=4), _all(rubric, 4, ask=2), _all(rubric, 4, ask=2)]
    ):
        start_mock(ready, now=T0 + timedelta(hours=index))
        score_mock(ready, scores, now=T0 + timedelta(hours=index, minutes=30))
    # The oldest mock was weak on coding, but it is no longer among the last three.
    start_mock(ready, now=T0 + timedelta(hours=5))
    score_mock(ready, _all(rubric, 4, ask=2), now=T0 + timedelta(hours=5, minutes=30))

    weak = recent_weak_part(ready)

    assert weak is not None
    assert weak.part_id == "ask"
