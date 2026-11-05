"""Interview sessions in the planner: one Consolidate a week from week 3, every session in week 14.

Low-energy sessions never become interviews. The planner is pure, so each rule is table-driven on
facts that the session layer would gather: the path week, the interviews already planned this
week, and the weakest part over the last three scored mocks.
"""

from __future__ import annotations

import pytest

from dogma.learning.planner import (
    Energy,
    PlanFacts,
    ProblemFacts,
    SessionKind,
    SessionPlan,
    TopicFacts,
    plan_session,
)

PER_DAY = 3


def _facts(
    *,
    path_week: int | None,
    interviews: int = 0,
    weak: str | None = None,
    weak_percent: int | None = None,
) -> PlanFacts:
    topic = TopicFacts(
        topic_id="tst/a",
        title="tst/a",
        status="reviewed",
        objective="Explain tst/a in one sentence",
        requires=(),
        problems=(
            ProblemFacts(problem_id="tst/a/p01-core", title="core", status="reviewed", core=True),
        ),
    )
    return PlanFacts(
        topics=(topic,),
        mastery={},
        due_count=4,
        familiar=(),
        path_week=path_week,
        interviews_this_week=interviews,
        weak_part=weak,
        weak_part_percent=weak_percent,
    )


def _plan(facts: PlanFacts, slot: int, energy: Energy = Energy.NORMAL) -> SessionPlan:
    return plan_session(facts, slot=slot, sessions_per_day=PER_DAY, energy=energy)


def test_the_session_kind_for_planned_interviews_is_stored_as_interview() -> None:
    assert SessionKind.INTERVIEW.value == "interview"


@pytest.mark.parametrize("week", [1, 2])
def test_before_week_three_the_consolidate_session_stays_a_consolidate(week: int) -> None:
    assert _plan(_facts(path_week=week), slot=3).kind is SessionKind.CONSOLIDATE


def test_from_week_three_the_first_consolidate_of_the_week_is_an_interview() -> None:
    result = _plan(_facts(path_week=3), slot=3)

    assert result.kind is SessionKind.INTERVIEW
    assert [(block.name, block.minutes) for block in result.blocks] == [
        ("Mock", 40),
        ("Debrief", 5),
    ]
    assert result.minutes == 45
    assert result.topic_id is None
    assert "dogma mock start" in result.first_action


def test_a_week_that_already_has_an_interview_keeps_its_other_consolidates() -> None:
    assert _plan(_facts(path_week=5, interviews=1), slot=3).kind is SessionKind.CONSOLIDATE


def test_learn_sessions_are_never_turned_into_interviews() -> None:
    for slot in (1, 2):
        assert _plan(_facts(path_week=6), slot=slot).kind is SessionKind.LEARN


def test_every_session_of_the_final_week_is_an_interview() -> None:
    for slot in (1, 2, 3):
        result = _plan(_facts(path_week=14, interviews=9), slot=slot)
        assert result.kind is SessionKind.INTERVIEW


@pytest.mark.parametrize("week", [None, 3, 14])
def test_low_energy_sessions_never_become_interviews(week: int | None) -> None:
    for slot in (1, 2, 3):
        result = _plan(_facts(path_week=week), slot=slot, energy=Energy.LOW)
        assert result.kind is SessionKind.MINIMUM


def test_a_day_outside_the_path_keeps_the_ordinary_plan() -> None:
    assert _plan(_facts(path_week=None), slot=3).kind is SessionKind.CONSOLIDATE


def test_the_weakest_part_is_named_in_the_interview_notes() -> None:
    result = _plan(_facts(path_week=4, weak="Case, structure", weak_percent=50), slot=3)

    assert any("Case, structure" in note and "50%" in note for note in result.notes)


def test_without_a_scored_mock_the_notes_say_so_plainly() -> None:
    result = _plan(_facts(path_week=4), slot=3)

    assert any("No scored mock" in note for note in result.notes)


def test_interview_plans_never_claim_streaks_or_trends() -> None:
    result = _plan(_facts(path_week=4, weak="Ask", weak_percent=62), slot=3)
    text = " ".join(note for note in result.notes) + " ".join(
        block.detail for block in result.blocks
    )

    for banned in ("streak", "trend", "backlog", "debt"):
        assert banned not in text.lower()
